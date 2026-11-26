"""Phase 3: Diffusion-based counterfactual explanations (unconstrained & tumor-mask-constrained).

Generates "what-if" MRIs that flip a trained classifier's prediction from --source_class to
--target_class while keeping the rest of the scan as close to the original as possible.

Examples:
    # Run both unconstrained and mask-constrained on 20 validation meningiomas -> glioma
    python scripts/counterfactual.py --cache cache/cache_512.npz --ckpt checkpoints/<B1_run>/best.pt --out results/cf_men_to_gli

    # With Swasti's LoRA weights and Person 2's EfficientNet-B0 independent judge
    python scripts/counterfactual.py --cache cache/cache_512.npz --ckpt checkpoints/<B1_run>/best.pt --judge_ckpt checkpoints/<EffNet_run>/best.pt --lora_dir checkpoints/<lora_dir> --out results/cf_men_to_gli
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from diffusers import DDIMScheduler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainmri import CLASS_NAMES  # noqa: E402
from brainmri.data import Cache, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from brainmri.diffusion import (  # noqa: E402
    DEFAULT_MODEL_ID,
    PROMPT_TEMPLATE,
    class_prompts,
    encode_prompts,
    gray_uint8_to_sd,
    load_pipelines,
)
from brainmri.models import build_model  # noqa: E402
from brainmri.utils import pick_device, seed_everything  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True, help="512px cache (.npz)")
    p.add_argument("--ckpt", type=Path, required=True, help="Primary B1 classifier checkpoint (best.pt)")
    p.add_argument("--judge_ckpt", type=Path, default=None, help="Optional independent classifier checkpoint (e.g. EfficientNet-B0)")
    p.add_argument("--lora_dir", type=Path, default=None, help="Fine-tuned SD 1.5 LoRA directory (uses base SD 1.5 if omitted)")
    p.add_argument("--out", type=Path, required=True, help="Output folder for CSVs, .npz, and figure grid")
    p.add_argument("--split", choices=["val", "train"], default="val", help="Split to sample from (never test)")
    p.add_argument("--source_class", choices=CLASS_NAMES, default="meningioma")
    p.add_argument("--target_class", choices=CLASS_NAMES, default="glioma")
    p.add_argument("--mode", choices=["both", "unconstrained", "constrained"], default="both")
    p.add_argument("--n_samples", type=int, default=20, help="Number of source images to process")
    p.add_argument("--n_grid", type=int, default=8, help="Number of examples in the visual figure grid")
    p.add_argument("--strength", type=float, default=0.5, help="img2img noise strength in (0, 1]")
    p.add_argument("--steps", type=int, default=30, help="Total DDIM inference steps")
    p.add_argument("--guidance", type=float, default=5.0, help="Classifier-free guidance scale")
    p.add_argument("--cls_scale", type=float, default=15.0, help="Classifier guidance gradient scale")
    p.add_argument("--dilate_px", type=int, default=12, help="Mask dilation radius in pixels (at 512px)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--prompt_template", default=PROMPT_TEMPLATE)
    p.add_argument("--min_gpu_gb", type=float, default=0.0)
    return p.parse_args()


def load_classifier(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["model"], pretrained=False).to(device).eval()
    model.load_state_dict(ck["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, int(cfg.get("img_size", 224))


def preprocess_for_clf(img_01: torch.Tensor, img_size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """(B, 1 or 3, H, W) float in [0, 1] -> (B, 3, img_size, img_size) ImageNet-normalized."""
    if img_01.shape[1] == 1:
        img_01 = img_01.expand(-1, 3, -1, -1)
    x = F.interpolate(img_01.float(), size=(img_size, img_size), mode="bilinear", antialias=True, align_corners=False)
    return (x - mean) / std


@torch.no_grad()
def predict_probs(model, img_01: torch.Tensor, img_size: int, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    x = preprocess_for_clf(img_01, img_size, mean, std)
    logits = model(x)
    return torch.softmax(logits, dim=1).cpu().numpy()


def dilate_mask(mask_512: np.ndarray, dilate_px: int) -> torch.Tensor:
    """(H, W) uint8 {0,1} -> (1, 1, H, W) float32 dilated binary mask."""
    m = torch.from_numpy((mask_512 > 0).astype(np.float32))[None, None]
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=dilate_px)
    return m


def generate_one_counterfactual(
    img_uint8: np.ndarray,
    mask_512: torch.Tensor,
    target_idx: int,
    t2i,
    scheduler: DDIMScheduler,
    cond_emb: torch.Tensor,
    uncond_emb: torch.Tensor,
    clf,
    clf_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    strength: float,
    steps: int,
    guidance: float,
    cls_scale: float,
    constrained: bool,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Run DDIM img2img with classifier guidance and optional RePaint-style mask resetting."""
    vae, unet = t2i.vae, t2i.unet
    dtype = unet.dtype

    # Encode original image to latent z_orig
    x_sd = gray_uint8_to_sd(img_uint8[None]).to(device, vae.dtype)
    with torch.no_grad():
        z_orig = (vae.encode(x_sd).latent_dist.sample() * vae.config.scaling_factor).to(dtype)

    scheduler.set_timesteps(steps, device=device)
    init_timestep = min(int(steps * strength), steps)
    t_start = max(steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(z_orig.shape, generator=gen, device=device, dtype=dtype)
    latents = scheduler.add_noise(z_orig, noise, timesteps[0].unsqueeze(0)) if len(timesteps) > 0 else z_orig.clone()

    lat_mask = F.interpolate(mask_512.to(device, dtype), size=z_orig.shape[-2:], mode="nearest")
    text_emb = torch.cat([uncond_emb, cond_emb], dim=0)

    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    for i, t in enumerate(timesteps):
        # 1. UNet noise prediction with classifier-free guidance (no grad to save VRAM)
        with torch.no_grad():
            latent_in = torch.cat([latents, latents], dim=0)
            latent_in = scheduler.scale_model_input(latent_in, t)
            eps_uncond, eps_cond = unet(latent_in, t, encoder_hidden_states=text_emb).sample.chunk(2)
            noise_pred = eps_uncond + guidance * (eps_cond - eps_uncond)

        # 2. Classifier guidance via one-step clean latent estimate z0_hat
        alpha_t = alphas_cumprod[t]
        sqrt_alpha = alpha_t.sqrt()
        sqrt_one_minus_alpha = (1.0 - alpha_t).sqrt()

        if cls_scale > 0:
            lat_g = latents.detach().float().requires_grad_(True)
            z0_hat = (lat_g - sqrt_one_minus_alpha.float() * noise_pred.detach().float()) / sqrt_alpha.float()
            img_dec = vae.decode(z0_hat / vae.config.scaling_factor, return_dict=False)[0]
            gray_01 = ((img_dec.clamp(-1, 1) + 1.0) / 2.0).mean(dim=1, keepdim=True)
            clf_in = preprocess_for_clf(gray_01, clf_size, mean, std)
            log_prob = torch.log_softmax(clf(clf_in), dim=1)[0, target_idx]
            grad = torch.autograd.grad(log_prob, lat_g)[0]
            if torch.isfinite(grad).all():
                grad = grad / (grad.norm() + 1e-8)
                # Nudge noise_pred toward higher target class probability
                noise_pred = noise_pred - cls_scale * sqrt_one_minus_alpha * grad.to(dtype)

        # 3. DDIM step
        with torch.no_grad():
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # 4. RePaint-style mask constraint: reset outside dilated tumor mask
            if constrained:
                if i < len(timesteps) - 1:
                    t_next = timesteps[i + 1].unsqueeze(0)
                    orig_step = scheduler.add_noise(z_orig, noise, t_next)
                else:
                    orig_step = z_orig
                latents = lat_mask * latents + (1.0 - lat_mask) * orig_step

    with torch.no_grad():
        decoded = vae.decode(latents.float() / vae.config.scaling_factor, return_dict=False)[0]
        gray = ((decoded.float().clamp(-1, 1) + 1.0) / 2.0).mean(dim=1).clamp(0, 1)
    return (gray[0] * 255.0).round().byte().cpu().numpy()


def compute_outside_ssim(orig_01: torch.Tensor, cf_01: torch.Tensor, mask_512: torch.Tensor) -> float:
    """Compute SSIM on the image with the tumor mask zeroed out in both images."""
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

    outside = (1.0 - mask_512.to(orig_01.device))
    return float(ssim_fn(cf_01 * outside, orig_01 * outside, data_range=1.0).item())


def save_comparison_grid(records: list[dict], out_path: Path, n_grid: int, source_cls: str, target_cls: str):
    """Save figure grid: Original | Unconstrained CF | Constrained CF | |Diff| (Constrained) | Tumor Mask."""
    rows = min(n_grid, len(records))
    if rows == 0:
        return
    has_both = "cf_unconstrained" in records[0] and "cf_constrained" in records[0]
    cols = 5 if has_both else 4
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.7 * rows), facecolor="#fcfcfb")
    if rows == 1:
        axes = axes[None, :]

    for r in range(rows):
        rec = records[r]
        orig = rec["orig"]
        mask = rec["mask"]
        if has_both:
            cf_u, cf_c = rec["cf_unconstrained"], rec["cf_constrained"]
            diff = np.abs(cf_c.astype(np.float32) - orig.astype(np.float32))
            panels = [
                (orig, "gray", f"Orig ({rec['orig_pred']})", (0, 255)),
                (cf_u, "gray", f"Unconstr ({rec['pred_unconstrained']})", (0, 255)),
                (cf_c, "gray", f"Constr ({rec['pred_constrained']})", (0, 255)),
                (diff, "inferno", "|Diff| Constr", (0, max(30.0, float(diff.max())))),
                (mask * 255, "gray", "Dilated Mask", (0, 255)),
            ]
        else:
            mode = "unconstrained" if "cf_unconstrained" in rec else "constrained"
            cf = rec[f"cf_{mode}"]
            diff = np.abs(cf.astype(np.float32) - orig.astype(np.float32))
            panels = [
                (orig, "gray", f"Orig ({rec['orig_pred']})", (0, 255)),
                (cf, "gray", f"{mode.title()} ({rec[f'pred_{mode}']})", (0, 255)),
                (diff, "inferno", "|Diff|", (0, max(30.0, float(diff.max())))),
                (mask * 255, "gray", "Dilated Mask", (0, 255)),
            ]
        for c, (img, cmap, title, (vmin, vmax)) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    fig.suptitle(f"Counterfactuals: {source_cls} -> {target_cls}", fontsize=11, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    a = parse_args()
    seed_everything(a.seed)
    device = pick_device(a.min_gpu_gb)
    if a.source_class == a.target_class:
        raise ValueError("--source_class and --target_class must differ.")

    src_idx_cls = CLASS_NAMES.index(a.source_class)
    tgt_idx_cls = CLASS_NAMES.index(a.target_class)
    modes = ["unconstrained", "constrained"] if a.mode == "both" else [a.mode]

    # ---- Load cache and pick source images from split
    cache = Cache(a.cache)
    split_indices = cache.indices(a.split)
    cand = [int(i) for i in split_indices if int(cache.labels[i]) == src_idx_cls]
    if not cand:
        raise RuntimeError(f"No {a.source_class} images found in split '{a.split}'.")
    selected = cand[: a.n_samples]

    # ---- Load classifiers
    clf, clf_size = load_classifier(a.ckpt, device)
    judge, judge_size = load_classifier(a.judge_ckpt, device) if a.judge_ckpt else (None, 0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    # ---- Load SD 1.5 + LoRA (memory-friendly for 10GB+ GPUs)
    lora_arg = str(a.lora_dir) if a.lora_dir else None
    t2i, _ = load_pipelines(a.model_id, lora_arg, device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    t2i.enable_attention_slicing()
    t2i.vae.to(torch.float32)  # prevents SD 1.5 fp16 VAE NaN / solid-black overflow
    t2i.vae.enable_slicing()
    for mod in (t2i.vae, t2i.unet, t2i.text_encoder):
        mod.requires_grad_(False)
    scheduler = DDIMScheduler.from_config(t2i.scheduler.config)

    prompts = class_prompts(a.prompt_template)
    cond_emb = encode_prompts(t2i.tokenizer, t2i.text_encoder, [prompts[tgt_idx_cls]], device).to(t2i.unet.dtype)
    uncond_emb = encode_prompts(t2i.tokenizer, t2i.text_encoder, [""], device).to(t2i.unet.dtype)

    # ---- Perceptual metrics
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    fid_by_mode = {m: FrechetInceptionDistance(feature=2048, normalize=True).to(device) for m in modes}

    a.out.mkdir(parents=True, exist_ok=True)
    per_image_rows = []
    grid_records = []
    saved_arrays = {m: [] for m in modes}

    for k, cache_idx in enumerate(selected):
        orig_u8 = cache.images[cache_idx]
        dil_mask = dilate_mask(cache.masks[cache_idx], a.dilate_px)  # (1, 1, 512, 512)
        orig_01 = torch.from_numpy(orig_u8.astype(np.float32) / 255.0)[None, None].to(device)

        p_orig = predict_probs(clf, orig_01, clf_size, mean, std)[0]
        orig_pred = CLASS_NAMES[int(np.argmax(p_orig))]
        j_orig = predict_probs(judge, orig_01, judge_size, mean, std)[0] if judge else None
        judge_orig_pred = CLASS_NAMES[int(np.argmax(j_orig))] if j_orig is not None else ""

        rec = {"orig": orig_u8, "mask": dil_mask[0, 0].numpy(), "orig_pred": orig_pred}

        for m in modes:
            cf_u8 = generate_one_counterfactual(
                orig_u8,
                dil_mask,
                tgt_idx_cls,
                t2i,
                scheduler,
                cond_emb,
                uncond_emb,
                clf,
                clf_size,
                mean,
                std,
                a.strength,
                a.steps,
                a.guidance,
                a.cls_scale,
                constrained=(m == "constrained"),
                seed=a.seed + k,
                device=device,
            )
            saved_arrays[m].append(cf_u8)
            cf_01 = torch.from_numpy(cf_u8.astype(np.float32) / 255.0)[None, None].to(device)

            p_cf = predict_probs(clf, cf_01, clf_size, mean, std)[0]
            cf_pred = CLASS_NAMES[int(np.argmax(p_cf))]
            flipped_b1 = int(cf_pred == a.target_class)

            if judge is not None:
                j_cf = predict_probs(judge, cf_01, judge_size, mean, std)[0]
                judge_cf_pred = CLASS_NAMES[int(np.argmax(j_cf))]
                flipped_judge = int(judge_cf_pred == a.target_class)
            else:
                judge_cf_pred, flipped_judge = "", ""

            diff = (cf_01 - orig_01).abs()
            l1 = float(diff.mean().item())
            m_dev = dil_mask.to(device)
            pct_in_mask = float((diff * m_dev).sum().item() / (diff.sum().item() + 1e-8))
            ssim_out = compute_outside_ssim(orig_01, cf_01, dil_mask)
            with torch.no_grad():
                lpips_val = float(lpips_fn(cf_01.expand(-1, 3, -1, -1), orig_01.expand(-1, 3, -1, -1)).item())
                fid_by_mode[m].update(orig_01.expand(-1, 3, -1, -1), real=True)
                fid_by_mode[m].update(cf_01.expand(-1, 3, -1, -1), real=False)

            rec[f"cf_{m}"] = cf_u8
            rec[f"pred_{m}"] = cf_pred

            per_image_rows.append(
                {
                    "cache_idx": cache_idx,
                    "path": cache.paths[cache_idx],
                    "patient_id": cache.patient_ids[cache_idx],
                    "mode": m,
                    "orig_pred": orig_pred,
                    "orig_prob_target": round(float(p_orig[tgt_idx_cls]), 4),
                    "cf_pred": cf_pred,
                    "cf_prob_target": round(float(p_cf[tgt_idx_cls]), 4),
                    "flipped_b1": flipped_b1,
                    "judge_orig_pred": judge_orig_pred,
                    "judge_cf_pred": judge_cf_pred,
                    "flipped_judge": flipped_judge,
                    "l1": round(l1, 5),
                    "lpips": round(lpips_val, 5),
                    "ssim_outside": round(ssim_out, 5),
                    "pct_change_in_mask": round(pct_in_mask, 5),
                }
            )
            print(f"[{k + 1}/{len(selected)}] {m:13s} | {orig_pred} -> {cf_pred} | L1={l1:.4f} LPIPS={lpips_val:.4f} SSIM_out={ssim_out:.4f} in_mask={pct_in_mask:.2%}")

        if len(grid_records) < a.n_grid:
            grid_records.append(rec)

    # ---- Write per-image CSV
    csv_path = a.out / "metrics_per_image.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader()
        w.writerows(per_image_rows)

    # ---- Write summary CSV
    summary_rows = []
    for m in modes:
        sub = [r for r in per_image_rows if r["mode"] == m]
        fid_val = round(float(fid_by_mode[m].compute().item()), 3) if len(sub) >= 2 else ""
        judge_flips = [r["flipped_judge"] for r in sub if r["flipped_judge"] != ""]
        summary_rows.append(
            {
                "mode": m,
                "n": len(sub),
                "flip_rate_b1": round(float(np.mean([r["flipped_b1"] for r in sub])), 4),
                "flip_rate_judge": round(float(np.mean(judge_flips)), 4) if judge_flips else "",
                "mean_l1": round(float(np.mean([r["l1"] for r in sub])), 5),
                "mean_lpips": round(float(np.mean([r["lpips"] for r in sub])), 5),
                "mean_ssim_outside": round(float(np.mean([r["ssim_outside"] for r in sub])), 5),
                "mean_pct_change_in_mask": round(float(np.mean([r["pct_change_in_mask"] for r in sub])), 5),
                "fid": fid_val,
            }
        )
    sum_path = a.out / "metrics_summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # ---- Save generated arrays (.npz for Grad-CAM IoU later) & figure grid
    for m in modes:
        np.savez_compressed(
            a.out / f"counterfactuals_{m}.npz",
            images=np.stack(saved_arrays[m]),
            orig_images=cache.images[selected],
            masks=cache.masks[selected],
            indices=np.array(selected),
            paths=cache.paths[selected],
        )
    save_comparison_grid(grid_records, a.out / "counterfactual_grid.png", a.n_grid, a.source_class, a.target_class)
    (a.out / "config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, indent=2)
    )
    print(f"Done! Saved metrics to {sum_path} and grid to {a.out / 'counterfactual_grid.png'}")


if __name__ == "__main__":
    main()
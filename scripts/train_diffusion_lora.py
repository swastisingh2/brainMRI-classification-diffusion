"""Fine-tune Stable Diffusion 1.5 with LoRA on TRAINING-split MRIs, conditioned on class via the text prompt.

Team rule: the generator only ever sees training-split images (optionally a subset of training groups
for low-data experiments, via --train_groups_file). Validation and test images are never loaded.

Speed: VAE latents (plus horizontally flipped copies) and the class prompt embeddings are computed once,
so each step only runs the UNet.

Writes <out_dir>/<run_name>/:
    config.json, train.log, loss.csv
    lora_step<N>/   LoRA weights (a few MB) every --save_every steps, and lora_final/
    samples_step<N>.png  sample grid per class (fixed seeds) every --sample_every steps
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainmri import CLASS_NAMES, SPLIT_VERSION  # noqa: E402
from brainmri.data import Cache  # noqa: E402
from brainmri.diffusion import (DEFAULT_MODEL_ID, PROMPT_TEMPLATE, class_prompts, encode_prompts,  # noqa: E402
                                gray_uint8_to_sd, sd_to_gray_uint8)
from brainmri.utils import code_version, device_name, pick_device, seed_everything, slurm_job_id  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True, help="cache built at the training resolution (e.g. 512)")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--latent_cache", type=Path, default=None, help="where to store/reuse precomputed latents")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--prompt_template", default=PROMPT_TEMPLATE)
    p.add_argument("--train_groups_file", type=Path, default=None,
                   help="optional text file of split_group IDs (one per line) to restrict training data")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--drop_prompt_prob", type=float, default=0.1,
                   help="train with an empty prompt this often, so classifier-free guidance works")
    p.add_argument("--snr_gamma", type=float, default=5.0, help="min-SNR loss weighting (0 = off)")
    p.add_argument("--no_flip", action="store_true")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--n_sample_per_class", type=int, default=6)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_gpu_gb", type=float, default=0.0)
    p.add_argument("--tag", default="")
    p.add_argument("--smoke", action="store_true", help="few steps, to test the pipeline")
    return p.parse_args()


@torch.no_grad()
def precompute_latents(vae, images_u8: np.ndarray, device, flip: bool, bs: int = 16) -> torch.Tensor:
    """Returns (N, [2,] 4, h, w) scaled latent means in fp16 on CPU (index 1 = horizontally flipped)."""
    out = []
    for i in range(0, len(images_u8), bs):
        x = gray_uint8_to_sd(images_u8[i:i + bs]).to(device, torch.float32)
        views = [x, torch.flip(x, dims=[3])] if flip else [x]
        lat = [vae.encode(v).latent_dist.mean * vae.config.scaling_factor for v in views]
        out.append(torch.stack(lat, 1).half().cpu())
    return torch.cat(out)


def sample_grid(pipe_cls, components, prompts, n, guidance, seed, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pipe = pipe_cls(**components)
    pipe.set_progress_bar_config(disable=True)
    fig, axes = plt.subplots(len(prompts), n, figsize=(2.0 * n, 2.1 * len(prompts)))
    for k, prompt in enumerate(prompts):
        g = torch.Generator(device=pipe.device).manual_seed(seed + k)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            imgs = pipe([prompt] * n, num_inference_steps=30, guidance_scale=guidance, generator=g,
                        output_type="pt").images
        gray = sd_to_gray_uint8(imgs)
        for j in range(n):
            ax = axes[k, j]
            ax.imshow(gray[j], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if j == 0:
                ax.set_title(CLASS_NAMES[k], fontsize=9, loc="left")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    a = parse_args()
    if a.smoke:
        a.steps, a.save_every, a.sample_every, a.n_sample_per_class, a.warmup_steps = 30, 30, 30, 2, 5
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.training_utils import cast_training_params, compute_snr
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    name = f"lora_r{a.rank}_seed{a.seed}" + (f"_{a.tag}" if a.tag else "") + ("_smoke" if a.smoke else "")
    name += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    run = a.out_dir / name
    run.mkdir(parents=True)
    log = logging.getLogger("lora")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(run / "train.log"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(h)

    seed_everything(a.seed)
    device = pick_device(a.min_gpu_gb)
    if device.type != "cuda":
        raise RuntimeError("LoRA training needs a CUDA GPU.")

    # ---- training images: train split only (optionally restricted to listed groups)
    cache = Cache(a.cache)
    idx = cache.indices("train")
    if a.train_groups_file:
        keep = {g.strip() for g in a.train_groups_file.read_text().split() if g.strip()}
        idx = idx[np.isin(cache.groups[idx], list(keep))]
    assert set(cache.split[idx]) == {"train"}, "generator must only see training images"
    labels = torch.from_numpy(cache.labels[idx])
    res = cache.images.shape[-1]

    # ---- frozen components
    tok = CLIPTokenizer.from_pretrained(a.model_id, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(a.model_id, subfolder="text_encoder", variant="fp16",
                                       torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained(a.model_id, subfolder="vae", variant="fp16").to(device, torch.float32)
    unet = UNet2DConditionModel.from_pretrained(a.model_id, subfolder="unet", variant="fp16",
                                                torch_dtype=torch.bfloat16).to(device)
    sched = DDPMScheduler.from_pretrained(a.model_id, subfolder="scheduler")
    for m in (te, vae, unet):
        m.requires_grad_(False)

    prompts = class_prompts(a.prompt_template)
    emb = encode_prompts(tok, te, prompts + [""], device).to(torch.bfloat16)  # rows 0..2 classes, 3 = empty

    # ---- latents (cached on disk, keyed by the exact training images)
    digest = hashlib.sha256("\n".join(cache.paths[idx]).encode()).hexdigest()[:10]
    key = f"{res}px_n{len(idx)}_flip{int(not a.no_flip)}_{digest}"
    lat_path = (a.latent_cache or run) / f"latents_{key}.pt"
    if lat_path.exists():
        latents = torch.load(lat_path)
        log.info(f"Loaded latents {tuple(latents.shape)} from {lat_path}")
    else:
        t = time.time()
        latents = precompute_latents(vae, cache.images[idx], device, flip=not a.no_flip)
        lat_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents, lat_path)
        log.info(f"Encoded latents {tuple(latents.shape)} in {time.time() - t:.0f}s -> {lat_path}")
    latents = latents.to(device)

    # ---- LoRA on the UNet attention projections
    unet.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    cast_training_params(unet, dtype=torch.float32)
    if a.grad_checkpointing:
        unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    lr_at = lambda s: min(1.0, (s + 1) / a.warmup_steps) * 0.5 * (1 + math.cos(math.pi * s / a.steps))  # noqa: E731
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    config = {"run_name": name, **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
              "prompts": prompts, "n_train_images": int(len(idx)), "resolution": res,
              "class_counts": dict(zip(CLASS_NAMES, np.bincount(labels.numpy(), minlength=3).tolist())),
              "n_lora_params": int(sum(p.numel() for p in params)), "device": device_name(device),
              "torch": torch.__version__, "code_version": code_version(), "slurm_job_id": slurm_job_id(),
              "split_version": SPLIT_VERSION, "started": datetime.now().isoformat(timespec="seconds"),
              "status": "running"}
    (run / "config.json").write_text(json.dumps(config, indent=2))
    log.info(f"Run {name}: {len(idx)} training images {config['class_counts']}, "
             f"{config['n_lora_params'] / 1e6:.2f}M LoRA params, {a.steps} steps x batch {a.batch_size}")

    def save_lora(dirname):
        sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(run / dirname, unet_lora_layers=sd, safe_serialization=True)

    def samples(step):
        unet.eval()
        comps = dict(vae=vae.to(torch.float16), text_encoder=te, tokenizer=tok, unet=unet,
                     scheduler=sched, safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        from diffusers import DPMSolverMultistepScheduler
        comps["scheduler"] = DPMSolverMultistepScheduler.from_config(sched.config)
        sample_grid(StableDiffusionPipeline, comps, prompts, a.n_sample_per_class, a.guidance, 1234,
                    run / f"samples_step{step:05d}.png", f"{name} - step {step} - guidance {a.guidance}")
        vae.to(torch.float32)
        unet.train()

    if not a.smoke:
        samples(0)  # base model before fine-tuning, for comparison
    loss_f = open(run / "loss.csv", "w", newline="")
    lw = csv.writer(loss_f)
    lw.writerow(["step", "loss", "lr", "sec_per_step"])
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    unet.train()
    t0, run_loss = time.time(), []
    for step in range(1, a.steps + 1):
        b = torch.randint(0, len(idx), (a.batch_size,), generator=g)
        view = torch.zeros(a.batch_size, dtype=torch.long) if a.no_flip else torch.randint(0, 2, (a.batch_size,), generator=g)
        x0 = latents[b.to(device), view.to(device)].float()
        cls = labels[b].clone()
        cls[torch.rand(a.batch_size, generator=g) < a.drop_prompt_prob] = 3  # empty prompt
        cond = emb[cls.to(device)]
        noise = torch.randn_like(x0)
        ts = torch.randint(0, sched.config.num_train_timesteps, (a.batch_size,), device=device).long()
        xt = sched.add_noise(x0, noise, ts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(xt.to(torch.bfloat16), ts, encoder_hidden_states=cond).sample
        if a.snr_gamma > 0:
            snr = compute_snr(sched, ts)
            w = torch.clamp(snr, max=a.snr_gamma) / snr
            loss = (F.mse_loss(pred.float(), noise, reduction="none").mean(dim=[1, 2, 3]) * w).mean()
        else:
            loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        lr_sched.step()
        run_loss.append(loss.item())
        if step % 50 == 0 or step == a.steps:
            sps = (time.time() - t0) / step
            lw.writerow([step, f"{np.mean(run_loss):.5f}", f"{lr_sched.get_last_lr()[0]:.2e}", f"{sps:.3f}"])
            loss_f.flush()
            log.info(f"step {step}/{a.steps} loss {np.mean(run_loss):.4f} | {sps:.2f}s/step | "
                     f"eta {(a.steps - step) * sps / 60:.0f} min")
            run_loss = []
        if step % a.save_every == 0 and step != a.steps:
            save_lora(f"lora_step{step:05d}")
        if step % a.sample_every == 0 or step == a.steps:
            samples(step)
    loss_f.close()
    save_lora("lora_final")
    config.update(status="done", finished=datetime.now().isoformat(timespec="seconds"),
                  minutes=round((time.time() - t0) / 60, 1))
    (run / "config.json").write_text(json.dumps(config, indent=2))
    log.info(f"Done. LoRA weights: {run / 'lora_final'}")


if __name__ == "__main__":
    main()

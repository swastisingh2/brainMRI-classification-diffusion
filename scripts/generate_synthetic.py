"""Generate a synthetic image set from a plan (scripts/make_synthetic_plan.py) with the fine-tuned LoRA.

Modes:
  img2img  (default) noise each plan row's source TRAINING image to --strength, then denoise with the
           class prompt. strength 0 = copy of the source, 1 = ignores the source entirely.
  txt2img  ignore sources; generate from the class prompt alone.

Every row has its own fixed noise seed (--seed + row), so a set can be regenerated exactly.

Output: <out>.npz with images (N,S,S) uint8 at --out_size (matching the classifier cache), labels,
source_paths and rows; <out>.json with settings; <out>_preview.png showing sources next to outputs.
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainmri import CLASS_NAMES  # noqa: E402
from brainmri.data import Cache  # noqa: E402
from brainmri.diffusion import (DEFAULT_MODEL_ID, PROMPT_TEMPLATE, class_prompts, gray_uint8_to_sd,  # noqa: E402
                                load_pipelines, sd_to_gray_uint8)
from brainmri.utils import code_version, device_name, pick_device, slurm_job_id  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--plan", type=Path, required=True)
p.add_argument("--lora_dir", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True, help="512px cache (source images for img2img)")
p.add_argument("--out", type=Path, required=True, help="output path without extension")
p.add_argument("--mode", choices=["img2img", "txt2img"], default="img2img")
p.add_argument("--strength", type=float, default=0.5)
p.add_argument("--guidance", type=float, default=5.0)
p.add_argument("--steps", type=int, default=30)
p.add_argument("--out_size", type=int, default=256)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--limit", type=int, default=0, help="only the first N plan rows (for quick tests)")
p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
p.add_argument("--prompt_template", default=PROMPT_TEMPLATE)
p.add_argument("--min_gpu_gb", type=float, default=0.0)
a = p.parse_args()

device = pick_device(a.min_gpu_gb)
plan = list(csv.DictReader(open(a.plan)))
if a.limit:
    plan = plan[:a.limit]
cache = Cache(a.cache)
pos = {path: i for i, path in enumerate(cache.paths)}
src_idx = np.array([pos[r["source_path"]] for r in plan])
if not np.all(cache.split[src_idx] == "train"):
    raise RuntimeError("Plan contains non-training source images - refusing (would leak val/test).")
labels = np.array([int(r["label"]) for r in plan])
prompts = class_prompts(a.prompt_template)

t2i, i2i = load_pipelines(a.model_id, str(a.lora_dir), device)
out_imgs = np.zeros((len(plan), a.out_size, a.out_size), dtype=np.uint8)
t0 = time.time()
for s in range(0, len(plan), a.batch_size):
    rows = range(s, min(s + a.batch_size, len(plan)))
    gens = [torch.Generator(device=device).manual_seed(a.seed + int(plan[i]["row"])) for i in rows]
    pr = [prompts[labels[i]] for i in rows]
    with torch.inference_mode():
        if a.mode == "txt2img":
            res = t2i(pr, num_inference_steps=a.steps, guidance_scale=a.guidance, generator=gens, output_type="pt")
        else:
            init = (gray_uint8_to_sd(cache.images[src_idx[list(rows)]]) + 1) / 2  # pipeline wants [0, 1]
            res = i2i(pr, image=init.to(device, torch.float16), strength=a.strength, num_inference_steps=a.steps,
                      guidance_scale=a.guidance, generator=gens, output_type="pt")
    out_imgs[list(rows)] = sd_to_gray_uint8(res.images, a.out_size)
    done = rows[-1] + 1
    if done % (a.batch_size * 10) == 0 or done == len(plan):
        rate = (time.time() - t0) / done
        print(f"{done}/{len(plan)}  {rate:.2f}s/img  eta {(len(plan) - done) * rate / 60:.0f} min", flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out.with_suffix(".npz"), images=out_imgs, labels=labels,
                    source_paths=np.array([r["source_path"] for r in plan]), rows=np.array([int(r["row"]) for r in plan]))
meta = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "n": len(plan), "per_class": dict(zip(CLASS_NAMES, np.bincount(labels, minlength=3).tolist())),
        "plan_summary": json.loads(a.plan.with_suffix(".json").read_text()) if a.plan.with_suffix(".json").exists() else None,
        "prompts": prompts, "device": device_name(device), "code_version": code_version(),
        "slurm_job_id": slurm_job_id(), "minutes": round((time.time() - t0) / 60, 1),
        "finished": datetime.now().isoformat(timespec="seconds")}
a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

# Preview: for each class, up to 6 pairs of (source, synthetic)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from brainmri.data import resize_pair  # noqa: E402
n = 6
fig, axes = plt.subplots(3, 2 * n, figsize=(1.6 * 2 * n, 1.8 * 3))
for c in range(3):
    ids = np.flatnonzero(labels == c)[:n]
    for j in range(n):
        for t in range(2):
            ax = axes[c, 2 * j + t]
            ax.axis("off")
            if j < len(ids):
                i = ids[j]
                if t == 0:
                    img, _ = resize_pair(cache.images[src_idx[i]].astype(np.float32) / 255, cache.masks[src_idx[i]], a.out_size)
                    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                    ax.set_title("source" if a.mode == "img2img" else "", fontsize=7)
                else:
                    ax.imshow(out_imgs[i], cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"synthetic {CLASS_NAMES[c][:4]}", fontsize=7)
fig.suptitle(f"{a.out.name}: {a.mode}, strength {a.strength}, guidance {a.guidance}", fontsize=9, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(a.out.parent / f"{a.out.name}_preview.png", dpi=80)
print(f"Wrote {a.out.with_suffix('.npz')} ({len(plan)} images) and preview")

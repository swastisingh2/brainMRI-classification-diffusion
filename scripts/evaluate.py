"""Evaluate a trained checkpoint on one split and save metrics, predictions and a confusion matrix.

TEAM RULE: the test split is for final evaluation only. Do not use test results to choose
settings, seeds, epochs or which classes to target. Choose everything with validation first.

Usage:
    python scripts/evaluate.py --ckpt <run_dir>/best.pt --cache <cache.npz> --split val
    python scripts/evaluate.py --ckpt <run_dir>/best.pt --cache <cache.npz> --split test --out <results>/<run_name>
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_classifier import predict, write_predictions  # noqa: E402
from brainmri.data import Cache, MRIDataset, make_transform  # noqa: E402
from brainmri.metrics import compute_metrics  # noqa: E402
from brainmri.models import build_model  # noqa: E402
from brainmri.plots import plot_confusion  # noqa: E402
from brainmri.utils import pick_device  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--split", choices=["train", "val", "test"], required=True)
p.add_argument("--out", type=Path, default=None, help="folder to write into (default: next to the checkpoint)")
p.add_argument("--min_gpu_gb", type=float, default=0.0)
a = p.parse_args()

device = pick_device(a.min_gpu_gb)
ck = torch.load(a.ckpt, map_location=device, weights_only=False)
cfg = ck["config"]
model = build_model(cfg["model"], pretrained=False).to(device)
model.load_state_dict(ck["model"])
cache = Cache(a.cache)
dl = DataLoader(MRIDataset(cache, cache.indices(a.split), make_transform("none", cfg["img_size"])),
                batch_size=64, shuffle=False, num_workers=4 if device.type == "cuda" else 0)
amp = torch.bfloat16 if device.type == "cuda" else None
res = predict(model, dl, device, nn.CrossEntropyLoss(), amp)
m = compute_metrics(res["labels"], res["pred"], cache.groups[res["idx"]], 1000, cfg["seed"])
m.update(run_name=cfg["run_name"], split=a.split, checkpoint_epoch=ck["epoch"])
out = a.out or a.ckpt.parent
out.mkdir(parents=True, exist_ok=True)
(out / f"metrics_{a.split}.json").write_text(json.dumps(m, indent=2))
write_predictions(out / f"predictions_{a.split}.csv", res, cache)
plot_confusion(m["confusion_matrix"], out / f"confusion_{a.split}.png",
               f"{cfg['exp']} seed {cfg['seed']} - {a.split} (n={m['n']})")
ci = m["bootstrap_95ci"]
print(f"{cfg['run_name']} on {a.split}: acc {m['accuracy']:.4f} [{ci['accuracy'][0]:.3f}, {ci['accuracy'][1]:.3f}]  "
      f"macro-F1 {m['macro_f1']:.4f} [{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}]  -> {out}")

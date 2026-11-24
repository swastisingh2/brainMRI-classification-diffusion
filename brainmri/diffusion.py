"""Shared pieces for the Stable Diffusion LoRA generator: prompts, image conversion, pipeline loading."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import CLASS_NAMES

DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPT_TEMPLATE = "a contrast-enhanced T1-weighted brain MRI slice showing a {cls} tumor"


def class_prompts(template: str = PROMPT_TEMPLATE) -> list[str]:
    return [template.format(cls=c) for c in CLASS_NAMES]


def gray_uint8_to_sd(images: np.ndarray) -> torch.Tensor:
    """(N,H,W) uint8 grayscale -> (N,3,H,W) float in [-1, 1], the range the SD VAE expects."""
    x = torch.from_numpy(images).float().div_(127.5).sub_(1.0)
    return x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()


def sd_to_gray_uint8(x: torch.Tensor, out_size: int | None = None) -> np.ndarray:
    """(N,3,H,W) in [0, 1] (diffusers 'pt' output) -> (N,out,out) uint8 grayscale, resized like the cache."""
    g = x.float().mean(1, keepdim=True).clamp(0, 1)
    if out_size and g.shape[-1] != out_size:
        g = F.interpolate(g, size=(out_size, out_size), mode="bilinear", antialias=True, align_corners=False)
    return (g[:, 0].clamp(0, 1) * 255).round().byte().cpu().numpy()


@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompts: list[str], device) -> torch.Tensor:
    ids = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(ids)[0]


def load_pipelines(model_id: str, lora_dir: str | None, device, dtype=torch.float16):
    """Text-to-image and image-to-image pipelines sharing one set of weights, with the LoRA loaded."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

    t2i = StableDiffusionPipeline.from_pretrained(model_id, variant="fp16", torch_dtype=dtype,
                                                  safety_checker=None, requires_safety_checker=False)
    t2i.scheduler = DPMSolverMultistepScheduler.from_config(t2i.scheduler.config)
    if lora_dir:
        t2i.load_lora_weights(lora_dir)
    t2i = t2i.to(device)
    t2i.set_progress_bar_config(disable=True)
    i2i = StableDiffusionImg2ImgPipeline(**t2i.components)
    i2i.set_progress_bar_config(disable=True)
    return t2i, i2i

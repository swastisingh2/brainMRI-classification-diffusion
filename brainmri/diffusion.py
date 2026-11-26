"""Shared pieces for the Stable Diffusion LoRA generator: prompts, image conversion, pipeline loading.

Background, briefly:
- Stable Diffusion (SD) 1.5 generates 512x512 images from a text prompt. It works in a compressed
  "latent" space: a VAE encodes an image into a 4x64x64 latent, a UNet removes noise from latents step by
  step guided by the prompt, and the VAE decodes the final latent back into an image.
- SD was trained on internet photos, not MRIs, so we fine-tune it with LoRA (small trainable matrices added
  to the UNet's attention layers; the original weights stay frozen) on our TRAINING images, with one prompt
  per tumor class. After fine-tuning, the class in the prompt controls which tumor type is drawn.
- img2img: instead of starting from pure noise, start from a real image with noise added up to a
  "strength" (0 = unchanged, 1 = pure noise) and denoise it with the class prompt. This gives realistic
  variations of a specific training image, which is how confusion-aware augmentation targets hard images.

MRIs are grayscale; SD works in RGB. We copy the gray channel into R, G and B on the way in, and average
the three channels on the way out.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import CLASS_NAMES

# Community mirror of the original SD 1.5 weights (the original repository was taken down in 2024).
# License: CreativeML OpenRAIL-M, which permits research use.
DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPT_TEMPLATE = "a contrast-enhanced T1-weighted brain MRI slice showing a {cls} tumor"


def class_prompts(template: str = PROMPT_TEMPLATE) -> list[str]:
    """One prompt per class, in label order: [meningioma, glioma, pituitary]."""
    return [template.format(cls=c) for c in CLASS_NAMES]


def gray_uint8_to_sd(images: np.ndarray) -> torch.Tensor:
    """(N,H,W) uint8 grayscale -> (N,3,H,W) float in [-1, 1], the range the SD VAE expects."""
    x = torch.from_numpy(images).float().div_(127.5).sub_(1.0)  # 0..255 -> -1..1
    return x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()  # copy gray into 3 identical channels


def sd_to_gray_uint8(x: torch.Tensor, out_size: int | None = None) -> np.ndarray:
    """(N,3,H,W) in [0, 1] (diffusers 'pt' output) -> (N,out,out) uint8 grayscale, resized like the cache.

    Resizing uses the same bilinear+antialias method as the classifier cache, so synthetic and real
    images go through identical processing before the classifier sees them.
    """
    g = x.float().mean(1, keepdim=True).clamp(0, 1)  # average R, G, B back to one gray channel
    if out_size and g.shape[-1] != out_size:
        g = F.interpolate(g, size=(out_size, out_size), mode="bilinear", antialias=True, align_corners=False)
    return (g[:, 0].clamp(0, 1) * 255).round().byte().cpu().numpy()


@torch.no_grad()
def encode_prompts(tokenizer, text_encoder, prompts: list[str], device) -> torch.Tensor:
    """Turn prompts into the CLIP text embeddings the UNet is conditioned on: (N, 77, 768)."""
    ids = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(ids)[0]


def load_pipelines(model_id: str, lora_dir: str | None, device, dtype=torch.float16):
    """Text-to-image and image-to-image pipelines sharing one set of weights, with the LoRA loaded.

    - The safety checker is disabled: it is a nudity filter for photos and would only produce false
      positives (black images) on medical scans.
    - DPMSolverMultistepScheduler gives good samples in ~30 steps instead of the default 50.
    - Both pipelines share the same UNet/VAE/text encoder in GPU memory (no second copy).
    """
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

    t2i = StableDiffusionPipeline.from_pretrained(model_id, variant="fp16", torch_dtype=dtype,
                                                  safety_checker=None, requires_safety_checker=False)
    t2i.scheduler = DPMSolverMultistepScheduler.from_config(t2i.scheduler.config)
    if lora_dir:
        t2i.load_lora_weights(lora_dir)  # adds our fine-tuned LoRA on top of the frozen base UNet
    t2i = t2i.to(device)
    t2i.set_progress_bar_config(disable=True)
    i2i = StableDiffusionImg2ImgPipeline(**t2i.components)
    i2i.set_progress_bar_config(disable=True)
    return t2i, i2i

# LoRA for Stable Diffusion 1.5, fine-tuned on Figshare brain MRI

The generator used in Phase 2: a LoRA adapter (12.8 MB) for
[`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
The base model is not stored here; it downloads automatically on first use (about 4 GB).

| | |
|---|---|
| File | `pytorch_lora_weights.safetensors` (diffusers format), SHA-256 `9a73ccc2…7a20619a6034` |
| Trained on | the 2,193 **training-split** images of `splits_v1_provisional` only (no val/test images) |
| Resolution | 512 × 512, grayscale copied to RGB |
| LoRA | rank 16 on the UNet attention projections (to_q, to_k, to_v, to_out.0), 3.19 M parameters |
| Training | 6,000 steps, batch 8, lr 1e-4 cosine, min-SNR γ = 5, prompt dropout 0.1, horizontal flips |
| Code | `scripts/train_diffusion_lora.py` at commit `2148239`; all settings in `config.json` |

Prompt (one per class; `{cls}` = meningioma, glioma or pituitary):

    a contrast-enhanced T1-weighted brain MRI slice showing a {cls} tumor

## Load it

```python
from brainmri.diffusion import load_pipelines, class_prompts

t2i, i2i = load_pipelines("stable-diffusion-v1-5/stable-diffusion-v1-5",
                          "models/lora_sd15_figshare_r16", device="cuda")
```

`brainmri/diffusion.py` also has `gray_uint8_to_sd` / `sd_to_gray_uint8` to convert between our cached
images and the pipeline's input/output. Plain diffusers works too: `pipe.load_lora_weights("models/lora_sd15_figshare_r16")`.
fp16 on a GPU with about 8 GB or more is enough for generation.

## What we know about it (from Phase 2)

- **Use img2img, not text-to-image.** Text-only samples (`samples_final.png`) look like real anatomy but are
  mostly sagittal with weak or missing tumors.
- **Keep strength at 0.3 or lower if the tumor must survive.** At strength 0.5, about 30% of outputs lost more
  than half their tumor contrast (glioma worst); at 0.3, 94% kept it. `scripts/filter_synthetic.py`
  measures this per image using the source's tumor mask.
- Guidance 5 with the DPM-Solver scheduler (the `load_pipelines` default) is what we used.

## Notes for counterfactuals

- Start img2img from a real image with the **target** class prompt (e.g. a meningioma slice with the glioma
  prompt) and sweep strength from low to high; the lowest strength that flips the classifier is the
  "minimal" edit.
- The generator never saw validation or test images, so counterfactuals can be made from them. Report them on
  validation while developing; touch test only for the final run.
- Compare each edit against the tumor mask: does the change happen inside the tumor, or elsewhere
  (orientation, skull, contrast)? The latter would point to a classifier shortcut.

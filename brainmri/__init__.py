"""Shared code for the brain tumor MRI classification + diffusion project.

Modules:
    data       read .mat files, preprocess, cache, PyTorch dataset and augmentations
    models     ImageNet-pretrained classifiers with a 3-class head
    metrics    accuracy, macro-F1, per-class scores, confusion, patient-group bootstrap CIs
    plots      training curves, confusion matrices, summary charts
    diffusion  Stable Diffusion helpers (prompts, image conversion, pipelines)
    utils      seeding, device choice, code version, run registry
Scripts that use these live in scripts/.
"""

CLASS_NAMES = ["meningioma", "glioma", "pituitary"]  # 0-based model labels = CSV label - 1
SPLIT_VERSION = "splits_v1_provisional"  # recorded in every run so results can be tied to the split used

"""Classifier factory: builds an ImageNet-pretrained CNN with a new 3-class output layer.

All models start from ImageNet weights (transfer learning): the early layers already detect edges and
textures, so with only ~2,200 training images we fine-tune instead of training from scratch.
Only the final layer is replaced; it is freshly initialised and outputs 3 scores (one per tumor class).
"""
import torch.nn as nn
from torchvision import models

from . import CLASS_NAMES


def build_model(name: str = "resnet18", pretrained: bool = True, num_classes: int = len(CLASS_NAMES)) -> nn.Module:
    """Return the requested architecture with `num_classes` outputs.

    resnet18        11M parameters; the baseline (B0/B1) model
    resnet50        25M parameters; a larger alternative
    efficientnet_b0 5M parameters; a different architecture family, planned as the independent
                    "second judge" classifier for Phase 3 counterfactuals
    pretrained=False gives random initial weights (used only in quick tests without internet access).
    """
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)  # replace the 1000-class ImageNet head
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)  # last layer of its head
    else:
        raise ValueError(f"unknown model '{name}'")
    return m

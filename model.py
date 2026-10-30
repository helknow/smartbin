"""Shared definitions for training and the app: classes, transforms, experiment architectures."""
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

BIN_INFO = {
    "cardboard": {"bin": "Paper & cardboard bin (blue)", "color": "#2563eb",
                  "tip": "Flatten boxes and keep them dry."},
    "glass": {"bin": "Glass bin (green)", "color": "#16a34a",
              "tip": "Empty and rinse bottles and jars; remove the lids."},
    "metal": {"bin": "Metal bin (grey)", "color": "#6b7280",
              "tip": "Rinse cans and crush them if you can."},
    "paper": {"bin": "Paper & cardboard bin (blue)", "color": "#2563eb",
              "tip": "Keep it clean and dry; greasy paper goes in general trash."},
    "plastic": {"bin": "Plastic bin (yellow)", "color": "#ca8a04",
                "tip": "Empty and rinse containers; squash bottles to save space."},
    "trash": {"bin": "General waste bin (black)", "color": "#1f2937",
              "tip": "This item can't be recycled with the other materials."},
}

EXPERIMENTS = ["E1", "E2", "E3", "E4", "E5"]

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def _freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def _smaller_head():
    return nn.Sequential(nn.Linear(25088, 1024), nn.ReLU(), nn.Dropout(0.3), nn.Linear(1024, len(CLASSES)))


def build_model(exp_id, pretrained=True):
    """Build the architecture for one experiment. E2 and E3 differ only in optimiser (see train.py)."""
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {exp_id}")
    weights = "IMAGENET1K_V1" if pretrained else None

    if exp_id == "E5":
        model = models.mobilenet_v2(weights=weights)
        _freeze(model.features)
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, len(CLASSES)))
        return model

    model = models.vgg16(weights=weights)
    _freeze(model.features)
    if exp_id == "E1":
        model.classifier[6] = nn.Linear(4096, len(CLASSES))
    else:
        model.classifier = _smaller_head()
    if exp_id == "E4":
        for p in model.features[24:].parameters():
            p.requires_grad = True
    return model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, exp_id, test_acc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"exp_id": exp_id, "classes": CLASSES, "state_dict": model.state_dict(), "test_acc": test_acc}, path)

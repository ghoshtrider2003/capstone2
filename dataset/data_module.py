"""Dataset and dataloader helpers for HMSAN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    data_dir: str
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train/eval transforms according to project specification."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def _split_indices(dataset_size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split indices as 70/15/15 train/val/test."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator)

    train_end = int(0.70 * dataset_size)
    val_end = train_end + int(0.15 * dataset_size)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx


def create_dataloaders(config: DataConfig):
    """Build ImageFolder datasets and dataloaders with fixed split indices."""
    root = Path(config.data_dir)
    train_tf, eval_tf = build_transforms(config.image_size)

    base_dataset = datasets.ImageFolder(root=root)
    class_names = base_dataset.classes

    train_idx, val_idx, test_idx = _split_indices(len(base_dataset), config.seed)

    train_dataset = datasets.ImageFolder(root=root, transform=train_tf)
    val_dataset = datasets.ImageFolder(root=root, transform=eval_tf)
    test_dataset = datasets.ImageFolder(root=root, transform=eval_tf)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(test_dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names

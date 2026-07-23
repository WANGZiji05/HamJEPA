from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import os
from torch.utils.data import ConcatDataset, Dataset
# Direct imports to avoid triggering torchvision.datasets (which needs requests)
from torchvision.transforms import (
    ColorJitter,
    Compose,
    GaussianBlur,
    Normalize,
    RandomApply,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomSolarize,
    ToTensor,
)
# Lazy import: ImageFolder is only needed for ImageNet datasets, not Physion
# from torchvision.datasets import ImageFolder

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class MultiCropCfg:
    num_global_views: int = 2
    num_local_views: int = 6

    out_size: int = 224

    global_scale: Tuple[float, float] = (0.4, 1.0)
    local_scale: Tuple[float, float] = (0.05, 0.4)

    hflip_p: float = 0.5
    cj_p: float = 0.8
    grayscale_p: float = 0.2
    blur_p: float = 0.5
    solarize_p: float = 0.2
    solarize_thresh: int = 128

    cj_brightness: float = 0.4
    cj_contrast: float = 0.4
    cj_saturation: float = 0.2
    cj_hue: float = 0.1


def _build_transform(cfg: MultiCropCfg, *, scale: Tuple[float, float]) -> Compose:
    color_jitter = ColorJitter(
        brightness=cfg.cj_brightness,
        contrast=cfg.cj_contrast,
        saturation=cfg.cj_saturation,
        hue=cfg.cj_hue,
    )
    blur = GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))

    return Compose(
        [
            RandomResizedCrop(cfg.out_size, scale=scale),
            RandomHorizontalFlip(p=cfg.hflip_p),
            RandomApply([color_jitter], p=cfg.cj_p),
            RandomGrayscale(p=cfg.grayscale_p),
            RandomApply([blur], p=cfg.blur_p),
            RandomSolarize(threshold=cfg.solarize_thresh, p=cfg.solarize_p),
            ToTensor(),
            Normalize(mean=_MEAN, std=_STD),
        ]
    )


class ImageNetMultiCrop(Dataset):
    def __init__(self, root: str, split: str, cfg: MultiCropCfg):
        super().__init__()
        roots = _resolve_split_roots(root, split)
        self.base = _build_imagefolder_concat(roots)
        self.cfg = cfg
        self.global_tf = _build_transform(cfg, scale=cfg.global_scale)
        self.local_tf = _build_transform(cfg, scale=cfg.local_scale)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        views: List[torch.Tensor] = []

        for _ in range(self.cfg.num_global_views):
            views.append(self.global_tf(img))

        for _ in range(self.cfg.num_local_views):
            views.append(self.local_tf(img))

        coarse = -1
        return views, int(label), int(coarse)


def _resolve_split_roots(root: str, split: str) -> List[str]:
    split = split.lower()
    root = os.path.abspath(root)
    if split in ("train", "trainset", "train_set"):
        primary = os.path.join(root, "train")
        if os.path.isdir(primary):
            return [primary]
        # Kaggle ImageNet-100 style: train.X1 .. train.X4
        parts = sorted(
            d for d in os.listdir(root) if d.startswith("train.") and os.path.isdir(os.path.join(root, d))
        )
        return [os.path.join(root, d) for d in parts]
    if split in ("val", "valid", "validation"):
        primary = os.path.join(root, "val")
        if os.path.isdir(primary):
            return [primary]
        parts = sorted(
            d for d in os.listdir(root) if d.startswith("val.") and os.path.isdir(os.path.join(root, d))
        )
        return [os.path.join(root, d) for d in parts]
    raise ValueError(f"Unknown split '{split}'. Expected train/val.")


def _build_imagefolder_concat(roots: List[str]) -> Dataset:
    from torchvision.datasets import ImageFolder  # lazy import (needs requests on some envs)

    if not roots:
        raise RuntimeError("No dataset roots found for the requested split.")
    datasets: List[ImageFolder] = []
    ref_classes = None
    for r in roots:
        ds = ImageFolder(root=r, transform=None)
        if ref_classes is None:
            ref_classes = ds.classes
        elif ds.classes != ref_classes:
            raise RuntimeError(f"Class list mismatch between splits: {r}")
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

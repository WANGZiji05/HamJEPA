"""
Physion++ Multi-Crop Frame Dataset for HamJEPA Self-Supervised Pretraining.

Loads random frames from Physion++ training videos (data_v1/) and applies
multi-crop augmentations for JEPA-style self-supervised learning.

Each __getitem__ call:
  1. Randomly picks a video from the training split
  2. Randomly picks a frame index within that video
  3. Decodes the frame with decord (PIL Image)
  4. Applies V augmentations (global + local views)
  5. Returns (views, dummy_label, dummy_coarse)

Usage (same interface as ImageNetMultiCrop):
    from eval.datasets.physion_multicrop import PhysionMultiCrop, MultiCropCfg

    cfg = MultiCropCfg(num_global_views=2, num_local_views=0, out_size=224)
    ds = PhysionMultiCrop(data_root="/path/to/physion_data", split="train", cfg=cfg)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Same augmentations as ImageNetMultiCrop
from eval.datasets.imagenet_multicrop import MultiCropCfg, _build_transform

# Physion++ property mapping (scenario prefix → physical property)
PREFIX_TO_PROPERTY = {
    "bouncy": "elasticity",
    "deform": "deformability",
    "friction": "friction",
    "mass": "mass",
}


def _scan_physion_videos(data_dir: str) -> List[str]:
    """
    Recursively scan data_dir for *_img.mp4 files.

    Physion++ directory structure:
        data_v1/
        ├── bouncy_wall_pp/
        │   └── bouncy_wall-.../
        │       ├── 0000_img.mp4   ← we use this
        │       ├── 0000_id.mp4    ← skip
        │       └── 0000.pkl
        └── ...

    Returns:
        List of absolute paths to all *_img.mp4 video files.
    """
    paths = []
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Physion++ data directory not found: {data_dir}")

    for video_file in sorted(data_path.rglob("*_img.mp4")):
        paths.append(str(video_file))

    if len(paths) == 0:
        raise RuntimeError(
            f"No *_img.mp4 files found in {data_dir}. "
            f"Please check the data directory structure."
        )

    return paths


class PhysionMultiCrop(Dataset):
    """
    Physion++ multi-crop dataset for HamJEPA pretraining.

    Randomly samples single frames from Physion++ training videos and
    applies multi-view augmentations. Compatible with the HamJEPA
    training loop (same interface as ImageNetMultiCrop).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        cfg: MultiCropCfg = None,
        max_videos: int = None,
        frames_per_video: int = 150,
        seed: int = 42,
    ):
        """
        Args:
            root: Path to Physion++ data root (contains data_v1/, readout_data_v1/, testdata_v1/)
            split: "train" → data_v1/, "readout" → readout_data_v1/
            cfg: MultiCropCfg for augmentations
            max_videos: Limit number of videos (None = use all)
            frames_per_video: Estimated number of usable frames per video
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.cfg = cfg or MultiCropCfg()

        # Resolve data directory
        split_map = {
            "train": "data_v1",
            "readout": "readout_data_v1",
            "test": "testdata_v1",
        }
        subdir = split_map.get(split.lower(), "data_v1")
        data_dir = os.path.join(root, subdir)

        # Scan for video files
        self.video_paths = _scan_physion_videos(data_dir)
        if max_videos is not None and max_videos > 0:
            rng = np.random.RandomState(seed)
            rng.shuffle(self.video_paths)
            self.video_paths = self.video_paths[:max_videos]

        self.frames_per_video = frames_per_video

        # Build transforms
        self.global_tf = _build_transform(cfg, scale=cfg.global_scale)
        self.local_tf = _build_transform(cfg, scale=cfg.local_scale)

        # Cache video readers per worker (lazy init)
        self._vr_cache = {}
        self._vr_path = None

        print(f"PhysionMultiCrop [{split}]: {len(self.video_paths)} videos, "
              f"~{len(self) // 1000}k samples")

    def __len__(self) -> int:
        return len(self.video_paths) * self.frames_per_video

    def _load_frame(self, video_path: str, frame_idx: int) -> Image.Image:
        """Load a single frame from a video file using decord."""
        try:
            from decord import VideoReader, cpu as decord_cpu
        except ImportError:
            raise ImportError(
                "decord is required for Physion++ video loading. "
                "Install with: pip install decord"
            )

        # Cache VideoReader per worker (avoids reopening for same video)
        if self._vr_path != video_path or video_path not in self._vr_cache:
            try:
                self._vr_cache = {}  # clear old cache
                self._vr_cache[video_path] = VideoReader(
                    video_path, num_threads=1, ctx=decord_cpu(0)
                )
                self._vr_path = video_path
            except Exception:
                return None

        vr = self._vr_cache[video_path]
        total = len(vr)

        # Clamp frame index to valid range
        idx = max(0, min(frame_idx, total - 1))

        try:
            frame = vr[idx].asnumpy()  # [H, W, C] uint8
            return Image.fromarray(frame)
        except Exception:
            return None

    def __getitem__(self, idx: int):
        """
        Returns:
            views: List[Tensor[C, H, W]] — multi-crop augmented views
            label: int (dummy, always 0 for unsupervised pretraining)
            coarse: int (dummy, always -1)
        """
        # Map flat index → (video_idx, frame_idx)
        video_idx = idx % len(self.video_paths)
        max_attempts = 10

        for _ in range(max_attempts):
            video_path = self.video_paths[video_idx]

            # Random frame within the video
            frame_idx = random.randint(0, self.frames_per_video - 1)

            img = self._load_frame(video_path, frame_idx)
            if img is not None:
                break

            # Try another video if this one fails
            video_idx = random.randint(0, len(self.video_paths) - 1)
        else:
            # Fallback: create a blank image
            img = Image.new("RGB", (self.cfg.out_size, self.cfg.out_size), (128, 128, 128))

        # Apply multi-crop augmentations
        views: List[torch.Tensor] = []

        for _ in range(self.cfg.num_global_views):
            views.append(self.global_tf(img))

        for _ in range(self.cfg.num_local_views):
            views.append(self.local_tf(img))

        # Return dummy labels (unsupervised pretraining)
        return views, 0, -1

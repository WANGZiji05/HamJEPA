"""
Physion++ Multi-Crop Frame Dataset for HamJEPA Self-Supervised Pretraining.

Loads random frames from Physion++ training videos (data_v1/) and applies
multi-crop augmentations for JEPA-style self-supervised learning.

Performance-critical design:
  - Pre-scans all videos during init to get actual frame counts
  - Builds a flat (video_idx, frame_idx) index sorted by video
  - Shuffles video access order (not individual frames), so frames from
    the same video are contiguous → VideoReader stays open for many frames
  - Avoids the 100-500ms overhead of random h264 seek per frame

Usage (same interface as ImageNetMultiCrop):
    cfg = MultiCropCfg(num_global_views=2, num_local_views=0, out_size=224)
    ds = PhysionMultiCrop(root="/path/to/physion_data", split="train", cfg=cfg)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from eval.datasets.imagenet_multicrop import MultiCropCfg, _build_transform


def _scan_physion_videos(data_dir: str) -> List[str]:
    """Recursively scan data_dir for *_img.mp4 files. Returns absolute paths."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Physion++ data directory not found: {data_dir}")
    paths = sorted(str(f) for f in data_path.rglob("*_img.mp4"))
    if not paths:
        raise RuntimeError(f"No *_img.mp4 files found in {data_dir}")
    return paths


def _get_video_frame_counts(video_paths: List[str], max_frames_per_video: int = 150) -> List[int]:
    """Quickly scan all videos to get actual frame counts (via decord)."""
    from decord import VideoReader, cpu as decord_cpu

    counts = []
    t0 = time.time()
    for i, vp in enumerate(video_paths):
        try:
            vr = VideoReader(vp, num_threads=1, ctx=decord_cpu(0))
            n = min(len(vr), max_frames_per_video)
            counts.append(n)
        except Exception:
            counts.append(0)
        if (i + 1) % 1000 == 0:
            print(f"  Scanned {i + 1}/{len(video_paths)} videos ({time.time() - t0:.0f}s)")
    print(f"  Frame scan complete: {sum(counts)} usable frames in {time.time() - t0:.0f}s")
    return counts


class PhysionMultiCrop(Dataset):
    """
    Physion++ multi-crop dataset for HamJEPA pretraining.

    Pre-builds a flat index sorted by video, so consecutive __getitem__
    calls read from the same video → VideoReader stays open → fast.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        cfg: MultiCropCfg = None,
        max_videos: int = None,
        max_frames_per_video: int = 150,
        seed: int = 42,
    ):
        super().__init__()
        self.cfg = cfg or MultiCropCfg()

        # Resolve data directory
        split_map = {"train": "data_v1", "readout": "readout_data_v1", "test": "testdata_v1"}
        subdir = split_map.get(split.lower(), "data_v1")
        data_dir = os.path.join(root, subdir)

        # Scan videos
        self.video_paths = _scan_physion_videos(data_dir)
        rng = np.random.RandomState(seed)
        rng.shuffle(self.video_paths)
        if max_videos is not None and max_videos > 0:
            self.video_paths = self.video_paths[:max_videos]

        # Get actual frame counts
        print(f"Scanning frame counts for {len(self.video_paths)} videos...")
        self.frame_counts = _get_video_frame_counts(self.video_paths, max_frames_per_video)

        # Filter out videos with 0 frames
        valid = [(i, c) for i, c in enumerate(self.frame_counts) if c > 0]
        self._valid_videos = valid
        self._total_frames = sum(c for _, c in valid)

        # Build flat index: list of (video_idx, frame_idx) — SORTED by video
        # (do NOT shuffle; VideoBlockSampler handles shuffling at video level)
        self._index = []
        self._video_blocks = []  # [(start, end), ...] — one block per video
        for vid_idx, count in valid:
            step = max(1, count // max_frames_per_video) if count > max_frames_per_video else 1
            frame_indices = list(range(0, count, step))[:max_frames_per_video]
            start = len(self._index)
            for fi in frame_indices:
                self._index.append((vid_idx, fi))
            end = len(self._index)
            if end > start:
                self._video_blocks.append((start, end))

        # Build transforms
        self.global_tf = _build_transform(cfg, scale=cfg.global_scale)
        self.local_tf = _build_transform(cfg, scale=cfg.local_scale)

        # Per-worker VideoReader cache (stored on the dataset object for simplicity)
        self._vr_cache = {}  # video_path → VideoReader
        self._vr_max_cache = 5  # keep at most 5 readers open

        print(f"PhysionMultiCrop [{split}]: {len(valid)} videos, "
              f"{len(self._index)} frames")
        print(f"  ~{len(self) // 1000}k dataset size, "
              f"{self._index[0] if self._index else 'empty'}")

    def __len__(self) -> int:
        return len(self._index)

    def _load_frame(self, video_idx: int, frame_idx: int) -> Image.Image:
        """Load a single frame from a video. Uses LRU cache of VideoReaders."""
        from decord import VideoReader, cpu as decord_cpu

        video_path = self.video_paths[video_idx]

        # Check cache
        if video_path in self._vr_cache:
            vr = self._vr_cache[video_path]
        else:
            # Evict oldest if cache is full
            if len(self._vr_cache) >= self._vr_max_cache:
                oldest = next(iter(self._vr_cache.keys()))
                del self._vr_cache[oldest]
            try:
                vr = VideoReader(video_path, num_threads=1, ctx=decord_cpu(0))
                self._vr_cache[video_path] = vr
            except Exception:
                return None

        try:
            total = len(vr)
            idx = max(0, min(frame_idx, total - 1))
            frame = vr[idx].asnumpy()  # [H, W, C] uint8
            return Image.fromarray(frame)
        except Exception:
            return None

    def __getitem__(self, idx: int):
        """Returns (views, label, coarse) where views is a list of Tensor[C,H,W]."""
        video_idx, frame_idx = self._index[idx]
        max_attempts = 5

        for attempt in range(max_attempts):
            img = self._load_frame(video_idx, frame_idx)
            if img is not None:
                break
            # On failure, try a nearby frame or video
            if attempt < max_attempts - 1:
                idx = (idx + 1) % len(self._index)
                video_idx, frame_idx = self._index[idx]
        else:
            img = Image.new("RGB", (self.cfg.out_size, self.cfg.out_size), (128, 128, 128))

        # Apply multi-crop augmentations
        views: List[torch.Tensor] = []
        for _ in range(self.cfg.num_global_views):
            views.append(self.global_tf(img))
        for _ in range(self.cfg.num_local_views):
            views.append(self.local_tf(img))

        return views, 0, -1


class VideoBlockSampler(torch.utils.data.Sampler):
    """
    Sampler that shuffles video blocks, not individual frames.

    Each video forms a contiguous block of indices. Per epoch, we:
      1. Shuffle the list of video blocks
      2. Yield all frame indices within each block sequentially

    This ensures consecutive __getitem__ calls from the same worker
    read from the same video → VideoReader stays open → fast MP4 access.
    """

    def __init__(self, dataset: PhysionMultiCrop, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        blocks = list(self.dataset._video_blocks)
        if self.shuffle:
            perm = torch.randperm(len(blocks), generator=g).tolist()
            blocks = [blocks[p] for p in perm]
        for start, end in blocks:
            for i in range(start, end):
                yield i

    def __len__(self):
        return len(self.dataset)

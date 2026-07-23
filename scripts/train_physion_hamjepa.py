#!/usr/bin/env python3
"""
HamJEPA Pretraining on Physion++ Video Frames.

Minimal wrapper: overrides the dataset factory to use PhysionMultiCrop
(random frames from Physion++ videos) instead of ImageNetMultiCrop.

Usage:
    python scripts/train_physion_hamjepa.py --config configs/physion_hjepa_mv.yaml
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Patch: replace ImageNetMultiCrop with PhysionMultiCrop globally ──
import eval.datasets.imagenet_multicrop as _mc_mod
from eval.datasets.physion_multicrop import PhysionMultiCrop

_mc_mod.ImageNetMultiCrop = PhysionMultiCrop

# ── Run the standard training loop ──
if __name__ == "__main__":
    from scripts.train_imagenet_hamjepa import main
    main()

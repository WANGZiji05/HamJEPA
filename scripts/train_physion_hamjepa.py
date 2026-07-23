#!/usr/bin/env python3
"""
HamJEPA Pretraining on Physion++ Video Frames.

Minimal wrapper: overrides the dataset factory to use PhysionMultiCrop
(random frames from Physion++ videos) instead of ImageNetMultiCrop.

Usage:
    python scripts/train_physion_hamjepa.py --config configs/physion_hjepa_mv.yaml
"""

import sys
import types

# ── Fake requests module BEFORE any torchvision import ──
# torchvision.__init__ unconditionally imports torchvision.datasets, which
# imports _optical_flow → utils → requests. On compute nodes without network
# access the SSL module is broken, so pip cannot install requests.
# We inject a dummy module to satisfy the import chain.
if "requests" not in sys.modules:
    _dummy_requests = types.ModuleType("requests")
    sys.modules["requests"] = _dummy_requests

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

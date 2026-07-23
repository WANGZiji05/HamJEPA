"""
HamJEPA Physion++ Linear Probe Evaluation
==========================================

Evaluates a frozen HamJEPA encoder on the Physion++ OCP
(Object Contact Prediction) task by training a linear probe.

Key differences from V-JEPA's attentive probe:
  - HamJEPA encoder (ResNet-18) processes single images, not video clips.
    We process each frame independently, then mean-pool across frames.
  - HamJEPA outputs flat feature vectors [B, D], not patch tokens [B, N, D].
    The probe is a simple Linear/MLP, not an AttentiveClassifier.
  - For split_qp models, we can probe q-only, p-only, or qp-concat.

Experiment design (per physical property):
  1. Load frozen HamJEPA encoder from checkpoint
  2. For each sample: encode all T frames → mean pool → [B, D] feature
  3. Train linear classifier [D → 2] on readout_data features
  4. Evaluate OCP accuracy on test_data

Evaluation modes (same as V-JEPA):
  - per_property (default): separate probe per physical property
  - joint: one probe trained on all properties mixed

Usage:
  python -m eval.physion_linear_probe.eval \
      --config configs/physion_linear_probe.yaml \
      --device cuda:0
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Path setup: allow imports from HamJEPA root and V-JEPA ----
_HAMJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "V-JEPA"))
if _HAMJEPA_ROOT not in sys.path:
    sys.path.insert(0, _HAMJEPA_ROOT)
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)

from eval.models.encoder_resnet import ResNetEncoder
from eval.models.projector import IdentityProjector, MLPProjector
from eval.physion_linear_probe.utils import make_physion_frame_transform

# V-JEPA imports for Physion dataset
from src.datasets.physion_dataset import (
    make_physion_dataset,
    PHYSION_PROPERTIES,
    PROPERTY_TO_INDEX,
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


# ==============================================================================
# Probe classifier variants
# ==============================================================================

class LinearProbe(nn.Module):
    """Simple linear classifier: LayerNorm + Linear."""
    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        return self.linear(self.norm(x))


class MLPProbe(nn.Module):
    """2-layer MLP probe with hidden dim."""
    def __init__(self, in_dim: int, num_classes: int = 2, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# HamJEPA Model Loading
# ==============================================================================

def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Remove a prefix from all keys in a state dict."""
    out = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def load_hamjepa_encoder(
    training_cfg: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
    checkpoint_key: str = "encoder",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Build a HamJEPA encoder from a training config and load pretrained weights.

    Args:
        training_cfg: Parsed YAML dict used for training (contains 'model' section)
        checkpoint_path: Path to .pt checkpoint file
        device: torch device
        checkpoint_key: Key in checkpoint dict for encoder state_dict

    Returns:
        (encoder, info): frozen encoder module + info dict with metadata
    """
    mcfg = training_cfg["model"]

    encoder = ResNetEncoder(
        out_dim=int(mcfg["embed_dim"]),
        mode=str(mcfg["encoder_mode"]),
        token_layer=str(mcfg.get("token_layer", "layer3")),
        token_d_f=int(mcfg.get("token_d_f", 32)),
        token_hw=(
            int(mcfg["token_hw"])
            if "token_hw" in mcfg and mcfg["token_hw"] is not None
            else None
        ),
        stem=str(mcfg.get("encoder_stem", "imagenet")),
        split_qp=bool(mcfg.get("split_qp", False)),
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Extract encoder state_dict
    if isinstance(ckpt, dict):
        if checkpoint_key in ckpt:
            sd = ckpt[checkpoint_key]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Try various key prefixes to match encoder parameters
    candidates = [
        sd,
        _strip_prefix(sd, "encoder."),
        _strip_prefix(sd, "module."),
        _strip_prefix(sd, "module.encoder."),
    ]

    info = {}
    loaded = False
    for i, cand in enumerate(candidates):
        missing, unexpected = encoder.load_state_dict(cand, strict=False)
        if len(missing) == 0 and len(unexpected) == 0:
            info["loaded_from_try"] = i
            loaded = True
            break
        info[f"try_{i}"] = {"missing": len(missing), "unexpected": len(unexpected)}

    if not loaded:
        logger.warning(f"Encoder state_dict did not match perfectly: {info}")
        # Try last candidate anyway
        encoder.load_state_dict(candidates[-1], strict=False)

    # Freeze
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Parse metadata
    embed_dim = int(mcfg["embed_dim"])
    split_qp = bool(mcfg.get("split_qp", False))
    q_dim = embed_dim // 2 if split_qp else embed_dim

    info.update({
        "embed_dim": embed_dim,
        "split_qp": split_qp,
        "q_dim": q_dim,
        "encoder_mode": mcfg["encoder_mode"],
    })

    logger.info(f"Loaded HamJEPA encoder: embed_dim={embed_dim}, split_qp={split_qp}, "
                f"mode={mcfg['encoder_mode']}")
    return encoder, info


# ==============================================================================
# Data Loading
# ==============================================================================

def _make_dataloaders(
    train_csv: str,
    val_csv: str,
    crop_size: int,
    frames_per_clip: int,
    frame_step: int,
    eval_duration: Optional[float],
    batch_size: int,
    properties_filter: Optional[List[str]],
    random_horizontal_flip: bool = False,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create Physion++ train/val DataLoaders for HamJEPA evaluation.

    Uses the same PhysionDataset from V-JEPA but with frame-level transforms.
    """

    train_transform = make_physion_frame_transform(
        training=True,
        crop_size=crop_size,
        random_horizontal_flip=random_horizontal_flip,
    )

    val_transform = make_physion_frame_transform(
        training=False,
        crop_size=crop_size,
        random_horizontal_flip=False,
    )

    train_loader, _ = make_physion_dataset(
        data_paths=[train_csv],
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=1,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        duration=eval_duration,
        transform=train_transform,
        shared_transform=None,
        rank=0,
        world_size=1,
        num_workers=num_workers,
        pin_mem=True,
        drop_last=False,
        properties=properties_filter,
        return_property_label=False,
    )

    val_loader, _ = make_physion_dataset(
        data_paths=[val_csv],
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=1,
        random_clip_sampling=False,
        allow_clip_overlap=False,
        duration=eval_duration,
        transform=val_transform,
        shared_transform=None,
        rank=0,
        world_size=1,
        num_workers=num_workers,
        pin_mem=True,
        drop_last=False,
        properties=properties_filter,
        return_property_label=False,
    )

    return train_loader, val_loader


# ==============================================================================
# Per-frame encoding with temporal pooling
# ==============================================================================

@torch.no_grad()
def encode_video_frames(
    encoder: nn.Module,
    clip_frames: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode all frames of a video clip through the image-based HamJEPA encoder.

    Args:
        encoder: HamJEPA ResNet encoder (takes [B, C, H, W] images)
        clip_frames: [T, C, H, W] -- T frames from a single video clip
        device: torch device

    Returns:
        torch.Tensor: [D] -- mean-pooled feature vector across frames
    """
    T = clip_frames.shape[0]
    if T == 0:
        return torch.empty(0, device=device)

    # Encode each frame independently
    frames = clip_frames.to(device=device, dtype=next(encoder.parameters()).dtype)
    # [T, C, H, W] → encoder → [T, D]
    feats = encoder(frames)  # ResNetEncoder handles batch dim

    # Mean-pool across frames → [D]
    feat = feats.mean(dim=0)
    return feat


# ==============================================================================
# Feature slicing for split_qp models
# ==============================================================================

def slice_qp_variant(feats: torch.Tensor, variant: str, q_dim: int) -> torch.Tensor:
    """
    Slice features for q/p/qp variants when split_qp=True.

    Args:
        feats: [B, D] where D = 2*q_dim if split_qp
        variant: 'q', 'p', or 'qp'
        q_dim: dimension of q (and p)

    Returns:
        [B, d] where d = q_dim for 'q'/'p', D for 'qp'
    """
    if variant == "q":
        return feats[:, :q_dim]
    elif variant == "p":
        return feats[:, q_dim:]
    elif variant in ("qp", "concat"):
        return feats
    else:
        raise ValueError(f"Unknown qp_variant='{variant}'. Use 'q', 'p', or 'qp'.")


# ==============================================================================
# Training & Evaluation
# ==============================================================================

def run_one_epoch(
    device: torch.device,
    training: bool,
    encoder: nn.Module,
    probe: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    data_loader: torch.utils.data.DataLoader,
    q_dim: int,
    qp_variant: str,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """
    Run one epoch of Physion++ probe training or evaluation.

    For each video clip:
      1. Decode T frames
      2. Encode each frame through frozen HamJEPA encoder → [T, D]
      3. Mean-pool → [D]
      4. Slice by qp_variant (if split_qp) → [d]
      5. Classify → [2]

    Returns:
        (avg_loss, accuracy_pct)
    """
    probe.train(mode=training)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for itr, data in enumerate(data_loader):
        if training:
            optimizer.zero_grad(set_to_none=True)

        # ---- Physion++ data format ----
        # data[0]: video frames (list of lists of tensors or tensor)
        # data[1]: OCP labels [B]
        clips = data[0]
        labels = data[1].to(device)
        bs = len(labels)

        # ---- Unwrap nested list from PhysionDataset ----
        # Expected: [[tensor[C,T,H,W]], ...] or tensor [B, C, T, H, W]
        if isinstance(clips, list):
            # Process each sample in batch individually
            batch_feats = []
            for i in range(bs):
                item = clips[i]
                if isinstance(item, list):
                    if len(item) == 0:
                        continue
                    frame_tensor = item[0]  # tensor [C, T, H, W]
                else:
                    frame_tensor = item

                if not isinstance(frame_tensor, torch.Tensor):
                    continue

                # frame_tensor: [C, T, H, W] → permute to [T, C, H, W]
                if frame_tensor.ndim == 4:
                    # [C, T, H, W] → [T, C, H, W]
                    frame_tensor = frame_tensor.permute(1, 0, 2, 3)

                with torch.no_grad():
                    feat = encode_video_frames(encoder, frame_tensor, device)
                batch_feats.append(feat)

            if len(batch_feats) == 0:
                continue
            feats = torch.stack(batch_feats)  # [B, D]
        else:
            # Direct tensor format
            clips = clips.to(device=device, dtype=next(encoder.parameters()).dtype)
            # [B, C, T, H, W] → process per sample
            batch_feats = []
            for i in range(bs):
                frame_tensor = clips[i].permute(1, 0, 2, 3)  # [C,T,H,W] → [T,C,H,W]
                with torch.no_grad():
                    feat = encode_video_frames(encoder, frame_tensor, device)
                batch_feats.append(feat)
            feats = torch.stack(batch_feats)

        # ---- Slice q/p if split_qp ----
        feats = slice_qp_variant(feats, qp_variant, q_dim)

        # ---- Forward ----
        logits = probe(feats.float())
        loss = criterion(logits, labels)

        # ---- Backward ----
        if training:
            loss.backward()
            optimizer.step()

        # ---- Metrics ----
        with torch.no_grad():
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += bs

        if itr % 20 == 0:
            current_acc = 100.0 * total_correct / max(total_samples, 1)
            logger.info(f'  [{itr:5d}] loss={loss.item():.3f}  '
                        f'acc={current_acc:.1f}%')

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = 100.0 * total_correct / max(total_samples, 1)
    return avg_loss, accuracy


# ==============================================================================
# Single Property Evaluation
# ==============================================================================

def evaluate_property(
    device: torch.device,
    encoder: nn.Module,
    encoder_info: Dict[str, Any],
    train_csv: str,
    val_csv: str,
    crop_size: int,
    frames_per_clip: int,
    frame_step: int,
    eval_duration: Optional[float],
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    probe_type: str,
    qp_variant: str,
    out_dir: str,
    tag: str,
    property_name: str,
    num_workers: int = 4,
) -> Dict[str, Any]:
    """
    Train and evaluate a linear/MLP probe for a single physical property.

    Args:
        property_name: 'mass', 'friction', 'elasticity', 'deformability', or 'all'

    Returns:
        dict with best_val_acc, best_epoch, etc.
    """
    embed_dim = encoder_info["embed_dim"]
    split_qp = encoder_info["split_qp"]
    q_dim = encoder_info["q_dim"]

    # Effective feature dimension after q/p slicing
    if qp_variant in ("q", "p") and split_qp:
        feat_dim = q_dim
    else:
        feat_dim = embed_dim

    # ---- Create probe ----
    if probe_type == "linear":
        probe = LinearProbe(feat_dim, num_classes=2).to(device)
        logger.info(f"Using LinearProbe ({feat_dim} → 2)")
    elif probe_type == "mlp":
        probe = MLPProbe(feat_dim, num_classes=2, hidden_dim=512).to(device)
        logger.info(f"Using MLPProbe ({feat_dim} → 512 → 2)")
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")

    # ---- Create dataloaders ----
    props_filter = None if property_name == "all" else [property_name]
    train_loader, val_loader = _make_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        crop_size=crop_size,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=eval_duration,
        batch_size=batch_size,
        properties_filter=props_filter,
        random_horizontal_flip=False,
        num_workers=num_workers,
    )
    logger.info(f"Property [{property_name}]: {len(train_loader)} train iters/epoch, "
                f"{len(val_loader)} val iters")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader)
    )

    # ---- Training loop ----
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(num_epochs):
        logger.info(f"[{property_name}] Epoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc = run_one_epoch(
            device=device,
            training=True,
            encoder=encoder,
            probe=probe,
            optimizer=optimizer,
            data_loader=train_loader,
            q_dim=q_dim,
            qp_variant=qp_variant,
            use_amp=use_amp,
        )
        scheduler.step()

        # Validate
        val_loss, val_acc = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            probe=probe,
            optimizer=None,
            data_loader=val_loader,
            q_dim=q_dim,
            qp_variant=qp_variant,
            use_amp=False,
        )

        logger.info(
            f"[{property_name}] Epoch {epoch + 1:3d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            # Save best checkpoint
            ckpt_path = os.path.join(
                out_dir, f"{tag}_{property_name}_{qp_variant}_best.pt"
            )
            torch.save({
                "epoch": epoch + 1,
                "probe_state": best_state,
                "val_acc": val_acc,
                "feat_dim": feat_dim,
                "qp_variant": qp_variant,
            }, ckpt_path)

    # Load best
    if best_state is not None:
        probe.load_state_dict(best_state)

    logger.info(
        f"[{property_name}] Best val_acc: {best_val_acc:.2f}% at epoch {best_epoch}"
    )

    return {
        "property": property_name,
        "feat_dim": feat_dim,
        "qp_variant": qp_variant,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
    }


# ==============================================================================
# Main entry point
# ==============================================================================

def main(args_eval: Dict[str, Any], resume_preempt: bool = False) -> None:
    """
    Main entry point for HamJEPA Physion++ linear probe evaluation.

    Args:
        args_eval: Parsed YAML configuration dict
        resume_preempt: Ignored (no checkpoint resume in this version)
    """

    # ======================================================================
    # Parse configuration
    # ======================================================================

    # ---- Pretrained model ----
    pretrain_cfg = args_eval.get("pretrain", {})
    training_config_path = pretrain_cfg.get("training_config", None)
    checkpoint_path = pretrain_cfg.get("checkpoint", None)
    checkpoint_key = pretrain_cfg.get("checkpoint_key", "encoder")

    # ---- Data ----
    data_cfg = args_eval.get("data", {})
    train_csv = data_cfg.get("dataset_train", None)
    val_csv = data_cfg.get("dataset_val", None)
    frames_per_clip = data_cfg.get("frames_per_clip", 16)
    frame_step = data_cfg.get("frame_step", 4)
    eval_duration = data_cfg.get("clip_duration", None)

    # ---- Evaluation mode ----
    eval_mode = args_eval.get("eval_mode", "per_property")
    properties_to_eval = args_eval.get("properties", None)

    # ---- Optimization ----
    opt_cfg = args_eval.get("optimization", {})
    crop_size = opt_cfg.get("resolution", 224)
    batch_size = opt_cfg.get("batch_size", 32)
    num_epochs = opt_cfg.get("num_epochs", 50)
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    use_amp = opt_cfg.get("use_bfloat16", False)
    num_workers = opt_cfg.get("num_workers", 4)

    # ---- Probe ----
    probe_type = args_eval.get("probe_type", "linear")
    qp_variant = args_eval.get("qp_variant", "qp")
    tag = args_eval.get("tag", "hjepa_probe")

    # ======================================================================
    # Validate config
    # ======================================================================
    if training_config_path is None:
        raise ValueError("pretrain.training_config is required")
    if checkpoint_path is None:
        raise ValueError("pretrain.checkpoint is required")
    if train_csv is None or val_csv is None:
        raise ValueError("data.dataset_train and data.dataset_val are required")

    # ======================================================================
    # Device
    # ======================================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    logger.info(f"Using device: {device}")

    # ======================================================================
    # Load training config (YAML)
    # ======================================================================
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    with open(training_config_path, "r") as f:
        training_cfg = yaml.safe_load(f)

    # ======================================================================
    # Load encoder
    # ======================================================================
    encoder, encoder_info = load_hamjepa_encoder(
        training_cfg=training_cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        checkpoint_key=checkpoint_key,
    )

    # ======================================================================
    # Determine properties to evaluate
    # ======================================================================
    if properties_to_eval is None:
        properties_to_eval = PHYSION_PROPERTIES
    logger.info(f"Properties to evaluate: {properties_to_eval}")
    logger.info(f"Eval mode: {eval_mode}")
    logger.info(f"QP variant: {qp_variant}")
    logger.info(f"Probe type: {probe_type}")

    # ======================================================================
    # Output directory
    # ======================================================================
    out_dir = os.path.join(os.path.dirname(checkpoint_path), "physion_linear_probe")
    os.makedirs(out_dir, exist_ok=True)

    # ======================================================================
    # Run evaluation
    # ======================================================================
    all_results = {}

    if eval_mode == "joint":
        results = evaluate_property(
            device=device,
            encoder=encoder,
            encoder_info=encoder_info,
            train_csv=train_csv,
            val_csv=val_csv,
            crop_size=crop_size,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            eval_duration=eval_duration,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            use_amp=use_amp,
            probe_type=probe_type,
            qp_variant=qp_variant,
            out_dir=out_dir,
            tag=tag,
            property_name="all",
            num_workers=num_workers,
        )
        all_results["all"] = results

    elif eval_mode == "per_property":
        for prop in properties_to_eval:
            logger.info(f"\n{'='*60}\nEvaluating: {prop}\n{'='*60}")
            results = evaluate_property(
                device=device,
                encoder=encoder,
                encoder_info=encoder_info,
                train_csv=train_csv,
                val_csv=val_csv,
                crop_size=crop_size,
                frames_per_clip=frames_per_clip,
                frame_step=frame_step,
                eval_duration=eval_duration,
                batch_size=batch_size,
                num_epochs=num_epochs,
                lr=lr,
                weight_decay=weight_decay,
                use_amp=use_amp,
                probe_type=probe_type,
                qp_variant=qp_variant,
                out_dir=out_dir,
                tag=tag,
                property_name=prop,
                num_workers=num_workers,
            )
            all_results[prop] = results

    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")

    # ======================================================================
    # Report
    # ======================================================================
    _report_results(out_dir, tag, qp_variant, probe_type, all_results)


def _report_results(
    out_dir: str,
    tag: str,
    qp_variant: str,
    probe_type: str,
    all_results: Dict[str, Any],
) -> None:
    """Print and save evaluation results."""
    results_path = os.path.join(
        out_dir, f"{tag}_{qp_variant}_{probe_type}_results.txt"
    )

    lines = []
    lines.append("=" * 70)
    lines.append("HamJEPA Physion++ Linear Probe Evaluation Results")
    lines.append("=" * 70)
    lines.append(f"  Tag:        {tag}")
    lines.append(f"  QP variant: {qp_variant}")
    lines.append(f"  Probe type: {probe_type}")
    lines.append("-" * 70)

    total_acc = 0.0
    n_props = 0
    for prop, result in all_results.items():
        acc = result["best_val_acc"]
        total_acc += acc
        n_props += 1
        lines.append(
            f"  {prop:20s}: {acc:.2f}% "
            f"(epoch {result['best_epoch']}, dim={result['feat_dim']})"
        )

    if n_props > 0:
        avg_acc = total_acc / n_props
        lines.append("-" * 70)
        lines.append(f"  {'Average':20s}: {avg_acc:.2f}%")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print("\n" + report)

    with open(results_path, "w") as f:
        f.write(report + "\n")
    logger.info(f"Results saved to: {results_path}")


# ==============================================================================
# CLI entry point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HamJEPA Physion++ Linear Probe Evaluation"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to run on"
    )
    parser.add_argument(
        "--qp_variant", type=str, default=None,
        help="Override qp_variant from config (q, p, or qp)"
    )
    args = parser.parse_args()

    # Load YAML config
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. pip install pyyaml")

    with open(args.config, "r") as f:
        eval_cfg = yaml.safe_load(f)

    # CLI overrides
    if args.qp_variant is not None:
        eval_cfg["qp_variant"] = args.qp_variant

    main(eval_cfg)

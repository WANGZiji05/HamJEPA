"""
HamJEPA Physion++ Latent Dynamics (Rollout) Evaluation
=======================================================

Evaluates the HamJEPA Hamiltonian flow predictor's ability to model
temporal dynamics in Physion++ videos.

Core experiment -- Prediction Horizon Sweep:
  For each video, sample frame pairs at increasing temporal gaps (k = 1..7).
  For each pair (z_t, z_{t+k}):
    - Predict z_{t+k} from z_t using HamiltonianFlowPredictor
    - Measure cosine similarity between prediction and actual z_{t+k}

  Larger k → harder prediction → lower similarity.
  Slower decay = better dynamics modeling.

Auxiliary experiment -- Shuffled Baseline:
  For each horizon, randomly pair z_t with z_j from a DIFFERENT video.
  If predictor captures true dynamics, real-pair similarity should exceed
  shuffled-pair similarity.

Key differences from V-JEPA's rollout:
  - V-JEPA predictor uses diffusion (context + noisy target → clean target)
  - HamJEPA predictor uses Hamiltonian flow (z0 → integrate → zT)
  - HamJEPA wraps a predictor-free baseline (identity map): z_t vs z_{t+k}
    to isolate "how much does the flow help vs just encoding similarity?"

Per-property evaluation:
  Separately for mass, friction, elasticity, deformability.

Usage:
  python -m eval.physion_rollout.eval \
      --config configs/physion_rollout.yaml \
      --device cuda:0
"""

from __future__ import annotations

import sys
import types

# ── Fake requests module BEFORE any torchvision import ──
if "requests" not in sys.modules:
    sys.modules["requests"] = types.ModuleType("requests")

import os
import logging
import argparse
import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---- Path setup ----
_HAMJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VJEPA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "V-JEPA"))
if _HAMJEPA_ROOT not in sys.path:
    sys.path.insert(0, _HAMJEPA_ROOT)
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)

from eval.models.encoder_resnet import ResNetEncoder
from hamjepa.predictor import HamiltonianFlowPredictor

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

PHYSION_PROPS = ["mass", "friction", "elasticity", "deformability"]

# ImageNet normalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default horizons: temporal gap between frames (in frame_steps)
# frame_step=4 means k=1 is 4 actual frames apart
HORIZONS = [1, 2, 3, 4, 5, 6, 7]


# ============================================================================
# Model loading
# ============================================================================

def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def load_models(
    training_cfg: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Optional[nn.Module], Dict[str, Any]]:
    """
    Load HamJEPA encoder and predictor from checkpoint.

    Returns:
        (encoder, predictor, info)
    """
    mcfg = training_cfg["model"]
    hcfg = training_cfg.get("hjepa", {})

    # ---- Encoder ----
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
    ).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    embed_dim = int(mcfg["embed_dim"])
    split_qp = bool(mcfg.get("split_qp", False))
    q_dim = embed_dim // 2 if split_qp else embed_dim

    # ---- Predictor ----
    predictor = None
    has_predictor = (
        hcfg.get("hamiltonian") is not None
        or ("hjepa" in training_cfg and training_cfg["hjepa"].get("method") is not None)
    )

    if has_predictor:
        predictor = HamiltonianFlowPredictor(
            state_dim=embed_dim,
            hamiltonian=str(hcfg.get("hamiltonian", "separable")),
            hidden_dim=int(hcfg.get("hidden_dim", 256)),
            depth=int(hcfg.get("depth", 2)),
            activation=str(hcfg.get("activation", "gelu")),
            residual_scale=float(hcfg.get("residual_scale", 0.01)),
            base_coeff=float(hcfg.get("base_coeff", 1.0)),
            method=str(hcfg.get("method", "leapfrog")),
            steps=int(hcfg.get("steps", 2)),
            dt=float(hcfg.get("dt", 0.05)),
            learn_dt=bool(hcfg.get("learn_dt", False)),
            integrate_fp32=bool(hcfg.get("integrate_fp32", True)),
        ).to(device).eval()
        for p in predictor.parameters():
            p.requires_grad = False
        logger.info(f"Predictor: {hcfg.get('hamiltonian')}, {hcfg.get('steps')} steps, "
                    f"dt={hcfg.get('dt')}")

    # ---- Load checkpoint ----
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Encoder
    if "encoder" in ckpt:
        enc_sd = ckpt["encoder"]
    else:
        enc_sd = ckpt

    candidates = [
        enc_sd,
        _strip_prefix(enc_sd, "encoder."),
        _strip_prefix(enc_sd, "module."),
    ]
    for cand in candidates:
        missing, unexpected = encoder.load_state_dict(cand, strict=False)
        if len(missing) == 0 and len(unexpected) == 0:
            break

    # Predictor
    if predictor is not None and "predictor" in ckpt:
        pred_sd = ckpt["predictor"]
        pred_candidates = [
            pred_sd,
            _strip_prefix(pred_sd, "predictor."),
            _strip_prefix(pred_sd, "module."),
        ]
        for cand in pred_candidates:
            try:
                predictor.load_state_dict(cand, strict=False)
                break
            except Exception:
                continue
        logger.info("Predictor weights loaded")
    elif predictor is not None:
        logger.warning("No 'predictor' key in checkpoint! Predictor uses random init.")

    info = {
        "embed_dim": embed_dim,
        "split_qp": split_qp,
        "q_dim": q_dim,
        "has_predictor": predictor is not None,
    }
    return encoder, predictor, info


# ============================================================================
# Video loading helpers (minimal, avoids V-JEPA dependency)
# ============================================================================

def _load_video_frames(
    vpath: str,
    num_frames: int,
    frame_step: int,
    start_frame: int,
    crop_size: int,
) -> Optional[torch.Tensor]:
    """
    Load and preprocess a clip of frames from a video file.

    Args:
        vpath: path to MP4 file
        num_frames: number of frames to load
        frame_step: step between sampled frames
        start_frame: first frame index
        crop_size: target spatial size

    Returns:
        Tensor [T, C, H, W] of normalized frames, or None on failure
    """
    try:
        from decord import VideoReader, cpu as decord_cpu
    except ImportError:
        logger.error("decord is required for video loading")
        return None

    try:
        vr = VideoReader(vpath, num_threads=1, ctx=decord_cpu(0))
    except Exception:
        return None

    total = len(vr)
    indices = np.clip(
        np.arange(start_frame, start_frame + num_frames * frame_step, frame_step),
        0, total - 1
    ).astype(np.int64)

    try:
        buf = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    except Exception:
        return None

    T, H, W, C = buf.shape

    # Resize (short side → crop_size * 256/224, then center crop)
    short = int(crop_size * 256 / 224)
    scale = short / min(H, W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))

    # Simple resize (nearest-neighbor for speed in eval)
    resized = np.zeros((T, new_h, new_w, C), dtype=buf.dtype)
    for t in range(T):
        for i in range(new_h):
            si = min(int(i / scale), H - 1)
            resized[t, i] = buf[t, si]
        for j in range(new_w):
            sj = min(int(j / scale), W - 1)
            resized[t, :, j] = buf[t, :, sj]

    # Center crop
    hs = (new_h - crop_size) // 2
    ws = (new_w - crop_size) // 2
    buf = resized[:, hs:hs + crop_size, ws:ws + crop_size, :]

    # Normalize
    buf = buf.astype(np.float32) / 255.0
    buf = (buf - _MEAN) / _STD

    # [T, H, W, C] → [T, C, H, W]
    return torch.from_numpy(buf).permute(0, 3, 1, 2)


# ============================================================================
# Core evaluation
# ============================================================================

@torch.no_grad()
def evaluate_property_rollout(
    device: torch.device,
    encoder: nn.Module,
    predictor: Optional[nn.Module],
    video_paths: List[str],
    horizons: List[int],
    num_frames: int,
    frame_step: int,
    crop_size: int,
    max_videos: int = 200,
) -> Dict[int, Dict[str, Any]]:
    """
    Evaluate rollout dynamics for a list of videos belonging to one property.

    For each video:
      1. Load num_frames + max(horizons) frames from the middle
      2. Encode all frames → features
      3. For each horizon k:
         a. Real pair: predict(z_t → z_{t+k}), measure cos_sim(pred, actual z_{t+k})
         b. Identity baseline: cos_sim(z_t, z_{t+k}) -- "how similar are the frames?"
         c. Shuffled baseline: cos_sim(predict(z_t), z_j) where z_j is from a DIFFERENT video

    Returns:
        {horizon: {"real_mean", "identity_mean", "shuffled_mean", "delta_real", "delta_shuf", "n"}}
    """
    np.random.seed(42)

    results: Dict[int, Dict[str, List[float]]] = {
        h: {"real": [], "identity": [], "shuffled_pred": [], "shuffled_tgt": []}
        for h in horizons
    }

    # Phase 1: Encode all valid videos and collect features
    all_features = []  # List of [T, D] tensors
    valid_paths = []

    for vi, vpath in enumerate(video_paths):
        if len(valid_paths) >= max_videos:
            break

        max_gap = max(horizons)
        total_needed = num_frames + max_gap * frame_step

        # Load enough frames from the middle of the video
        try:
            from decord import VideoReader, cpu as decord_cpu
            vr = VideoReader(vpath, num_threads=1, ctx=decord_cpu(0))
            total_len = len(vr)
        except Exception:
            continue

        if total_len < total_needed:
            continue

        # Start from middle region
        start_frame = max(0, (total_len - total_needed) // 2)
        frames = _load_video_frames(
            vpath, num_frames + max_gap, frame_step, start_frame, crop_size
        )
        if frames is None or frames.shape[0] < num_frames + max_gap:
            continue

        # Encode each frame independently
        frames = frames.to(device=device, dtype=next(encoder.parameters()).dtype)
        # frames: [T, C, H, W] → encoder → [T, D]
        feats = encoder(frames)  # ResNetEncoder processes batch
        all_features.append(feats.cpu())
        valid_paths.append(vpath)

        if (vi + 1) % 50 == 0:
            logger.info(f"  Encoded {len(valid_paths)} videos...")

    if len(all_features) < 10:
        logger.warning(f"  Only {len(all_features)} valid videos (need >= 10)")
        return {h: {"real_mean": 0, "n": 0} for h in horizons}

    logger.info(f"  Encoded {len(all_features)} videos for property evaluation")

    # Phase 2: Compute real-pair and identity similarities for each horizon
    feat_data = all_features  # list of [T_i, D]

    for h in horizons:
        real_sims = []
        identity_sims = []

        for feats in feat_data:
            T = feats.shape[0]
            if T <= h:
                continue
            # Sample pairs: for each possible t where t+h < T
            max_t = T - h
            for t_idx in range(0, max_t, 1):  # sample all possible pairs
                z_t = feats[t_idx:t_idx + 1].to(device)  # [1, D]
                z_th = feats[t_idx + h:t_idx + h + 1].to(device)  # [1, D]

                # Real prediction (needs grad for dV/dq)
                if predictor is not None:
                    with torch.enable_grad():
                        z_pred = predictor(z_t.requires_grad_(True), direction=1)
                    real_sim = F.cosine_similarity(z_pred.detach(), z_th, dim=-1).mean().item()
                else:
                    # No predictor: use identity baseline
                    real_sim = F.cosine_similarity(z_t, z_th, dim=-1).mean().item()
                real_sims.append(real_sim)

                # Identity baseline (no predictor)
                id_sim = F.cosine_similarity(z_t, z_th, dim=-1).mean().item()
                identity_sims.append(id_sim)

        results[h]["real"] = real_sims
        results[h]["identity"] = identity_sims

    # Phase 3: Shuffled baseline (cross-video pairing)
    # Take a subset of cached features, shuffle targets
    n_shuffle = min(len(feat_data), 100)

    # Use first frame from each video as anchor, random frame from another video as target
    anchor_feats = []
    target_feats = []
    for i in range(n_shuffle):
        feats_i = feat_data[i]
        t_idx = 0  # first frame as anchor
        anchor_feats.append(feats_i[t_idx:t_idx + 1])

        # Random target from a different video
        j = (i + np.random.randint(1, n_shuffle)) % n_shuffle
        feats_j = feat_data[j]
        rand_t = np.random.randint(0, feats_j.shape[0])
        target_feats.append(feats_j[rand_t:rand_t + 1])

    anchors = torch.cat([f.to(device) for f in anchor_feats], dim=0)  # [N, D]
    targets = torch.cat([f.to(device) for f in target_feats], dim=0)  # [N, D]

    if predictor is not None:
        with torch.enable_grad():
            pred_shuffled = predictor(anchors.requires_grad_(True), direction=1)
        shuff_sims = F.cosine_similarity(pred_shuffled.detach(), targets, dim=-1).tolist()
    else:
        shuff_sims = F.cosine_similarity(anchors, targets, dim=-1).tolist()

    # Compute averages
    summary = {}
    for h in horizons:
        real = results[h]["real"]
        identity = results[h]["identity"]

        if len(real) == 0:
            summary[h] = {"real_mean": 0, "identity_mean": 0,
                          "shuffled_mean": 0, "delta_pred": 0, "n": 0}
            continue

        real_mean = float(np.mean(real))
        id_mean = float(np.mean(identity))
        shuff_mean = float(np.mean(shuff_sims))

        # Delta: how much does the predictor improve over identity?
        delta_pred = real_mean - id_mean
        # Delta vs shuffled: how much does correct pairing help?
        delta_shuf = real_mean - shuff_mean

        summary[h] = {
            "real_mean": real_mean,
            "identity_mean": id_mean,
            "shuffled_mean": shuff_mean,
            "delta_pred": delta_pred,       # predictor gain over identity
            "delta_shuf": delta_shuf,        # correct-pairing gain over shuffled
            "n": len(real),
        }

        s = summary[h]
        logger.info(
            f"  H={h}: real={real_mean:.4f}  identity={id_mean:.4f}  "
            f"shuffled={shuff_mean:.4f}  "
            f"Δ_pred={delta_pred:+.4f}  Δ_shuf={delta_shuf:+.4f}  "
            f"(n={len(real)})"
        )

    return summary


# ============================================================================
# Main
# ============================================================================

def main(args_eval: Dict[str, Any], resume_preempt: bool = False) -> None:
    """Main entry point."""

    # ---- Config ----
    pretrain_cfg = args_eval.get("pretrain", {})
    training_config_path = pretrain_cfg.get("training_config", None)
    checkpoint_path = pretrain_cfg.get("checkpoint", None)

    data_cfg = args_eval.get("data", {})
    test_csv = data_cfg.get("dataset", None)
    frames_per_clip = data_cfg.get("frames_per_clip", 16)
    frame_step = data_cfg.get("frame_step", 4)

    opt_cfg = args_eval.get("optimization", {})
    crop_size = opt_cfg.get("resolution", 224)

    properties_to_eval = args_eval.get("properties", None) or PHYSION_PROPS
    horizons = args_eval.get("horizons", HORIZONS)
    max_videos = args_eval.get("max_videos", 200)
    tag = args_eval.get("tag", "hjepa_rollout")

    # ---- Device ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    logger.info(f"Device: {device}")

    # ---- Load training config ----
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required")

    with open(training_config_path, "r") as f:
        training_cfg = yaml.safe_load(f)

    # ---- Load models ----
    encoder, predictor, info = load_models(
        training_cfg=training_cfg,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # ---- Load video paths from CSV ----
    videos = {p: [] for p in PHYSION_PROPS}
    with open(test_csv) as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            path, prop, label = row[0], row[1].strip(), int(row[2])
            if prop in videos and label >= 0:
                videos[prop].append(path)

    total = sum(len(v) for v in videos.values())
    logger.info(f"Loaded {total} videos from test CSV")
    for prop in PHYSION_PROPS:
        logger.info(f"  {prop}: {len(videos[prop])} videos")

    # ---- Evaluate per property ----
    all_results = {}
    for prop in properties_to_eval:
        logger.info(f"\n{'='*60}\nProperty: {prop}\n{'='*60}")
        vpaths = videos.get(prop, [])
        if len(vpaths) < 10:
            logger.warning(f"  Skipping {prop}: only {len(vpaths)} videos")
            continue
        np.random.shuffle(vpaths)
        all_results[prop] = evaluate_property_rollout(
            device=device,
            encoder=encoder,
            predictor=predictor,
            video_paths=vpaths[:max_videos],
            horizons=horizons,
            num_frames=frames_per_clip,
            frame_step=frame_step,
            crop_size=crop_size,
            max_videos=max_videos,
        )

    # ---- Report ----
    out_dir = os.path.join(os.path.dirname(checkpoint_path), "physion_rollout")
    os.makedirs(out_dir, exist_ok=True)
    _report(out_dir, tag, all_results, horizons, info)


def _report(out_dir, tag, all_results, horizons, info):
    """Generate and save the results report."""
    path = os.path.join(out_dir, f"{tag}_dynamics_results.txt")
    lines = [
        "=" * 85,
        "HamJEPA Physion++ Latent Dynamics Evaluation",
        "=" * 85,
        f"  Has predictor: {info['has_predictor']}",
        f"  split_qp:      {info['split_qp']}",
        "",
        "  Δ_pred = RealPair - Identity  (predictor gain over no-predictor baseline)",
        "  Δ_shuf = RealPair - Shuffled  (correct-pairing gain over random pairs)",
        "  Δ_shuf > 0 且 Δ_pred > 0 → predictor captures meaningful dynamics",
        "",
    ]

    for prop in PHYSION_PROPS:
        r = all_results.get(prop, {})
        if not r:
            continue
        lines.append(f"  --- {prop} ---")
        lines.append(
            f"  {'H':>5}  {'Real':>8}  {'Identity':>8}  "
            f"{'Shuffled':>8}  {'Δ_pred':>8}  {'Δ_shuf':>8}  {'N':>5}"
        )
        for h in horizons:
            rh = r.get(h, {})
            if rh.get("n", 0) == 0:
                continue
            lines.append(
                f"  {h:>5}  {rh['real_mean']:>8.4f}  "
                f"{rh['identity_mean']:>8.4f}  "
                f"{rh['shuffled_mean']:>8.4f}  "
                f"{rh['delta_pred']:>+8.4f}  {rh['delta_shuf']:>+8.4f}  "
                f"{rh['n']:>5}"
            )
        lines.append("")

    # Cross-property comparison
    lines.append("  " + "=" * 72)
    lines.append("  Cross-property: Δ_pred / Δ_shuf")
    header = f"  {'H':>5}"
    for prop in PHYSION_PROPS:
        header += f"  {prop:>18}"
    lines.append(header)
    for h in horizons:
        row = f"  {h:>5}"
        for prop in PHYSION_PROPS:
            rh = all_results.get(prop, {}).get(h, {})
            if rh.get("n", 0) > 0:
                row += f"  {rh['delta_pred']:+.4f}/{rh['delta_shuf']:+.4f}"
            else:
                row += f"  {'N/A':>18}"
        lines.append(row)
    lines.append("=" * 85)

    report = "\n".join(lines)
    print("\n" + report)
    with open(path, "w") as f:
        f.write(report + "\n")
    logger.info(f"Saved to: {path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HamJEPA Physion++ Latent Dynamics (Rollout) Evaluation"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required")

    with open(args.config, "r") as f:
        eval_cfg = yaml.safe_load(f)

    main(eval_cfg)

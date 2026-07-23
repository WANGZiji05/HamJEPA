"""
Minimal training script with config-driven options for baseline / SIGReg / HamSIGReg.

Example:
  python scripts/train_cifar_hamjepa.py --config configs/cifar100_hjepa_mv.yaml
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.datasets.imagenet_multicrop import ImageNetMultiCrop, MultiCropCfg
from eval.models.encoder_resnet import ResNetEncoder
from eval.models.projector import IdentityProjector, MLPProjector
from hamjepa.losses import (
    HamiltonianConsistencyLoss,
    PhaseSpaceEnergyBudget,
    ProjectedLogDetFloor,
    VarianceFloor,
)
from hamjepa.predictor import HamiltonianFlowPredictor
from lejepa.hamiltonian.ham_sigreg import HamSIGReg
from lejepa.losses import SIGReg

try:
    import yaml
except ImportError as e:  # pragma: no cover - guidance for user
    raise ImportError("Please install pyyaml to use the training config loader.") from e


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_requires_grad(params, flag: bool) -> None:
    for p in params:
        p.requires_grad_(flag)


def max_abs_grad(params) -> float:
    mx = 0.0
    for p in params:
        if p.grad is not None:
            mx = max(mx, float(p.grad.detach().abs().max().item()))
    return mx


def save_checkpoint(
    path: str,
    *,
    cfg: Dict[str, Any],
    epoch: int,
    encoder: torch.nn.Module,
    projector: torch.nn.Module,
    reg_module: Optional[torch.nn.Module],
    predictor: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    global_step: int = 0,
) -> None:
    ckpt_dict = {
        "config": cfg,
        "epoch": epoch,
        "global_step": global_step,
        "encoder": encoder.state_dict(),
        "projector": projector.state_dict(),
        "regularizer": reg_module.state_dict() if reg_module is not None else None,
        "predictor": predictor.state_dict() if predictor is not None else None,
    }
    if optimizer is not None:
        ckpt_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt_dict["scheduler"] = scheduler.state_dict()
    torch.save(ckpt_dict, path)


def config_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def lejepa_prediction_loss(z_views: torch.Tensor, num_global_views: int) -> torch.Tensor:
    """
    LeJEPA prediction loss: all views predict the mean of GLOBAL views.

    z_views: [V, B, K]
    """
    if z_views.dim() != 3:
        raise ValueError(f"Expected z_views shape [V,B,K], got {tuple(z_views.shape)}")
    V, _, _ = z_views.shape
    Vg = int(num_global_views)
    if not (1 <= Vg <= V):
        raise ValueError(f"num_global_views must be in [1,{V}], got {Vg}")
    centers = z_views[:Vg].mean(dim=0)  # [B, K]
    return 0.5 * (centers.unsqueeze(0) - z_views).pow(2).mean()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def build_regularizer(cfg: Dict[str, Any], d: int, device: torch.device):
    reg_cfg = cfg.get("regularizer", {})
    reg_type = reg_cfg.get("type", "ham_sigreg")
    num_slices = reg_cfg.get("num_slices", 256)
    t_min = reg_cfg.get("t_min", None)
    t_max = reg_cfg.get("t_max", 3.0)
    t_range = (t_min, t_max) if t_min is not None else None
    sigreg_kwargs = dict(
        t_max=t_max,
        t_range=t_range,
        n_points=reg_cfg.get("n_points", reg_cfg.get("num_t", 17)),
        weight_type=reg_cfg.get("weight_type", "gaussian"),
        ddp_sync=reg_cfg.get("ddp_sync", True),
        force_fp32=reg_cfg.get("force_fp32", True),
        refresh_interval=reg_cfg.get("refresh_interval", 1),
        subsample=reg_cfg.get("subsample", None),
        reduction=reg_cfg.get("reduction", "mean"),
        clip_value=reg_cfg.get("clip_value", None),
    )

    if reg_type == "none":
        return None
    if reg_type == "sigreg":
        return SIGReg(num_slices=num_slices, **sigreg_kwargs).to(device)
    if reg_type == "ham_sigreg":
        ham_kind = reg_cfg.get("ham_kind", "chain")
        ham_kwargs = reg_cfg.get("ham_kwargs", {})
        return HamSIGReg(
            d=d,
            kind=ham_kind,
            num_slices=num_slices,
            sigreg_kwargs=sigreg_kwargs,
            device=device,
            **ham_kwargs,
        ).to(device)
    raise ValueError(f"Unknown regularizer type: {reg_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/imagenet_sigreg_tokens.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. checkpoints/latest.pth)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision") and hasattr(torch._C, "_set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception as e:
                print(f"[WARN] set_float32_matmul_precision('high') failed, continuing. err={e}")

    is_main = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        is_main = torch.distributed.get_rank() == 0

    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "./data")
    batch_size = data_cfg.get("batch_size", 256)
    num_workers = data_cfg.get("num_workers", 4)
    drop_last = data_cfg.get("drop_last", True)
    num_global_views = data_cfg.get("num_global_views", 2)
    num_local_views = data_cfg.get("num_local_views", 0)

    def _to_tuple(value, fallback):
        if value is None:
            return fallback
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
        return fallback

    global_scale = _to_tuple(data_cfg.get("global_scale", None), (0.3, 1.0))
    local_scale = _to_tuple(data_cfg.get("local_scale", None), (0.05, 0.3))
    out_size = int(data_cfg.get("out_size", 32))

    mc_cfg = MultiCropCfg(
        num_global_views=int(num_global_views),
        num_local_views=int(num_local_views),
        out_size=out_size,
        global_scale=global_scale,
        local_scale=local_scale,
    )

    dataset = ImageNetMultiCrop(root=root, split="train", cfg=mc_cfg)
    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = data_cfg.get("prefetch_factor", 2)

    def _worker_init_fn(_: int) -> None:
        # Reduce CPU thread oversubscription per worker
        torch.set_num_threads(1)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
        loader_kwargs["worker_init_fn"] = _worker_init_fn

    loader = DataLoader(dataset, **loader_kwargs)

    model_cfg = cfg.get("model", {})
    embed_dim = model_cfg.get("embed_dim", 512)
    proj_hidden = model_cfg.get("proj_hidden", 2048)
    proj_out = model_cfg.get("proj_out", 2048)
    encoder_mode = model_cfg.get("encoder_mode", "global")
    token_layer = model_cfg.get("token_layer", "layer3")
    token_d_f = model_cfg.get("token_d_f", 32)
    token_hw = model_cfg.get("token_hw", None)
    token_hw = int(token_hw) if token_hw is not None else None
    encoder_stem = str(model_cfg.get("encoder_stem", "cifar"))
    proj_type = model_cfg.get("projector_type", "mlp").lower()
    split_qp = bool(model_cfg.get("split_qp", False))

    encoder = ResNetEncoder(
        out_dim=embed_dim,
        mode=encoder_mode,
        token_layer=token_layer,
        token_d_f=token_d_f,
        token_hw=token_hw,
        stem=encoder_stem,
        split_qp=split_qp,
    ).to(device)
    if proj_type == "identity":
        projector = IdentityProjector().to(device)
    else:
        projector = MLPProjector(in_dim=embed_dim, hidden_dim=proj_hidden, out_dim=proj_out).to(device)

    rep_dim = embed_dim if proj_type == "identity" else proj_out

    use_hjepa = isinstance(cfg.get("hjepa", None), dict)
    predictor = None
    pred_loss_fn = None
    budget_fn = None
    var_floor_fn = None
    q_logdet_floor_fn = None
    p_logdet_floor_fn = None
    bidirectional = False
    lambda_budget = 0.0
    lambda_var = 0.0
    lambda_logdet = 0.0

    reg_module = None
    lambda_reg_target = float(cfg.get("train", {}).get("lambda_reg", 1e-2))
    lambda_reg_warmup_epochs = int(cfg.get("train", {}).get("lambda_reg_warmup_epochs", 0))
    reg_cfg = cfg.get("regularizer", {})
    spectral_weight = float(reg_cfg.get("spectral_weight", 0.0))

    if use_hjepa:
        if not split_qp:
            raise ValueError("MV-HJEPA requires model.split_qp: true (channel-wise q/p split).")
        if rep_dim % 2 != 0:
            raise ValueError(f"MV-HJEPA requires even embed_dim, got {rep_dim}.")
        if proj_type != "identity":
            raise ValueError("MV-HJEPA requires model.projector_type=identity (q/p semantics must be preserved).")

        hjepa_cfg = cfg.get("hjepa", {})
        loss_cfg = cfg.get("loss", {})
        reg_cfg = cfg.get("regularizer", {})
        train_cfg = cfg.get("train", {})

        predictor = HamiltonianFlowPredictor(
            state_dim=rep_dim,
            hamiltonian=hjepa_cfg.get("hamiltonian", "separable"),
            hidden_dim=hjepa_cfg.get("hidden_dim", 256),
            depth=hjepa_cfg.get("depth", 2),
            residual_scale=hjepa_cfg.get("residual_scale", 0.01),
            base_coeff=float(hjepa_cfg.get("base_coeff", 1.0)),
            method=hjepa_cfg.get("method", "leapfrog"),
            steps=int(hjepa_cfg.get("steps", 1)),
            dt=float(hjepa_cfg.get("dt", 0.1)),
            learn_dt=bool(hjepa_cfg.get("learn_dt", False)),
            integrate_fp32=bool(hjepa_cfg.get("integrate_fp32", True)),
        ).to(device)

        if predictor.raw_dt is not None:
            raise ValueError("MV-HJEPA requires learn_dt=false to avoid dt-collapse.")

        pred_loss_fn = HamiltonianConsistencyLoss(
            predictor=predictor,
            detach_target=bool(loss_cfg.get("detach_target", True)),
            energy_weight=float(loss_cfg.get("energy_weight", 0.0)),
            match=str(loss_cfg.get("match", "q")),
            p_weight=float(loss_cfg.get("p_weight", 0.0)),
        )

        budget_fn = PhaseSpaceEnergyBudget(
            state_dim=rep_dim,
            q_target=float(reg_cfg.get("q_per_dim_target", 1.0)),
            p_target=float(reg_cfg.get("p_per_dim_target", 1.0)),
            ddp_sync=bool(reg_cfg.get("ddp_sync", True)),
        )

        var_floor_fn = VarianceFloor(
            dim=rep_dim // 2,
            std_floor=float(reg_cfg.get("q_std_floor", 0.2)),
            eps=float(reg_cfg.get("eps", 1e-4)),
            ddp_sync=bool(reg_cfg.get("ddp_sync", True)),
        )

        q_logdet_floor_fn = ProjectedLogDetFloor(
            dim=rep_dim // 2,
            proj_dim=int(reg_cfg.get("q_logdet_proj_dim", 64)),
            logdet_floor=float(reg_cfg.get("q_logdet_floor", -2.0)),
            pr_norm_floor=reg_cfg.get("q_pr_norm_floor", None),
            eigmax_frac_ceiling=reg_cfg.get("q_eigmax_frac_ceiling", None),
            eps=float(reg_cfg.get("q_logdet_eps", 1e-4)),
            ddp_sync=bool(reg_cfg.get("ddp_sync", True)),
            refresh_interval=int(reg_cfg.get("q_logdet_refresh_interval", 256)),
        )

        p_logdet_floor_fn = ProjectedLogDetFloor(
            dim=rep_dim // 2,
            proj_dim=int(reg_cfg.get("p_logdet_proj_dim", reg_cfg.get("q_logdet_proj_dim", 64))),
            logdet_floor=float(reg_cfg.get("p_logdet_floor", reg_cfg.get("q_logdet_floor", -2.0))),
            pr_norm_floor=reg_cfg.get("p_pr_norm_floor", reg_cfg.get("q_pr_norm_floor", None)),
            eigmax_frac_ceiling=reg_cfg.get("p_eigmax_frac_ceiling", reg_cfg.get("q_eigmax_frac_ceiling", None)),
            eps=float(reg_cfg.get("p_logdet_eps", reg_cfg.get("q_logdet_eps", 1e-4))),
            ddp_sync=bool(reg_cfg.get("ddp_sync", True)),
            refresh_interval=int(reg_cfg.get("p_logdet_refresh_interval", reg_cfg.get("q_logdet_refresh_interval", 256))),
        )

        lambda_budget = float(train_cfg.get("lambda_budget", 0.0))
        lambda_var = float(train_cfg.get("lambda_var", 0.0))
        lambda_logdet = float(train_cfg.get("lambda_logdet", 0.0))
        lambda_mean = float(train_cfg.get("lambda_mean", 0.0))
        bidirectional = bool(loss_cfg.get("bidirectional", False))
    else:
        reg_module = build_regularizer(cfg, d=rep_dim, device=device)

    train_cfg = cfg.get("train", {})
    base_lr = train_cfg.get("lr", 1e-3)
    h_lr = float(train_cfg.get("h_lr", base_lr * 0.1))
    weight_decay = train_cfg.get("weight_decay", 0.0)
    reg_lr = train_cfg.get("reg_lr", base_lr)
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.0))
    precision = str(train_cfg.get("precision", "fp32")).lower()
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    reg_params = list(reg_module.parameters()) if reg_module is not None else []
    h_params = list(reg_module.learnable_h.parameters()) if hasattr(reg_module, "learnable_h") else []
    h_param_ids = {id(p) for p in h_params}

    main_params = list(encoder.parameters()) + list(projector.parameters())
    if predictor is not None:
        predictor_h_params = list(predictor.H.parameters()) if hasattr(predictor, "H") else []
        predictor_h_ids = {id(p) for p in predictor_h_params}
        main_params.extend([p for p in predictor.parameters() if id(p) not in predictor_h_ids])
    else:
        predictor_h_params = []
        predictor_h_ids = set()
    if reg_params:
        main_params.extend([p for p in reg_params if id(p) not in h_param_ids])

    if predictor_h_params:
        opt_main = optim.AdamW(
            [
                {"params": main_params, "lr": base_lr},
                {"params": predictor_h_params, "lr": h_lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        opt_main = optim.AdamW(main_params, lr=base_lr, weight_decay=weight_decay)
    opt_h = optim.AdamW(h_params, lr=reg_lr, weight_decay=0.0) if h_params else None

    num_epochs = train_cfg.get("epochs", 100)
    freeze_h_epochs = int(train_cfg.get("freeze_h_epochs", 0))
    h_update_interval = int(train_cfg.get("h_update_interval", 1))
    h_grad_clip = float(train_cfg.get("h_grad_clip", 0.0))
    h_grad_scale = float(train_cfg.get("h_grad_scale", 1.0))
    h_identity_epochs = int(train_cfg.get("h_identity_epochs", 0))
    h_base_ramp_epochs = int(train_cfg.get("h_base_ramp_epochs", 0))
    h_learnable_start_epoch = int(train_cfg.get("h_learnable_start_epoch", 0))
    h_learnable_ramp_epochs = int(train_cfg.get("h_learnable_ramp_epochs", 0))
    h_update_mode = str(train_cfg.get("h_update_mode", "min")).lower()
    h_adv_mult = float(train_cfg.get("h_adv_mult", 1.0))
    h_alpha_start_epoch = int(train_cfg.get("h_alpha_start_epoch", freeze_h_epochs))
    h_alpha_ramp_epochs = int(train_cfg.get("h_alpha_ramp_epochs", 0))
    h_alpha_max = float(train_cfg.get("h_alpha_max", 1.0))
    log_every = int(train_cfg.get("log_every", 0))
    ckpt_every = max(1, int(train_cfg.get("checkpoint_every", 1)))
    profile_timing = bool(train_cfg.get("profile_timing", False))
    profile_every = int(train_cfg.get("profile_every", 20))
    total_steps = num_epochs * max(1, len(loader))
    warmup_steps = warmup_epochs * max(1, len(loader))
    scheduler = None
    if total_steps > 0 and warmup_steps > 0 and warmup_steps < total_steps:
        s1 = optim.lr_scheduler.LinearLR(opt_main, start_factor=0.01, total_iters=warmup_steps)
        min_lr = base_lr * max(min_lr_ratio, 0.0)
        s2 = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=total_steps - warmup_steps, eta_min=min_lr)
        scheduler = optim.lr_scheduler.SequentialLR(opt_main, schedulers=[s1, s2], milestones=[warmup_steps])
    elif total_steps > 0 and warmup_steps == 0 and min_lr_ratio > 0.0:
        min_lr = base_lr * max(min_lr_ratio, 0.0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_main, T_max=total_steps, eta_min=min_lr)

    if opt_h is not None:
        n_h = sum(p.numel() for p in h_params)
        print(f"[debug] H params: {n_h:,}  reg_lr={reg_lr}")
    reg_cfg = cfg.get("regularizer", {})
    if reg_cfg.get("type", "none") == "ham_sigreg" and reg_cfg.get("ham_kind", "").lower() == "learnable":
        if encoder_mode.lower() != "tokens":
            raise ValueError("ham_kind=learnable requires model.encoder_mode=tokens (so z is a real h×w token grid).")
        if proj_type != "identity":
            raise ValueError("ham_kind=learnable requires model.projector_type=identity (dense MLP destroys grid semantics).")

    ckpt_dir = cfg.get("train", {}).get("ckpt_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    cfg_stem = config_stem(args.config)
    use_amp = (precision == "bf16") and (device.type == "cuda")
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        if use_amp
        else nullcontext()
    )
    imagenet_mean = torch.tensor((0.485, 0.456, 0.406), device=device, dtype=torch.float32).view(1, 3, 1, 1)
    imagenet_std = torch.tensor((0.229, 0.224, 0.225), device=device, dtype=torch.float32).view(1, 3, 1, 1)

    global_step = 0

    # ── Resume from checkpoint ──
    latest_path = os.path.join(ckpt_dir, f"{cfg_stem}_latest.pth")
    resume_path = args.resume or (os.path.exists(latest_path) and latest_path)
    start_epoch = 1
    if resume_path and os.path.exists(resume_path):
        print(f"\n{'='*60}")
        print(f"Resuming from: {resume_path}")
        print(f"{'='*60}\n")
        ckpt = torch.load(resume_path, map_location="cpu")
        encoder.load_state_dict(ckpt["encoder"])
        projector.load_state_dict(ckpt["projector"])
        if predictor is not None and ckpt.get("predictor") is not None:
            predictor.load_state_dict(ckpt["predictor"])
        if reg_module is not None and ckpt.get("regularizer") is not None:
            reg_module.load_state_dict(ckpt["regularizer"])
        if ckpt.get("optimizer") is not None:
            opt_main.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("global_step", 0) > 0:
            global_step = ckpt["global_step"]
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {ckpt.get('epoch', 0)}. Starting at epoch {start_epoch}/{num_epochs}")

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'#'*60}\n# EPOCH {epoch}/{num_epochs}\n{'#'*60}")
        epoch_idx0 = epoch - 1

        if lambda_reg_warmup_epochs > 0 and epoch_idx0 < lambda_reg_warmup_epochs:
            lambda_reg = lambda_reg_target * float(epoch_idx0 + 1) / float(lambda_reg_warmup_epochs)
        else:
            lambda_reg = lambda_reg_target

        alpha = 1.0
        beta = 1.0
        if reg_module is not None and hasattr(reg_module, "set_h_schedule"):
            if h_identity_epochs == 0 and h_base_ramp_epochs == 0:
                beta = 1.0
            elif epoch <= h_identity_epochs:
                beta = 0.0
            elif epoch <= h_identity_epochs + h_base_ramp_epochs:
                beta = (epoch - h_identity_epochs) / float(h_base_ramp_epochs)
            else:
                beta = 1.0
            beta = float(max(0.0, min(1.0, beta)))

            if h_learnable_start_epoch == 0 and h_learnable_ramp_epochs == 0:
                alpha = 1.0
            elif epoch <= h_learnable_start_epoch:
                alpha = 0.0
            elif epoch <= h_learnable_start_epoch + h_learnable_ramp_epochs:
                alpha = (epoch - h_learnable_start_epoch) / float(h_learnable_ramp_epochs)
            else:
                alpha = 1.0
            alpha = float(max(0.0, min(1.0, alpha)))

            reg_module.set_h_schedule(alpha=alpha, beta=beta)

        if reg_module is not None and hasattr(reg_module, "set_alpha"):
            if epoch_idx0 < h_alpha_start_epoch:
                alpha = 0.0
            elif h_alpha_ramp_epochs <= 0:
                alpha = h_alpha_max
            else:
                frac = float(epoch_idx0 - h_alpha_start_epoch + 1) / float(h_alpha_ramp_epochs)
                alpha = h_alpha_max * max(0.0, min(1.0, frac))
            reg_module.set_alpha(alpha)

        can_update_h_epoch = (opt_h is not None) and (epoch > freeze_h_epochs) and (alpha > 0.0)
        if h_params:
            # Optionally freeze learnable H for warm-start stability
            set_requires_grad(h_params, can_update_h_epoch)
        running_loss = 0.0
        running_pred = 0.0
        running_reg = 0.0
        running_spec = 0.0
        running_logdet = 0.0
        running_mean = 0.0
        end_time = time.time() if profile_timing else None
        for it, (views, _, _) in enumerate(loader):
            global_step += 1
            if profile_timing and end_time is not None:
                data_time = time.time() - end_time
            # views: list length V, each is [B,C,H,W]
            views = [v.to(device, non_blocking=True) for v in views]
            x_cat = torch.cat(views, dim=0)            # [V*B, C, H, W]
            # Data loader now emits uint8 tensors; normalize on device to cut worker IPC bandwidth.
            if x_cat.dtype == torch.uint8:
                x_cat = x_cat.to(dtype=torch.float32).div_(255.0)
                x_cat.sub_(imagenet_mean).div_(imagenet_std)
            timed_iter = False
            if profile_timing and device.type == "cuda":
                timed_iter = (it % max(1, profile_every)) == 0
                if timed_iter:
                    torch.cuda.synchronize()
                    t0 = time.time()
            with amp_ctx:
                z_cat = projector(encoder(x_cat))      # [V*B, K]
            B = views[0].size(0)
            V = len(views)
            K = z_cat.size(1)
            z_views = z_cat.view(V, B, K)              # [V, B, K]

            if use_hjepa:
                if V < 2:
                    raise ValueError("MV-HJEPA requires at least 2 views for q-only matching.")

                z0 = z_views[0]
                z1 = z_views[1]
                q_dim = rep_dim // 2

                loss_pred = pred_loss_fn(z0, z1)
                if bidirectional:
                    loss_pred = 0.5 * (loss_pred + pred_loss_fn(z1, z0))

                z_all = z_views.reshape(-1, K)
                q_all = z_all[:, :q_dim]
                loss_budget = budget_fn(z_all)
                loss_var = var_floor_fn(q_all)
                p_all = z_all[:, q_dim:]
                loss_logdet_q = q_logdet_floor_fn(q_all)
                loss_logdet_p = p_logdet_floor_fn(p_all)
                loss_logdet = loss_logdet_q + loss_logdet_p
                z_all_f = z_all.float()
                loss_mean = z_all_f.mean(dim=0).square().mean()

                loss = (
                    loss_pred
                    + lambda_budget * loss_budget
                    + lambda_var * loss_var
                    + lambda_logdet * loss_logdet
                    + lambda_mean * loss_mean
                )

                opt_main.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(main_params, grad_clip)
                opt_main.step()
                if scheduler is not None:
                    scheduler.step()

                if log_every > 0 and (global_step % log_every == 0):
                    with torch.no_grad():
                        q_all_f = q_all.float()
                        p_all_f = z_all[:, q_dim:].float()
                        z_mean2 = z_all_f.mean(dim=0).square().mean()
                        q2 = q_all_f.square().sum(dim=-1).mean() / float(q_dim)
                        p2 = p_all_f.square().sum(dim=-1).mean() / float(q_dim)
                        q_mean = q_all_f.mean(dim=0, keepdim=True)
                        q_std = (q_all_f - q_mean).std(dim=0, unbiased=False)
                        q_std_min = q_std.min()
                        q_logdet_per_dim = q_logdet_floor_fn.last_logdet_per_dim
                        p_logdet_per_dim = p_logdet_floor_fn.last_logdet_per_dim
                        q_pr_norm = q_logdet_floor_fn.last_pr_norm
                        p_pr_norm = p_logdet_floor_fn.last_pr_norm
                        q_eigmax_frac = q_logdet_floor_fn.last_eigmax_frac
                        p_eigmax_frac = p_logdet_floor_fn.last_eigmax_frac
                        V_var = None
                        if hasattr(predictor.H, "potential"):
                            V = predictor.H.potential(q_all_f)
                            V_var = V.var(unbiased=False)
                        print(
                            f"[MV] pred={loss_pred.item():.4f} "
                            f"budget={loss_budget.item():.4f} var={loss_var.item():.4f} "
                            f"logdet={loss_logdet.item():.4f} q2={q2.item():.3e} p2={p2.item():.3e} "
                            f"q_std_min={q_std_min.item():.3e} "
                            f"q_pr={q_pr_norm.item():.3e} p_pr={p_pr_norm.item():.3e} "
                            f"q_logdet={q_logdet_per_dim.item():.3e} p_logdet={p_logdet_per_dim.item():.3e} "
                            f"q_eigmax={q_eigmax_frac.item():.3e} p_eigmax={p_eigmax_frac.item():.3e}"
                            + f" z_mean2={z_mean2.item():.3e}"
                            + (f" V_var={V_var.item():.3e}" if V_var is not None else "")
                        )

                running_loss += loss.item()
                running_pred += loss_pred.item()
                running_reg += loss_budget.item()
                running_spec += loss_var.item()
                running_logdet += loss_logdet.item()
                running_mean += loss_mean.item()
            else:
                loss_pred = lejepa_prediction_loss(z_views, num_global_views=num_global_views)

                can_update_h = can_update_h_epoch
                do_h_update = can_update_h and (h_update_interval > 0) and (global_step % h_update_interval == 0)

                # Gate H gradients without a second reg_module call.
                if h_params:
                    set_requires_grad(h_params, do_h_update)

                if reg_module is not None:
                    reg_loss = reg_module(z_views)
                    spec_loss = (
                        reg_module.spectral_regularizer()
                        if hasattr(reg_module, "spectral_regularizer")
                        else torch.zeros([], device=device, dtype=loss_pred.dtype)
                    )
                else:
                    reg_loss = torch.zeros([], device=device, dtype=loss_pred.dtype)
                    spec_loss = torch.zeros([], device=device, dtype=loss_pred.dtype)

                if not (0.0 <= lambda_reg <= 1.0):
                    raise ValueError(f"lambda_reg should be in [0,1], got {lambda_reg}")

                loss = loss_pred + lambda_reg * reg_loss + spectral_weight * spec_loss

                opt_main.zero_grad(set_to_none=True)
                if opt_h is not None:
                    opt_h.zero_grad(set_to_none=True)
                loss.backward(retain_graph=do_h_update)

                # Main update always.
                opt_main.step()
                if scheduler is not None:
                    scheduler.step()

                # Minimax H update: maximize reg_loss, minimize spec_loss.
                if do_h_update and (opt_h is not None) and h_params:
                    for p in h_params:
                        p.grad = None

                    g_reg = torch.autograd.grad(reg_loss, h_params, retain_graph=True, allow_unused=True)
                    if spec_loss.requires_grad:
                        g_spec = torch.autograd.grad(spec_loss, h_params, retain_graph=False, allow_unused=True)
                    else:
                        g_spec = [None] * len(h_params)

                    sign = -1.0 if h_update_mode == "max" else 1.0
                    for p, gr, gs in zip(h_params, g_reg, g_spec):
                        if gr is None:
                            gr = torch.zeros_like(p)
                        if gs is None:
                            gs = torch.zeros_like(p)
                        p.grad = sign * (lambda_reg * h_adv_mult) * gr + spectral_weight * gs

                    if h_grad_scale != 1.0:
                        for p in h_params:
                            if p.grad is not None:
                                p.grad.mul_(h_grad_scale)
                    if h_grad_clip > 0.0:
                        torch.nn.utils.clip_grad_norm_(h_params, h_grad_clip)
                    opt_h.step()

                if log_every > 0 and (global_step % log_every == 0):
                    msg = f"[M] pred={loss_pred.item():.4f} reg={reg_loss.item():.4f} spec={spec_loss.item():.3e}"
                    if do_h_update and h_params:
                        h_grad = max_abs_grad(h_params)
                        msg += f" | [H] max|grad|={h_grad:.3e}"
                    print(msg)

                running_loss += loss.item()
                running_pred += loss_pred.item()
                running_reg += reg_loss.item()
                running_spec += spec_loss.item()

            if profile_timing and device.type == "cuda":
                if timed_iter:
                    torch.cuda.synchronize()
                    step_time = time.time() - t0
                    if is_main:
                        ratio = data_time / max(step_time, 1e-6)
                        print(f"[time] data_time={data_time:.3f}s  step_time={step_time:.3f}s  ratio={ratio:.2f}")
                end_time = time.time()

        n_batches = len(loader)
        if use_hjepa:
            print(
                f"Epoch {epoch:03d}: total={running_loss / n_batches:.4f} "
                f"pred={running_pred / n_batches:.4f} "
                f"budget={running_reg / n_batches:.4f} "
                f"var={running_spec / n_batches:.4f} "
                f"logdet={running_logdet / n_batches:.4f} "
                f"mean={running_mean / n_batches:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d}: total={running_loss / n_batches:.4f} "
                f"pred={running_pred / n_batches:.4f} reg={running_reg / n_batches:.4f} "
                f"spec={running_spec / n_batches:.3e}"
            )
            if hasattr(reg_module, "learnable_h") and hasattr(reg_module.learnable_h, "health_stats"):
                stats = reg_module.learnable_h.health_stats(device=device)
                spec_snapshot = reg_module.spectral_regularizer().item()
                print(
                    f"    H lam_min={stats['lam_min'].item():.3e} "
                    f"lam_max={stats['lam_max'].item():.3e} "
                    f"lam_cond={stats['lam_cond'].item():.3e} "
                    f"g0={stats['g0'].item():.3e} "
                    f"g_last={stats['g_last'].item():.3e} "
                    f"loglam_min={stats['loglam_min'].item():.3e} "
                    f"loglam_max={stats['loglam_max'].item():.3e} "
                    f"spec={spec_snapshot:.3e}"
                )

        # ── Save latest checkpoint after EVERY epoch (for resume) ──
        if is_main:
            t_ckpt = time.time()
            save_checkpoint(
                latest_path,
                cfg=cfg,
                epoch=epoch,
                encoder=encoder,
                projector=projector,
                reg_module=reg_module,
                predictor=predictor,
                optimizer=opt_main,
                scheduler=scheduler,
                global_step=global_step,
            )
            print(f"[ckpt] latest saved ({time.time() - t_ckpt:.1f}s) -> {latest_path}")

            if epoch % ckpt_every == 0:
                epoch_path = os.path.join(ckpt_dir, f"{cfg_stem}_epoch_{epoch:03d}.pth")
                save_checkpoint(
                    epoch_path,
                    cfg=cfg,
                    epoch=epoch,
                    encoder=encoder,
                    projector=projector,
                    reg_module=reg_module,
                    predictor=predictor,
                    optimizer=opt_main,
                    scheduler=scheduler,
                    global_step=global_step,
                )
                print(f"[ckpt] epoch snapshot -> {epoch_path}")

    # ── Final checkpoint ──
    ckpt_name = f"{cfg_stem}.pth"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if is_main:
        t_ckpt = time.time()
        save_checkpoint(
            ckpt_path,
            cfg=cfg,
            epoch=num_epochs,
            encoder=encoder,
            projector=projector,
            reg_module=reg_module,
            predictor=predictor,
            optimizer=opt_main,
            scheduler=scheduler,
            global_step=global_step,
        )
        print(f"[ckpt] final saved ({time.time() - t_ckpt:.1f}s) -> {ckpt_path}")


if __name__ == "__main__":
    main()

"""PointCloud — SSL pretraining entrypoint (view-invariant 3D shape SSL).

Research question: can a two-view SSL objective learn a VIEW-INVARIANT shape
representation on an unordered/irregular modality (point clouds), and how does the
linear-probe accuracy degrade as we demand more rotation invariance (none -> z ->
SO(3))?

Point clouds have no temporal frames, so the objective is a two-view VICReg (the
image-JEPA / audio / EEG recipe), NOT a predictive JEPA. Two independent augmented
samplings + rotations of the same object are the two views.

The DATA + TRAINING LOOP are provided. The two modelling pieces you implement are
marked `# TODO` below — that is the whole point of the track:
  1. the PointNet encoder over [B, 3, N]
  2. the two-view VICReg objective

Run:  python -m examples.pointcloud.main --fname examples/pointcloud/cfgs/train.yaml
"""
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, make_loader

# Reuse the eb_jepa core — DO NOT reimplement these:
#   eb_jepa.architectures: Projector (MLP from a '256-512-128'-style spec string)
#   eb_jepa.losses:        VICRegLoss (invariance + variance + covariance)
from eb_jepa.architectures import Projector
from eb_jepa.losses import VICRegLoss


# --------------------------------------------------------------------------- #
# 1) ENCODER  — # TODO
# --------------------------------------------------------------------------- #
class PointNetEncoder(nn.Module):
    """PointNet (Qi et al. 2017) without T-Net.

    A shared per-point MLP of 1x1 Conv1d layers (in_channels -> 64 -> 64 -> 128 ->
    out_dim, each Conv1d + BatchNorm1d + ReLU) applied identically to every point,
    followed by a symmetric max-pool over the N points. The max-pool is the
    permutation-invariant aggregator (point order is meaningless); rotation
    invariance is NOT built in — it is learned from the two augmented views.
    """

    def __init__(self, in_channels=3, out_dim=1024, hidden=(64, 64, 128)):
        super().__init__()
        dims = [in_channels, *hidden, out_dim]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [
                nn.Conv1d(d_in, d_out, kernel_size=1, bias=False),  # 1x1 == shared per-point MLP
                nn.BatchNorm1d(d_out),
                nn.ReLU(inplace=True),
            ]
        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim

    def represent(self, x):
        """[B, 3, N] -> [B, out_dim]: per-point features then symmetric max-pool."""
        h = self.mlp(x)                 # [B, out_dim, N]
        return h.max(dim=2).values      # [B, out_dim]  (permutation-invariant)

    def forward(self, x):
        return self.represent(x)


def build_encoder(cfg):
    """Return a PointNet encoder mapping [B, 3, N] -> [B, out_dim], exposing
    `.represent(x)` (the frozen-feature API eval.py calls) and `.out_dim`."""
    return PointNetEncoder(
        in_channels=getattr(cfg, "in_channels", 3),
        out_dim=cfg.out_dim,
    )


# --------------------------------------------------------------------------- #
# 2) SSL OBJECTIVE  — # TODO
# --------------------------------------------------------------------------- #
class TwoViewVICReg(nn.Module):
    """Two-view VICReg head: encoder.represent -> Projector -> VICRegLoss.

    The invariance (MSE) term pulls the two augmented views of the same object
    together (-> view-invariant features); the variance + covariance terms are the
    anti-collapse ingredient. VICReg is computed on the PROJECTION, not directly on
    the representation (per the EB-JEPA recipe)."""

    def __init__(self, encoder, proj_spec, std_coeff, cov_coeff):
        super().__init__()
        self.encoder = encoder
        self.projector = Projector(proj_spec)
        self.vicreg = VICRegLoss(std_coeff, cov_coeff)

    def compute_loss(self, batch):
        v1, v2, _label = batch                      # [B,3,N], [B,3,N], [B] (label unused for SSL)
        z1 = self.projector(self.encoder.represent(v1))
        z2 = self.projector(self.encoder.represent(v2))
        out = self.vicreg(z1, z2)                   # dict: loss, invariance_loss, var_loss, cov_loss
        logs = {k: float(v.detach()) for k, v in out.items()}
        return out["loss"], logs


def build_ssl(encoder, cfg):
    """Return the two-view VICReg objective exposing `compute_loss(batch)`."""
    proj_spec = getattr(cfg, "proj_spec", f"{cfg.out_dim}-2048-2048")
    return TwoViewVICReg(encoder, proj_spec, cfg.std_coeff, cfg.cov_coeff)


# --------------------------------------------------------------------------- #
# TRAINING LOOP  — provided
# --------------------------------------------------------------------------- #
def run(fname="examples/pointcloud/cfgs/train.yaml", cfg=None, folder=None, **overrides):
    if cfg is None:
        cfg = OmegaConf.load(fname)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.meta.seed)

    dcfg = PointCloudConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.split = "train"
    dcfg.mode = "ssl"
    loader = make_loader(dcfg)

    encoder = build_encoder(cfg.model).to(device)
    ssl = build_ssl(encoder, cfg.model).to(device)
    opt = torch.optim.AdamW(ssl.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    ckpt_dir = folder or cfg.meta.ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    history = []
    for epoch in range(cfg.optim.epochs):
        ssl.train()
        for batch in loader:
            batch = batch.to(device) if torch.is_tensor(batch) else [b.to(device) for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = ssl.compute_loss(batch)
            loss.backward(); opt.step()
        print(f"[pointcloud:{cfg.data.rotate}] epoch {epoch} loss={loss.item():.4f} {logs}", flush=True)
        history.append({"epoch": epoch, **logs})
        torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                    "projector": ssl.projector.state_dict(),  # to probe the projection too
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(ckpt_dir, "latest.pth.tar"))

    # presentation artifacts: per-epoch CSV + training-dynamics curve
    keys = ["epoch", "loss", "invariance_loss", "var_loss", "cov_loss"]
    with open(os.path.join(ckpt_dir, "history.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for h in history:
            f.write(",".join(f"{h.get(k, '')}" for k in keys) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [h["epoch"] for h in history]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k in ["loss", "invariance_loss", "cov_loss", "var_loss"]:
            ax.plot(ep, [h[k] for h in history], label=k, marker="." if len(ep) < 40 else None)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_yscale("log")
        ax.set_title(f"Training dynamics — rotate={cfg.data.rotate} ep{cfg.optim.epochs} "
                     f"seed{cfg.meta.seed} (std={cfg.model.std_coeff},cov={cfg.model.cov_coeff})")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(ckpt_dir, "training_curve.png"), dpi=110, bbox_inches="tight")
        print(f"[pointcloud] curve saved -> {ckpt_dir}/training_curve.png", flush=True)
    except Exception as e:
        print("[pointcloud] curve skipped:", e, flush=True)
    print(f"[pointcloud] done -> {ckpt_dir}/latest.pth.tar")


if __name__ == "__main__":
    argv = sys.argv[1:]
    fname = argv[argv.index("--fname") + 1] if "--fname" in argv \
        else "examples/pointcloud/cfgs/train.yaml"
    # dotted overrides, e.g. optim.epochs=1 data.batch_size=32 (field-guide convention)
    overrides = dict(tok.split("=", 1) for tok in argv if "=" in tok and not tok.startswith("--"))
    run(fname=fname, **overrides)

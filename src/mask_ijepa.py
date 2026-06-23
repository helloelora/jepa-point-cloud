"""Masked I-JEPA for point clouds (à la I-JEPA / the neighbouring team's design).

A CONTEXT encoder sees a MASKED (partial) cloud -> a PREDICTOR maps it to a predicted
latent; an EMA TARGET encoder sees the FULL cloud -> target latent. We align the
predicted latent to the (stop-grad) EMA target with a cosine predictive loss; a light
variance hinge prevents collapse. The encoder must produce a useful representation from
a partial view -> "recognize a masked shape".

Reuses build_encoder (PointNet) unchanged. Anti-collapse = EMA+predictor asymmetry
(BYOL/I-JEPA) + a variance hinge (NOT the full VICReg covariance term, which fought the
EMA dynamics and destabilised training).

Occlusion strategies (MASK): random | crop (half-plane) | part (extremity blob) | mix
Usage:
  python -m examples.pointcloud.mask_ijepa ROT=so3 MASK=mix RATIO=0.5 EPOCHS=100 EMA=0.996 OUT=<dir>
"""
import csv
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, PointCloudDataset, _rand_rot
from examples.pointcloud.main import build_encoder


def kv():
    return dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)


def occlude(pc, mode, ratio, rng):
    """pc [N,3] -> boolean keep-mask [N]. Drops ~ratio of points by one of 3 strategies."""
    n = pc.shape[0]; k = int(ratio * n); m = np.ones(n, bool)
    if mode == "mix":
        mode = rng.choice(["random", "crop", "part"])
    if mode == "random":                                   # scattered dropout
        m[rng.choice(n, k, replace=False)] = False
    elif mode == "crop":                                   # remove a coherent half (plane cut)
        d = rng.normal(size=3); d /= (np.linalg.norm(d) + 1e-9)
        proj = pc @ d; m = proj < np.quantile(proj, 1 - ratio)
    elif mode == "part":                                   # remove a protruding part (extremity blob)
        ext = pc[np.argmax(np.linalg.norm(pc - pc.mean(0), axis=1))]
        dd = np.linalg.norm(pc - ext, axis=1); m[np.argsort(dd)[:k]] = False
    else:
        raise ValueError(mode)
    if m.sum() < 16:                                       # never mask everything
        m[rng.choice(n, 16, replace=False)] = True
    return m


class MaskedPairPCD(PointCloudDataset):
    """(context = masked partial cloud, target = full cloud) in the SAME pose, each [3, n_points]."""
    def __init__(self, cfg, mode, ratio):
        super().__init__(cfg); self.mode = mode; self.ratio = float(ratio)

    def __getitem__(self, i):
        rng = np.random.default_rng(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        pc, y = self.data[i], int(self.label[i]); c = self.cfg
        idx = rng.choice(pc.shape[0], c.n_points, replace=c.n_points > pc.shape[0])
        base = pc[idx] @ _rand_rot(rng, c.rotate).T
        base = base * rng.uniform(c.scale_lo, c.scale_hi)
        base = base + rng.normal(0, c.jitter, size=base.shape).astype(np.float32)
        center = base.mean(0, keepdims=True)
        scale = np.max(np.linalg.norm(base - center, axis=1)) + 1e-6
        target = (base - center) / scale                            # full view [N,3]
        keep = occlude(base, self.mode, self.ratio, rng)
        ctx = target[keep]                                          # partial, same frame
        ridx = rng.choice(ctx.shape[0], c.n_points, replace=True)   # -> fixed size
        context = ctx[ridx]
        return (torch.from_numpy(context.T.astype(np.float32)),
                torch.from_numpy(target.T.astype(np.float32)), y)


class MaskedIJEPA(nn.Module):
    def __init__(self, encoder, cfg, ema=0.996, var_coeff=1.0):
        super().__init__()
        self.ctx = encoder; d = encoder.out_dim; self.ema = float(ema); self.var_coeff = float(var_coeff)
        self.predictor = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(True), nn.Linear(d, d))
        self.tgt = copy.deepcopy(encoder)                           # EMA target encoder
        for p in self.tgt.parameters():
            p.requires_grad_(False)

    def compute_loss(self, batch):
        ctx, tgt, _ = batch
        p = self.predictor(self.ctx.represent(ctx))                 # [B,D] predicted (grad)
        with torch.no_grad():
            t = self.tgt.represent(tgt)                             # [B,D] target (stop-grad)
        pn = F.normalize(p, dim=1); tn = F.normalize(t, dim=1)
        inv = (2 - 2 * (pn * tn).sum(1)).mean()                     # cosine predictive loss
        std = torch.sqrt(p.var(0) + 1e-4)
        var = torch.relu(1.0 - std).mean()                          # variance hinge (anti-collapse)
        loss = inv + self.var_coeff * var
        return loss, {"loss": float(loss.detach()), "inv": float(inv.detach()), "var": float(var.detach())}

    @torch.no_grad()
    def update_target(self):
        for q, k in zip(self.ctx.parameters(), self.tgt.parameters()):
            k.data.mul_(self.ema).add_(q.data, alpha=1 - self.ema)
        for q, k in zip(self.ctx.buffers(), self.tgt.buffers()):
            k.data.copy_(q.data)


def main():
    a = kv()
    rot = a.get("ROT", "so3"); mode = a.get("MASK", "mix"); ratio = float(a.get("RATIO", 0.5))
    epochs = int(a.get("EPOCHS", 100)); ema = float(a.get("EMA", 0.996)); vc = float(a.get("VAR", 1.0))
    out = a["OUT"]; os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load("examples/pointcloud/cfgs/train.yaml")
    cfg.data.rotate = rot; cfg.optim.epochs = epochs
    if a.get("DATA"):
        cfg.data.data_root = a["DATA"]          # override for the cluster path
    cfg.meta.mask_mode = mode; cfg.meta.mask_ratio = ratio; cfg.meta.ema = ema
    torch.manual_seed(cfg.meta.seed)

    dcfg = PointCloudConfig(**OmegaConf.to_container(cfg.data, resolve=True))
    dcfg.split, dcfg.mode = "train", "ssl"
    ds = MaskedPairPCD(dcfg, mode, ratio)
    loader = torch.utils.data.DataLoader(ds, batch_size=dcfg.batch_size, shuffle=True,
                                         num_workers=dcfg.num_workers, pin_memory=True,
                                         drop_last=True, persistent_workers=dcfg.num_workers > 0)

    enc = build_encoder(cfg.model).to(device)
    model = MaskedIJEPA(enc, cfg.model, ema=ema, var_coeff=vc).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    hist = open(os.path.join(out, "history.csv"), "w", newline="")
    wr = csv.writer(hist); wr.writerow(["epoch", "loss", "inv", "var"])
    print(f"[mask-ijepa] mode={mode} ratio={ratio} rot={rot} epochs={epochs} ema={ema} var={vc} "
          f"| device={device} N={len(ds)}", flush=True)
    for e in range(epochs):
        model.train(); model.tgt.eval()                 # target uses running BN stats (EMA-copied)
        last = None
        for batch in loader:
            batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            opt.zero_grad(set_to_none=True)
            loss, logs = model.compute_loss(batch)
            loss.backward(); opt.step(); model.update_target(); last = logs
        wr.writerow([e, last["loss"], last["inv"], last["var"]]); hist.flush()
        print(f"[mask-ijepa:{mode} r={ratio} {rot}] epoch {e} {last}", flush=True)
        torch.save({"epoch": e, "encoder": enc.state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True)},
                   os.path.join(out, "latest.pth.tar"))
    hist.close()
    print(f"[mask-ijepa] done -> {out}/latest.pth.tar", flush=True)


if __name__ == "__main__":
    main()

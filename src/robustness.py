"""Rotation-robustness probe matrix (the view-invariance result).

For each pretrained encoder (rotate=none|z|so3, + optional random floor) we train
ONE linear probe on ALIGNED train features, then score on aligned and SO(3)-rotated
test features. A rotation-invariant encoder (so3 pretraining) keeps its accuracy
under rotated test; a non-invariant one (none) collapses. That gap is the result.

Rotations are applied per-sample to the clean unit-sphere point cloud (rotation about
the origin preserves the centering+scale normalization), then encoded.

Run:
  python -m examples.pointcloud.robustness \
      none=<ckpt> z=<ckpt> so3=<ckpt> --random
"""
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, PointCloudDataset, _rand_rot
from examples.pointcloud.main import build_encoder


@torch.no_grad()
def features(encoder, split, dcfg, device, rotate, seed):
    """Clean canonical points, optionally rotated per-sample, then encoded -> [N,D],[N]."""
    cfg = PointCloudConfig(**{**dcfg, "split": split, "mode": "supervised"})
    ds = PointCloudDataset(cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=8)
    rng = np.random.default_rng(seed)
    X, y = [], []
    for xb, yb in loader:                          # xb: [B,3,N] canonical (unit-sphere)
        xb = xb.to(device)
        if rotate != "none":
            R = np.stack([_rand_rot(rng, rotate) for _ in range(xb.shape[0])])   # [B,3,3]
            xb = torch.bmm(torch.from_numpy(R).to(device, xb.dtype), xb)         # rotate coords
        X.append(encoder.represent(xb).cpu().numpy())
        y.append(np.asarray(yb))
    return np.concatenate(X), np.concatenate(y)


def load_encoder(ckpt, device):
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    enc = build_encoder(cfg.model).to(device)
    enc.load_state_dict(state["encoder"]); enc.eval()
    return enc, OmegaConf.to_container(cfg.data, resolve=True)


def main():
    args = sys.argv[1:]
    encs = dict(a.split("=", 1) for a in args if "=" in a)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = {}
    for name, ckpt in encs.items():
        rows[name] = load_encoder(ckpt, device)
    if "--random" in args:
        cfg = OmegaConf.load("examples/pointcloud/cfgs/train.yaml")
        torch.manual_seed(cfg.meta.seed)
        rows["random"] = (build_encoder(cfg.model).to(device).eval(),
                          OmegaConf.to_container(cfg.data, resolve=True))

    print(f"{'encoder':>8} | {'test=aligned':>13} | {'test=SO3':>9} | {'drop':>7} | chance")
    print("-" * 56)
    for name, (enc, dcfg) in rows.items():
        Xtr, ytr = features(enc, "train", dcfg, device, "none", seed=0)    # probe on aligned train
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
        accs = {}
        for cond in ("none", "so3"):
            Xte, yte = features(enc, "test", dcfg, device, cond, seed=123)
            accs[cond] = float((clf.predict(scaler.transform(Xte)) == yte).mean())
        drop = accs["none"] - accs["so3"]
        print(f"{name:>8} | {100*accs['none']:>12.2f}% | {100*accs['so3']:>8.2f}% | "
              f"{100*drop:>6.1f} | 2.5%")


if __name__ == "__main__":
    main()

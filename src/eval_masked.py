"""Masked-recognition eval: can the frozen encoder recognise a PARTIALLY-OBSERVED shape?

Fit the linear probe on FULL (clean) train features, then test on MASKED test clouds
at several occlusion ratios x strategies (random / crop / part). Accuracy vs occlusion
is the "recognize a masked shape" curve — the headline for the masked-JEPA story.

Run:  python -m examples.pointcloud.eval_masked --ckpt <.../latest.pth.tar> [--out <dir>]
"""
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, PointCloudDataset
from examples.pointcloud.main import build_encoder
from examples.pointcloud.mask_ijepa import occlude          # reuse the 3 occlusion strategies


@torch.no_grad()
def feats(encoder, split, dcfg, device, mask=None, ratio=0.0, seed=0):
    """Frozen encoder -> [N,D] features. Each shape: deterministic clean subsample +
    normalize; if mask given, occlude then resample the partial cloud back to n_points."""
    cfg = PointCloudConfig(**{**dcfg, "split": split, "mode": "supervised"})
    ds = PointCloudDataset(cfg)
    N = cfg.n_points; rng = np.random.default_rng(seed)
    X, y, buf = [], [], []
    for i in range(len(ds.data)):
        pc = ds.data[i]
        idx = np.linspace(0, pc.shape[0] - 1, N).astype(int)        # deterministic clean view
        p = pc[idx].astype(np.float32)
        p = p - p.mean(0); p = p / (np.max(np.linalg.norm(p, axis=1)) + 1e-6)
        if mask is not None and ratio > 0:
            keep = occlude(p, mask, ratio, rng)
            pk = p[keep]; p = pk[rng.choice(pk.shape[0], N, replace=True)]
        buf.append(p.T); y.append(int(ds.label[i]))
        if len(buf) == 256:
            X.append(encoder.represent(torch.from_numpy(np.stack(buf)).to(device)).cpu().numpy()); buf = []
    if buf:
        X.append(encoder.represent(torch.from_numpy(np.stack(buf)).to(device)).cpu().numpy())
    return np.concatenate(X), np.array(y)


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.dirname(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"]); dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    if "--data" in sys.argv:
        dcfg["data_root"] = sys.argv[sys.argv.index("--data") + 1]   # override stale ckpt path
    enc = build_encoder(cfg.model).to(device); enc.load_state_dict(state["encoder"]); enc.eval()

    Xtr, ytr = feats(enc, "train", dcfg, device)                    # probe fit on FULL clean train
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)

    res = {}
    # full (no mask) baseline
    Xte, yte = feats(enc, "test", dcfg, device)
    res["full"] = round(100 * float((clf.predict(scaler.transform(Xte)) == yte).mean()), 2)
    print(f"[masked-eval] full test acc = {res['full']}%", flush=True)
    for mask in ["random", "crop", "part"]:
        for ratio in [0.3, 0.5, 0.7]:
            Xte, yte = feats(enc, "test", dcfg, device, mask=mask, ratio=ratio)
            acc = round(100 * float((clf.predict(scaler.transform(Xte)) == yte).mean()), 2)
            res[f"{mask}_{int(ratio*100)}"] = acc
            print(f"[masked-eval] mask={mask:7s} ratio={ratio} -> acc {acc}%", flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump({"chance": 2.5, "results": res}, open(os.path.join(out, "masked_eval.json"), "w"), indent=2)
    print("[masked-eval] saved", os.path.join(out, "masked_eval.json"), flush=True)
    print("[masked-eval] DONE", flush=True)


if __name__ == "__main__":
    main()

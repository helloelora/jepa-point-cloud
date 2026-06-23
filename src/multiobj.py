"""Multi-object compositional generalization test (jury suggestion).

The SSL encoder was trained on ISOLATED objects. Here we test whether its frozen
features generalize to SCENES of k objects it never saw. We synthesize scenes by
merging k distinct ModelNet40 objects (translated apart -> unit-sphere normalized
-> subsampled to N points). A multi-label linear probe (One-vs-Rest logistic),
trained on SINGLE objects only, then scores which classes are present.

Metric: mean Average Precision (mAP) vs number of objects k in {1,2,3}, comparing
the SSL encoder against a RANDOM-init floor. Degradation curve = the result.

Usage:
  python -m examples.pointcloud.multiobj SO3=<ckpt> [NONE=<ckpt>] OUT=<dir>
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from examples.pointcloud.main import build_encoder

NPTS = 1024
NCLASS = 40


def kv():
    return dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)


def load_split(root, split):
    data, label = [], []
    for p in sorted(glob.glob(os.path.join(root, f"ply_data_{split}*.h5"))):
        import h5py
        with h5py.File(p, "r") as f:
            data.append(f["data"][:].astype(np.float32))
            label.append(f["label"][:].reshape(-1).astype(int))
    return np.concatenate(data), np.concatenate(label)


def normalize(pc):
    pc = pc - pc.mean(0, keepdims=True)
    return pc / (np.linalg.norm(pc, axis=1).max() + 1e-6)


def make_scene(data, label, k, rng, by_class):
    """Merge k objects of DISTINCT classes, translated apart -> [3,NPTS], multi-hot[40]."""
    classes = rng.choice(NCLASS, size=k, replace=False)
    offs = np.array([[np.cos(t), 0, np.sin(t)] for t in
                     np.linspace(0, 2 * np.pi, k, endpoint=False)], dtype=np.float32) * 1.6
    pts, multi = [], np.zeros(NCLASS, np.float32)
    for c, off in zip(classes, offs):
        i = rng.choice(by_class[c])
        p = data[i]
        sub = rng.choice(p.shape[0], NPTS, replace=False)
        pts.append(normalize(p[sub]) + off)        # per-object normalize, then place
        multi[c] = 1.0
    allp = np.concatenate(pts, 0)
    sub = rng.choice(allp.shape[0], NPTS, replace=False)
    return normalize(allp[sub]).T.astype(np.float32), multi      # [3,NPTS], [40]


def class_names(root):
    p = os.path.join(root, "shape_names.txt")
    return [l.strip() for l in open(p)] if os.path.exists(p) else [str(i) for i in range(NCLASS)]


def save_scene_examples(data, label, by_class, root, out):
    """Render a grid of example INPUT scenes (k=1,2,3) -> scenes_examples.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = class_names(root)
        fig = plt.figure(figsize=(12, 10))
        rng = np.random.default_rng(7)
        row = 0
        for k in (1, 2, 3):
            for col in range(3):
                pc, multi = make_scene(data, label, k, rng, by_class)   # [3,N], [40]
                p = pc.T
                ax = fig.add_subplot(3, 3, row * 3 + col + 1, projection="3d")
                ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=2, c=p[:, 1], cmap="viridis")
                ax.set_axis_off()
                present = [names[c] for c in np.where(multi > 0)[0]]
                ax.set_title(f"k={k}: " + " + ".join(present), fontsize=9)
            row += 1
        fig.suptitle("Synthetic multi-object INPUT scenes (merged ModelNet40 objects)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "scenes_examples.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("saved scenes_examples.png", flush=True)
    except Exception as e:
        print("scene examples skipped:", e, flush=True)


@torch.no_grad()
def encode(enc, X, device, bs=256):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        out.append(enc.represent(xb).cpu().numpy())
    return np.concatenate(out)


def load_enc(ckpt, device):
    s = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(s["cfg"])
    enc = build_encoder(cfg.model).to(device); enc.load_state_dict(s["encoder"]); enc.eval()
    return enc, OmegaConf.to_container(cfg.data, resolve=True)


def evaluate(enc, dcfg, device, tag):
    root = dcfg["data_root"]
    dtr, ltr = load_split(root, "train"); dte, lte = load_split(root, "test")
    by_tr = {c: np.where(ltr == c)[0] for c in range(NCLASS)}
    by_te = {c: np.where(lte == c)[0] for c in range(NCLASS)}

    # train multi-label probe on SINGLE objects (k=1, in-distribution)
    rng = np.random.default_rng(0)
    Xtr = np.stack([normalize(dtr[i][rng.choice(dtr.shape[1], NPTS, replace=False)]).T
                    for i in range(len(dtr))]).astype(np.float32)
    Ytr = np.eye(NCLASS, dtype=np.float32)[ltr]
    Ftr = StandardScaler().fit(encode(enc, Xtr, device))
    Ztr = Ftr.transform(encode(enc, Xtr, device))
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000)).fit(Ztr, Ytr)

    res = {}
    for k in (1, 2, 3):
        rng = np.random.default_rng(100 + k)
        scenes = [make_scene(dte, lte, k, rng, by_te) for _ in range(2000)]
        Xs = np.stack([s[0] for s in scenes]); Ys = np.stack([s[1] for s in scenes])
        Zs = Ftr.transform(encode(enc, Xs, device))
        scores = clf.decision_function(Zs)
        mAP = float(average_precision_score(Ys, scores, average="macro"))
        res[k] = round(100 * mAP, 2)
        print(f"[{tag}] k={k}  mAP={100*mAP:.2f}%", flush=True)
    return res


def main():
    a = kv(); out = a["OUT"]; os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_res = {}

    enc_so3, dcfg = load_enc(a["SO3"], device)
    # render example input scenes (uses test split objects)
    dte0, lte0 = load_split(dcfg["data_root"], "test")
    by_te0 = {c: np.where(lte0 == c)[0] for c in range(NCLASS)}
    save_scene_examples(dte0, lte0, by_te0, dcfg["data_root"], out)
    all_res["so3"] = evaluate(enc_so3, dcfg, device, "so3")
    if a.get("NONE"):
        enc_none, _ = load_enc(a["NONE"], device)
        all_res["none"] = evaluate(enc_none, dcfg, device, "none")

    # random-init floor
    cfg = OmegaConf.load("examples/pointcloud/cfgs/train.yaml")
    torch.manual_seed(cfg.meta.seed)
    enc_rand = build_encoder(cfg.model).to(device).eval()
    all_res["random"] = evaluate(enc_rand, OmegaConf.to_container(cfg.data, resolve=True),
                                 device, "random")

    json.dump(all_res, open(os.path.join(out, "multiobj.json"), "w"), indent=2)
    print("=== MULTIOBJ RESULT (mAP %) ===", flush=True)
    print(f"{'encoder':>8} | {'k=1':>6} | {'k=2':>6} | {'k=3':>6}")
    for name, r in all_res.items():
        print(f"{name:>8} | {r[1]:>6} | {r[2]:>6} | {r[3]:>6}")
    print("=== MULTIOBJ_DONE ===", flush=True)


if __name__ == "__main__":
    main()

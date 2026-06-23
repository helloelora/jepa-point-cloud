"""Full rotation analysis for one checkpoint + presentation artifacts.

Builds the probe-train x test rotation matrix {aligned, SO3}^2 for BOTH the
backbone representation and the VICReg PROJECTION (tests the 'projector absorbs
rotation invariance' hypothesis), and saves, next to the checkpoint:
  matrices.png  rotation-matrix heatmaps (backbone | projection)
  pca.png       2D PCA of backbone feats colored by class
  tsne.png      2D t-SNE of backbone feats colored by class (best-effort)
  confusion.png confusion matrix of the aligned/aligned backbone probe
  results.json  all numbers

Run:  python -m examples.pointcloud.analysis --ckpt <.../latest.pth.tar>
"""
import json
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from eb_jepa.architectures import Projector
from eb_jepa.datasets.pointcloud.dataset import PointCloudConfig, PointCloudDataset, _rand_rot
from examples.pointcloud.main import build_encoder

ROTS = ("none", "so3")


@torch.no_grad()
def feats(fn, split, dcfg, device, rotate, seed):
    cfg = PointCloudConfig(**{**dcfg, "split": split, "mode": "supervised"})
    loader = torch.utils.data.DataLoader(PointCloudDataset(cfg), batch_size=256,
                                         shuffle=False, num_workers=8)
    rng = np.random.default_rng(seed)
    X, y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        if rotate != "none":
            R = np.stack([_rand_rot(rng, rotate) for _ in range(xb.shape[0])])
            xb = torch.bmm(torch.from_numpy(R).to(device, xb.dtype), xb)
        X.append(fn(xb).cpu().numpy()); y.append(np.asarray(yb))
    return np.concatenate(X), np.concatenate(y)


def rotation_matrix(fn, dcfg, device, want_confusion=False):
    """Return 2x2 accuracy matrix [probe-train{aligned,SO3}] x [test{aligned,SO3}]."""
    M = np.zeros((2, 2)); conf = None
    for i, ptr in enumerate(ROTS):
        Xtr, ytr = feats(fn, "train", dcfg, device, ptr, seed=0)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
        for j, tst in enumerate(ROTS):
            Xte, yte = feats(fn, "test", dcfg, device, tst, seed=123)
            pred = clf.predict(sc.transform(Xte))
            M[i, j] = 100 * float((pred == yte).mean())
            if want_confusion and ptr == "none" and tst == "none":
                conf = confusion_matrix(yte, pred)
    return M, conf


def heatmap(ax, M, title):
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks([0, 1], ["test aligned", "test SO3"])
    ax.set_yticks([0, 1], ["probe aligned", "probe SO3"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    color="white" if M[i, j] < 55 else "black", fontsize=12)
    ax.set_title(title, fontsize=10)
    return im


def main():
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = OmegaConf.create(state["cfg"])
    dcfg = OmegaConf.to_container(cfg.data, resolve=True)
    outdir = os.path.dirname(ckpt)
    enc = build_encoder(cfg.model).to(device); enc.load_state_dict(state["encoder"]); enc.eval()
    tag = f"rotate={cfg.data.rotate} ep{cfg.optim.epochs} seed{cfg.meta.seed}"
    print(f"=== {ckpt}  ({tag}) ===", flush=True)

    Mb, conf = rotation_matrix(lambda x: enc.represent(x), dcfg, device, want_confusion=True)
    print("BACKBONE matrix [probe x test]:\n", np.round(Mb, 2), flush=True)
    results = {"ckpt": ckpt, "tag": tag, "backbone_matrix": Mb.tolist(), "rows": ["aligned", "SO3"], "cols": ["aligned", "SO3"]}

    Mp = None
    if "projector" in state:
        proj = Projector(cfg.model.proj_spec).to(device); proj.load_state_dict(state["projector"]); proj.eval()
        Mp, _ = rotation_matrix(lambda x: proj(enc.represent(x)), dcfg, device)
        print("PROJECTION matrix [probe x test]:\n", np.round(Mp, 2), flush=True)
        results["projection_matrix"] = Mp.tolist()

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # rotation-matrix heatmaps
        n = 2 if Mp is not None else 1
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)
        im = heatmap(axes[0][0], Mb, f"BACKBONE  ({tag})")
        if Mp is not None:
            heatmap(axes[0][1], Mp, "PROJECTION")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="accuracy %")
        fig.savefig(os.path.join(outdir, "matrices.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

        # feature scatters (aligned test)
        Xte, yte = feats(lambda x: enc.represent(x), "test", dcfg, device, "none", seed=0)
        Xs = StandardScaler().fit_transform(Xte)
        Zp = PCA(n_components=2).fit_transform(Xs)
        for Z, name in [(Zp, "pca")]:
            plt.figure(figsize=(7, 6))
            plt.scatter(Z[:, 0], Z[:, 1], c=yte, s=5, cmap="tab20")
            plt.title(f"{name.upper()} backbone feats — {tag}")
            plt.savefig(os.path.join(outdir, f"{name}.png"), dpi=110, bbox_inches="tight")
            plt.close()
        try:
            from sklearn.manifold import TSNE
            Zt = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(Xs)
            plt.figure(figsize=(7, 6))
            plt.scatter(Zt[:, 0], Zt[:, 1], c=yte, s=5, cmap="tab20")
            plt.title(f"t-SNE backbone feats — {tag}")
            plt.savefig(os.path.join(outdir, "tsne.png"), dpi=110, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print("tsne skipped:", e, flush=True)

        # confusion matrix (aligned/aligned)
        if conf is not None:
            plt.figure(figsize=(7, 6))
            plt.imshow(conf, cmap="magma")
            plt.title(f"Confusion (aligned probe) — {tag}")
            plt.xlabel("predicted"); plt.ylabel("true"); plt.colorbar()
            plt.savefig(os.path.join(outdir, "confusion.png"), dpi=110, bbox_inches="tight")
            plt.close()
        print("figures saved in", outdir, flush=True)
    except Exception as e:
        print("figures skipped:", e, flush=True)


if __name__ == "__main__":
    main()

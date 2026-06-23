"""Assemble all presentation GIFs from data + precomputed arrays (no model, no GPU).

Produces, in OUT:
  spin_<class>.gif        a few objects tumbling in 3D                 (Data slide)
  augmentations.gif       one object -> the two-view SSL pipeline      (Data slide)
  embedding_evolution.gif backbone PCA organizing by class over epochs (Training slide)
  collapse.gif            projection covariance: healthy vs inv_only   (Collapse bonus)
  rotation_robustness.gif object spins; pred class none vs so3         (Result slide)

Usage:
  python -m examples.pointcloud.viz_gifs DATA_ROOT=<..> SNAP_SO3=<dir> \
     SNAP_INV=<dir> SNAP_BASE=<dir> ROBUST=<dir> OUT=<dir>
"""
import glob
import os
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def kv():
    return dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)


A = kv()
OUT = A["OUT"]; os.makedirs(OUT, exist_ok=True)
WR = PillowWriter(fps=12)


def save(anim, name):
    p = os.path.join(OUT, name)
    anim.save(p, writer=WR); plt.close("all")
    print("GIF:", p, flush=True)


def load_data(root):
    f = sorted(glob.glob(os.path.join(root, "ply_data_train*.h5")))[0]
    with h5py.File(f, "r") as h:
        return h["data"][:].astype(np.float32), h["label"][:].reshape(-1)


def load_snaps(d):
    fs = sorted(glob.glob(os.path.join(d, "epoch_*.npz")))
    return [np.load(f, allow_pickle=True) for f in fs]


# ---------- 1. spinning objects (data) ----------
def gif_spin(data, label):
    try:
        names = ["airplane", "chair", "guitar"]; picks = []
        # just take 3 distinct labels
        for i, l in enumerate(label):
            if int(l) not in [int(label[p]) for p in picks]:
                picks.append(i)
            if len(picks) == 3:
                break
        for k, i in enumerate(picks):
            pc = data[i]; sub = np.linspace(0, len(pc)-1, 800).astype(int); pc = pc[sub]
            fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111, projection="3d")

            def upd(t, pc=pc, ax=ax):
                ax.clear(); ax.set_axis_off()
                ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=3, c=pc[:, 1], cmap="viridis")
                ax.view_init(elev=20, azim=t * 6)
            save(FuncAnimation(fig, upd, frames=60, interval=80), f"spin_{k}.gif")
    except Exception as e:
        print("spin skipped:", e, flush=True)


# ---------- 2. augmentation pipeline ----------
def gif_aug(data):
    try:
        from eb_jepa.datasets.pointcloud.dataset import _rand_rot
        pc = data[30]; rng = np.random.default_rng(0)

        def view(rng):
            idx = rng.choice(pc.shape[0], 1024, replace=False); p = pc[idx]
            p = p @ _rand_rot(rng, "so3").T
            p = p * rng.uniform(0.8, 1.25) + rng.normal(0, 0.01, (1024, 3)).astype(np.float32)
            p = p - p.mean(0); return p / (np.linalg.norm(p, axis=1).max() + 1e-6)
        v1, v2 = view(np.random.default_rng(1)), view(np.random.default_rng(2))
        fig = plt.figure(figsize=(8, 4))
        a1 = fig.add_subplot(121, projection="3d"); a2 = fig.add_subplot(122, projection="3d")

        def upd(t):
            for ax, v, ttl in ((a1, v1, "view v1"), (a2, v2, "view v2")):
                ax.clear(); ax.set_axis_off(); ax.set_title(ttl)
                ax.scatter(v[:, 0], v[:, 2], v[:, 1], s=3, c=v[:, 1], cmap="plasma")
                ax.view_init(elev=20, azim=t * 6)
            fig.suptitle("Two augmented views of the SAME object  ->  pulled together by VICReg")
        save(FuncAnimation(fig, upd, frames=60, interval=80), "augmentations.gif")
    except Exception as e:
        print("aug skipped:", e, flush=True)


# ---------- 3. embedding evolution ----------
def gif_embed(snaps):
    try:
        zf = snaps[-1]["z"]; lab = snaps[-1]["labels"]
        sc = StandardScaler().fit(zf); pca = PCA(2).fit(sc.transform(zf))
        Z = [pca.transform(sc.transform(s["z"])) for s in snaps]
        allz = np.concatenate(Z); xl = (allz[:, 0].min(), allz[:, 0].max())
        yl = (allz[:, 1].min(), allz[:, 1].max())
        fig, ax = plt.subplots(figsize=(6, 5.5))

        def upd(t):
            ax.clear(); ax.set_xlim(xl); ax.set_ylim(yl)
            ax.scatter(Z[t][:, 0], Z[t][:, 1], c=lab, s=6, cmap="tab20")
            lg = snaps[t]["logs"][0]
            inv = lg.get("invariance_loss", float("nan"))
            ax.set_title(f"Backbone PCA — epoch {t}/{len(snaps)-1}   (invariance={inv:.3f})")
        save(FuncAnimation(fig, upd, frames=len(snaps), interval=150), "embedding_evolution.gif")
    except Exception as e:
        print("embed skipped:", e, flush=True)


# ---------- 4. collapse ----------
def gif_collapse(inv, base):
    try:
        n = min(len(inv), len(base))
        vi = [s["logs"][0].get("var_loss", np.nan) for s in inv]
        vb = [s["logs"][0].get("var_loss", np.nan) for s in base]
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.2),
                               gridspec_kw={"width_ratios": [1, 1, 1.1]})

        def upd(t):
            for a in ax:
                a.clear()
            for k, (s, ttl) in enumerate(((inv[t], "inv_only (std=0,cov=0) -> COLLAPSE"),
                                          (base[t], "baseline (std=25,cov=1) -> healthy"))):
                c = s["cov"]; d = c.shape[0]
                v = c[:64, :64]                       # zoom 64x64 block for readability
                ax[k].imshow(v, cmap="coolwarm", vmin=-1, vmax=1)
                ax[k].set_title(ttl, fontsize=10); ax[k].set_xticks([]); ax[k].set_yticks([])
            ax[2].plot(range(t + 1), vi[:t + 1], "r-", label="inv_only")
            ax[2].plot(range(t + 1), vb[:t + 1], "g-", label="baseline")
            ax[2].set_xlim(0, n); ax[2].set_ylim(-0.1, 2.2)
            ax[2].axhline(1.0, ls=":", c="gray"); ax[2].legend(loc="upper right")
            ax[2].set_title("var_loss (high = collapse)"); ax[2].set_xlabel("epoch")
            fig.suptitle(f"Projection covariance & variance — epoch {t}/{n-1}")
        save(FuncAnimation(fig, upd, frames=n, interval=150), "collapse.gif")
    except Exception as e:
        print("collapse skipped:", e, flush=True)


# ---------- 5. rotation robustness ----------
def gif_robust(d):
    try:
        z = np.load(os.path.join(d, "robustness_raw.npz"), allow_pickle=True)
        pts, ang = z["pts"], z["angles"]; ps, pn = z["probs_so3"], z["probs_none"]
        tl = int(z["true_label"]); names = list(z["class_names"]); cname = names[tl]
        topn = 5
        order = np.argsort(ps.mean(0) + pn.mean(0))[::-1][:topn]
        labs = [names[i] for i in order]
        fig = plt.figure(figsize=(12, 4.5))
        a0 = fig.add_subplot(131, projection="3d")
        a1 = fig.add_subplot(132); a2 = fig.add_subplot(133)

        def upd(t):
            a0.clear(); a0.set_axis_off()
            p = pts[t]; a0.scatter(p[:, 0], p[:, 2], p[:, 1], s=4, c=p[:, 1], cmap="viridis")
            a0.view_init(elev=20, azim=np.degrees(ang[t]))
            a0.set_title(f"true = {cname}\nrotation {np.degrees(ang[t]):.0f} deg")
            for ax, prob, ttl in ((a1, pn[t], "none-trained"), (a2, ps[t], "so3-trained")):
                ax.clear(); ax.barh(range(topn), prob[order],
                                    color=["crimson" if i == tl else "steelblue" for i in order])
                ax.set_yticks(range(topn), labs); ax.invert_yaxis(); ax.set_xlim(0, 1)
                ax.set_title(f"{ttl}\npred={names[int(np.argmax(prob))]}")
            fig.suptitle("Probe trained on ALIGNED objects, tested under rotation "
                         "(red = true class)")
        save(FuncAnimation(fig, upd, frames=len(ang), interval=120), "rotation_robustness.gif")
    except Exception as e:
        print("robust skipped:", e, flush=True)


def main():
    data, label = load_data(A["DATA_ROOT"])
    gif_spin(data, label)
    gif_aug(data)
    if A.get("SNAP_SO3"):
        gif_embed(load_snaps(A["SNAP_SO3"]))
    if A.get("SNAP_INV") and A.get("SNAP_BASE"):
        gif_collapse(load_snaps(A["SNAP_INV"]), load_snaps(A["SNAP_BASE"]))
    if A.get("ROBUST"):
        gif_robust(A["ROBUST"])
    print("=== GIFS_DONE ===", flush=True)


if __name__ == "__main__":
    main()

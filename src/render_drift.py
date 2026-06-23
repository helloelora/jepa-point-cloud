"""Render the rotation-invariance embedding-drift GIF from embed_drift.json.

Left: a point cloud spinning through a full turn. Middle/right: where its embedding lands
in a fixed PCA(2) of the whole ModelNet40 test set, for the trained PROJECTOR (stays glued
to its cluster) vs the BACKBONE (wanders). Live cos(emb_0, emb_t) readout. This reproduces
the neighbouring team's interactive demo as a static GIF for the README.

Run:  python render_drift.py results/embed_drift.json assets/rotation_embedding.gif
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/embed_drift.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "assets/rotation_embedding.gif"
OBJECTS = ["airplane", "chair", "guitar"]          # cycled in the GIF
VIEWS = [("projector", "Projector  ·  rotation-invariant"),
         ("backbone", "Backbone  ·  pose-sensitive")]

d = json.load(open(SRC))
angles = d["projector"]["angles"]; AX = np.array(d["projector"]["axis"])
NA = len(angles)


def rot_about(axis, a):
    x, y, z = axis; c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


frames = [(o, t) for o in OBJECTS for t in range(NA)]   # object x angle

fig = plt.figure(figsize=(13, 4.6))
ax3d = fig.add_subplot(1, 3, 1, projection="3d")
axp = fig.add_subplot(1, 3, 2)
axb = fig.add_subplot(1, 3, 3)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.04, wspace=0.18)
fig.suptitle("Spin an object, watch its embedding  ·  trained encoder, ModelNet40",
             fontsize=12, weight="bold")


def draw(i):
    obj, t = frames[i]
    for ax in (ax3d, axp, axb):
        ax.clear()
    # --- left: spinning cloud ---
    pts = np.array(d["projector"]["objects"][obj]["pts"])
    R = rot_about(AX, angles[t]); P = pts @ R.T
    ax3d.scatter(P[:, 0], P[:, 2], P[:, 1], s=3, c=P[:, 1], cmap="viridis", depthshade=True)
    ax3d.set_title(obj, fontsize=11); ax3d.set_axis_off()
    ax3d.set_xlim(-1, 1); ax3d.set_ylim(-1, 1); ax3d.set_zlim(-1, 1)
    ax3d.view_init(elev=18, azim=35)
    # --- embedding panels ---
    for ax, (key, title) in zip((axp, axb), VIEWS):
        v = d[key]; bg = np.array(v["bg"]); y = np.array(v["bg_y"])
        o = v["objects"][obj]; traj = np.array(o["traj"]); cls = o["label"]
        ax.scatter(bg[:, 0], bg[:, 1], s=4, c="0.82", alpha=0.6, linewidths=0)     # all classes
        same = y == cls
        ax.scatter(bg[same, 0], bg[same, 1], s=9, c="#3b7dd8", alpha=0.7, linewidths=0,
                   label=f"{obj} class")                                            # its cluster
        tr = traj[:t + 1]
        ax.plot(tr[:, 0], tr[:, 1], "-", c="#ff7f0e", lw=1.4, alpha=0.85)           # trail
        ax.scatter(traj[t, 0], traj[t, 1], s=90, c="#ff7f0e", edgecolors="k",
                   linewidths=0.8, zorder=5)                                         # current
        cos = o["cos"][t]
        ax.set_title(title, fontsize=10.5)
        ax.text(0.03, 0.96, f"cos(emb$_0$,emb$_t$) = {cos:.3f}\ndrift = {o['drift']:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        lo, hi = bg.min(0) - 1, bg.max(0) + 1
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


anim = FuncAnimation(fig, draw, frames=len(frames), interval=90)
anim.save(OUT, writer=PillowWriter(fps=11))
print("saved", OUT, "(%d frames)" % len(frames))

"""Render a rotation-invariance embedding-drift GIF from embed_drift.json.

Left: a point cloud spinning through a full turn. Middle/right: where its embedding lands
in a fixed PCA(2) of the whole ModelNet40 test set, for two encoder views. A live "drift"
counter (distance travelled from angle 0) shows how far the point has wandered right now,
which is where the with/without-JEPA gap really shows.

Run:
  python render_drift.py <src.json> <out.gif> left=projector:With JEPA right=random:Without JEPA
Defaults reproduce projector vs backbone.
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
args = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)


def spec(s, default_key, default_label):
    if s is None:
        return default_key, default_label
    key, _, label = s.partition(":")
    return key, (label or key)


LEFT = spec(args.get("left"), "projector", "Projector")
RIGHT = spec(args.get("right"), "backbone", "Backbone")
TITLE = args.get("title", "Spin a shape, watch its embedding  ·  ModelNet40")
OBJECTS = ["airplane", "chair", "guitar"]

d = json.load(open(SRC))
angles = d["projector"]["angles"]; AX = np.array(d["projector"]["axis"]); NA = len(angles)
GOOD, BAD = "#1f9e6b", "#e23b2e"          # with-JEPA (green) / without (red)


def rot_about(axis, a):
    x, y, z = axis; c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def panel_color(key):
    return GOOD if key == "projector" else BAD


frames = [(o, t) for o in OBJECTS for t in range(NA)]

fig = plt.figure(figsize=(13, 4.7))
ax3d = fig.add_subplot(1, 3, 1, projection="3d")
axL = fig.add_subplot(1, 3, 2)
axR = fig.add_subplot(1, 3, 3)
fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.04, wspace=0.16)
fig.suptitle(TITLE, fontsize=12.5, weight="bold")


def draw(i):
    obj, t = frames[i]
    for ax in (ax3d, axL, axR):
        ax.clear()
    pts = np.array(d["projector"]["objects"][obj]["pts"])
    R = rot_about(AX, angles[t]); P = pts @ R.T
    ax3d.scatter(P[:, 0], P[:, 2], P[:, 1], s=3, c=P[:, 1], cmap="viridis", depthshade=True)
    ax3d.set_title(obj, fontsize=11); ax3d.set_axis_off()
    ax3d.set_xlim(-1, 1); ax3d.set_ylim(-1, 1); ax3d.set_zlim(-1, 1)
    ax3d.view_init(elev=18, azim=35)
    for ax, (key, label) in zip((axL, axR), (LEFT, RIGHT)):
        col = panel_color(key)
        v = d[key]; bg = np.array(v["bg"]); y = np.array(v["bg_y"])
        o = v["objects"][obj]; traj = np.array(o["traj"]); cls = o["label"]
        ax.scatter(bg[:, 0], bg[:, 1], s=4, c="0.83", alpha=0.6, linewidths=0)
        same = y == cls
        ax.scatter(bg[same, 0], bg[same, 1], s=9, c="#3b7dd8", alpha=0.7, linewidths=0,
                   label=f"{obj} class")
        tr = traj[:t + 1]
        ax.plot(tr[:, 0], tr[:, 1], "-", c=col, lw=1.6, alpha=0.9)
        ax.scatter(traj[t, 0], traj[t, 1], s=95, c=col, edgecolors="k", linewidths=0.8, zorder=5)
        live = float(np.linalg.norm(traj[t] - traj[0]))          # distance travelled now
        ax.set_title(label, fontsize=11.5, color=col, weight="bold")
        ax.text(0.035, 0.955, f"drift  {live:4.1f}", transform=ax.transAxes, va="top",
                fontsize=20, weight="bold", color=col, family="monospace")
        ax.text(0.04, 0.80, f"cos {o['cos'][t]:.3f}", transform=ax.transAxes, va="top",
                fontsize=9.5, color="0.35", family="monospace")
        lo, hi = bg.min(0) - 1, bg.max(0) + 1
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


anim = FuncAnimation(fig, draw, frames=len(frames), interval=90)
anim.save(OUT, writer=PillowWriter(fps=11))
print("saved", OUT, "(%d frames)" % len(frames))

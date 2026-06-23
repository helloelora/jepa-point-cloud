"""Render a rotation-invariance embedding-drift GIF, styled like the interactive demo.

Three cards: a spinning point cloud (dark), then where its embedding lands in a fixed PCA(2)
of the ModelNet40 test set for two encoder views. A live "drift" counter (distance moved
from angle 0) is the headline number. Reproduces the look of demo/index.html as a static GIF.

Run:
  python render_drift.py <src.json> <out.gif> left=projector right=random
Defaults: left=projector right=random (the with/without-JEPA hero).
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/embed_drift.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "assets/jepa_vs_nojepa.gif"
args = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
LKEY = args.get("left", "projector")
RKEY = args.get("right", "random")
OBJECTS = ["airplane", "chair", "guitar"]

# --- demo palette ---
GROUND, PAPER, VOID = "#E6EAEF", "#F4F6F9", "#0E1722"
INK, MUTED, PT = "#15212E", "#6C7A89", "#AEB9C6"
ACCENT, BLUE = "#F25C18", "#2563A8"
GREEN, RED = "#1f9e6b", "#e23b2e"
MONO, SANS = "monospace", "sans-serif"

# per-view label / tag / color
STYLE = {
    "projector": dict(title="PROJECTOR", tag="WITH JEPA", color=GREEN,
                      tagfc="#dbe6f5", tagtc=BLUE, font=MONO),
    "backbone": dict(title="BACKBONE", tag="POSE-SENSITIVE", color=ACCENT,
                     tagfc="#fbe2d4", tagtc=ACCENT, font=MONO),
    "random": dict(title="Without JEPA", tag="RANDOM INIT", color=RED,
                   tagfc="#fbe2d4", tagtc=ACCENT, font=SANS),
}

d = json.load(open(SRC))
angles = d["projector"]["angles"]; AX = np.array(d["projector"]["axis"]); NA = len(angles)


def rot_about(axis, a):
    x, y, z = axis; c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


frames = [(o, t) for o in OBJECTS for t in range(NA)]

fig = plt.figure(figsize=(13.2, 5.7)); fig.patch.set_facecolor(GROUND)
R_OBJ = [0.012, 0.20, 0.265, 0.74]
R_L = [0.31, 0.20, 0.312, 0.74]
R_R = [0.655, 0.20, 0.312, 0.74]
R_BAR = [0.012, 0.045, 0.955, 0.10]
bgax = fig.add_axes([0, 0, 1, 1]); bgax.set_axis_off()
bgax.set_xlim(0, 1); bgax.set_ylim(0, 1); bgax.set_zorder(0)


def card(rect, fc):
    p = 0.012
    bgax.add_patch(FancyBboxPatch((rect[0]+p, rect[1]+p), rect[2]-2*p, rect[3]-2*p,
        boxstyle="round,pad=%.3f,rounding_size=0.018" % p, transform=bgax.transAxes,
        fc=fc, ec="#C7D0DA", lw=1.0, mutation_aspect=2.4))


card(R_OBJ, VOID); card(R_L, PAPER); card(R_R, PAPER); card(R_BAR, PAPER)
ax3d = fig.add_axes(R_OBJ, projection="3d"); ax3d.set_zorder(1)
axL = fig.add_axes(R_L); axR = fig.add_axes(R_R); axbar = fig.add_axes(R_BAR)
for a in (axL, axR, axbar):
    a.set_zorder(1)


def tag(ax, x, y, st):
    ax.text(x, y, st["title"], transform=ax.transAxes, va="center", ha="left",
            fontsize=12.5, weight="bold", family=st["font"],
            color=(RED if st["font"] == SANS else INK))
    tx = x + 0.027 * len(st["title"]) + 0.04
    ax.text(tx, y, " " + st["tag"] + " ", transform=ax.transAxes, va="center", ha="left",
            fontsize=8.5, weight="bold", family=MONO, color=st["tagtc"],
            bbox=dict(boxstyle="round,pad=0.3", fc=st["tagfc"], ec="none"))


def embed(ax, key, obj, t):
    ax.clear(); ax.set_facecolor("none"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    st = STYLE[key]; v = d[key]; bg = np.array(v["bg"]); y = np.array(v["bg_y"])
    o = v["objects"][obj]; traj = np.array(o["traj"]); cls = o["label"]
    ax.scatter(bg[:, 0], bg[:, 1], s=5, c=PT, alpha=0.45, linewidths=0)
    same = y == cls
    ax.scatter(bg[same, 0], bg[same, 1], s=11, c=BLUE, alpha=0.8, linewidths=0)
    tr = traj[:t+1]
    ax.plot(tr[:, 0], tr[:, 1], "-", c=ACCENT, lw=1.5, alpha=0.85)
    ax.scatter(traj[t, 0], traj[t, 1], s=230, c=ACCENT, alpha=0.18, linewidths=0)
    ax.scatter(traj[t, 0], traj[t, 1], s=70, c=ACCENT, edgecolors=INK, linewidths=1.0, zorder=5)
    live = float(np.linalg.norm(traj[t] - traj[0]))
    tag(ax, 0.04, 0.95, st)
    ax.text(0.965, 0.95, f"{live:.1f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=34, weight="bold", family=SANS, color=st["color"])
    ax.text(0.965, 0.79, "DRIFT · DISTANCE MOVED", transform=ax.transAxes, ha="right",
            va="top", fontsize=7.5, family=MONO, color=MUTED)
    ax.text(0.965, 0.73, f"cos {o['cos'][t]:.3f}", transform=ax.transAxes, ha="right",
            va="top", fontsize=8.5, family=MONO, color=MUTED)
    lo, hi = bg.min(0) - 1, bg.max(0) + 1
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])


def draw(i):
    obj, t = frames[i]
    ax3d.clear(); ax3d.set_axis_off(); ax3d.patch.set_alpha(0)
    pts = np.array(d["projector"]["objects"][obj]["pts"]); P = pts @ rot_about(AX, angles[t]).T
    ax3d.scatter(P[:, 0], P[:, 2], P[:, 1], s=4, c=P[:, 1], cmap="viridis", depthshade=True)
    ax3d.set_xlim(-1, 1); ax3d.set_ylim(-1, 1); ax3d.set_zlim(-1, 1); ax3d.view_init(elev=18, azim=35)
    ax3d.text2D(0.04, 0.95, "OBJECT", transform=ax3d.transAxes, fontsize=10, family=MONO, color="#7d93a8")
    ax3d.text2D(0.04, 0.02, obj, transform=ax3d.transAxes, fontsize=12, family=MONO, color="#cbd6e2")
    embed(axL, LKEY, obj, t); embed(axR, RKEY, obj, t)
    # bottom bar: play pill + slider + angle
    axbar.clear(); axbar.set_xlim(0, 1); axbar.set_ylim(0, 1); axbar.axis("off")
    axbar.add_patch(FancyBboxPatch((0.012, 0.28), 0.10, 0.44, boxstyle="round,pad=0.02,rounding_size=0.05",
        fc=INK, ec="none", transform=axbar.transAxes))
    axbar.text(0.062, 0.5, "❚❚ Pause", transform=axbar.transAxes, ha="center", va="center",
               fontsize=10, family=MONO, color="#fff")
    x0, x1 = 0.16, 0.90; frac = t / (NA - 1)
    axbar.plot([x0, x1], [0.5, 0.5], "-", c="#C7D0DA", lw=3, transform=axbar.transAxes)
    axbar.plot([x0, x0 + (x1 - x0) * frac], [0.5, 0.5], "-", c=ACCENT, lw=3, transform=axbar.transAxes)
    axbar.scatter([x0 + (x1 - x0) * frac], [0.5], s=130, c=ACCENT, edgecolors="#fff",
                  linewidths=1.2, transform=axbar.transAxes, zorder=5)
    axbar.text(0.965, 0.5, f"{round(angles[t]*180/np.pi)}°", transform=axbar.transAxes,
               ha="right", va="center", fontsize=20, family=MONO, color=INK)


anim = FuncAnimation(fig, draw, frames=len(frames), interval=90)
anim.save(OUT, writer=PillowWriter(fps=11))
print("saved", OUT, "(%d frames)" % len(frames))

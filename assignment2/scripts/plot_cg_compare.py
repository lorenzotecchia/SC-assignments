import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

df = pd.read_csv("output/cg_compare.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# ── 1. Per-step solve time ────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(df["step"], df["sor_solve_ms"], alpha=0.5, linewidth=0.8, label="SOR")
ax.plot(df["step"], df["cg_solve_ms"], alpha=0.5, linewidth=0.8, label="CG")
# Rolling average
w = 10
ax.plot(df["step"], df["sor_solve_ms"].rolling(w, center=True).mean(),
        linewidth=2, label=f"SOR (avg {w})")
ax.plot(df["step"], df["cg_solve_ms"].rolling(w, center=True).mean(),
        linewidth=2, label=f"CG (avg {w})")
ax.set_xlabel("Growth step")
ax.set_ylabel("Solve time (ms)")
ax.set_title("Per-step solve time")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 2. Per-step solver iterations ─────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(df["step"], df["sor_iters"], alpha=0.5, linewidth=0.8, label="SOR")
ax.plot(df["step"], df["cg_iters"], alpha=0.5, linewidth=0.8, label="CG")
ax.plot(df["step"], df["sor_iters"].rolling(w, center=True).mean(),
        linewidth=2, label=f"SOR (avg {w})")
ax.plot(df["step"], df["cg_iters"].rolling(w, center=True).mean(),
        linewidth=2, label=f"CG (avg {w})")
ax.set_xlabel("Growth step")
ax.set_ylabel("Solver iterations")
ax.set_title("Per-step solver iterations")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 3. Cumulative solve time ─────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(df["step"], df["sor_solve_ms"].cumsum(), linewidth=2, label="SOR")
ax.plot(df["step"], df["cg_solve_ms"].cumsum(), linewidth=2, label="CG")
ax.set_xlabel("Growth step")
ax.set_ylabel("Cumulative solve time (ms)")
ax.set_title("Cumulative solve time")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 4. Concentration difference heatmap ───────────────────────────────────────
ax = axes[1, 1]
try:
    diff = np.loadtxt("output/cg_conc_diff.txt")
    im = ax.imshow(diff, origin="lower", extent=[0, 1, 0, 1], cmap="hot")
    fig.colorbar(im, ax=ax, label=r"$|c_{\mathrm{SOR}} - c_{\mathrm{CG}}|$",
                 shrink=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Concentration difference (max = {diff.max():.2e})")
except Exception:
    ax.text(0.5, 0.5, "cg_conc_diff.txt\nnot found", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_title("Concentration difference")

fig.suptitle("DLA Solver Comparison: SOR vs Eigen-CG", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig("output/plots/cg_compare.png", dpi=150, bbox_inches="tight")
print("saved output/plots/cg_compare.png")
plt.close("all")

# ── DLA animation from CG solver ─────────────────────────────────────────────
from matplotlib.animation import FuncAnimation

growth = np.loadtxt("output/cg_growth_log.txt", dtype=int)

snapshots = []
snap_steps = []
with open("output/cg_snapshots.txt") as f:
    current = []
    for line in f:
        line = line.strip()
        if line.startswith("# step"):
            if current:
                snapshots.append(np.array(current, dtype=float))
                current = []
            snap_steps.append(int(line.split()[-1]))
        elif line:
            current.append([float(x) for x in line.split()])
    if current:
        snapshots.append(np.array(current, dtype=float))

snap_steps = np.array(snap_steps)
N = snapshots[0].shape[0]

def build_cluster_at_step(step):
    mask = np.zeros((N, N), dtype=int)
    mask[0, N // 2] = 1
    for s, i, j in growth:
        if s <= step:
            mask[i, j] = 1
        else:
            break
    return mask

fig, (ax_cluster, ax_conc) = plt.subplots(1, 2, figsize=(12, 5))

im_cluster = ax_cluster.imshow(
    np.zeros((N, N)), cmap=plt.cm.binary, origin="lower",
    extent=[0, 1, 0, 1], vmin=0, vmax=1,
)
ax_cluster.set_xlabel("x")
ax_cluster.set_ylabel("y")
ax_cluster.set_aspect("equal")
title_cluster = ax_cluster.set_title("DLA Cluster – CG (step 0)")

im_conc = ax_conc.imshow(
    snapshots[0], cmap=plt.cm.viridis, origin="lower",
    extent=[0, 1, 0, 1], vmin=0, vmax=1,
)
fig.colorbar(im_conc, ax=ax_conc, label=r"$c(x,y)$", shrink=0.8)
ax_conc.set_xlabel("x")
ax_conc.set_ylabel("y")
ax_conc.set_aspect("equal")
title_conc = ax_conc.set_title("Concentration – CG (step 0)")
fig.tight_layout()

def update(frame_idx):
    step = snap_steps[frame_idx]
    im_cluster.set_data(build_cluster_at_step(step))
    title_cluster.set_text(f"DLA Cluster – CG (step {step})")
    im_conc.set_data(snapshots[frame_idx])
    title_conc.set_text(f"Concentration – CG (step {step})")
    return im_cluster, im_conc

anim = FuncAnimation(fig, update, frames=len(snapshots),
                     interval=100, blit=False, repeat=True)
anim.save("output/plots/cg_dla_animation.gif", writer="pillow", fps=10, dpi=100)
print(f"saved output/plots/cg_dla_animation.gif ({len(snapshots)} frames)")
plt.close("all")

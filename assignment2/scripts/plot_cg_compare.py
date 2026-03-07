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

# ── Load multi-run data ──────────────────────────────────────────────────────
mdf = pd.read_csv("output/cg_compare_multi.csv")
n_seeds = mdf["seed"].nunique()

solvers = sorted(mdf["solver"].unique())
colors = {"SOR_cold": "#d62728", "SOR_warm": "#2ca02c", "CG_warm": "#1f77b4",
           "SOR": "#d62728", "CG": "#1f77b4"}

def pivot_stats(df, col):
    piv = df.pivot(index="step", columns="seed", values=col)
    return piv.index.values, piv.mean(axis=1).values, piv.std(axis=1).values, piv

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# ── 1. Per-step solve time (mean ± std) ──────────────────────────────────────
ax = axes[0, 0]
for s in solvers:
    sub = mdf[mdf["solver"] == s]
    steps, mu, sigma, _ = pivot_stats(sub, "solve_ms")
    c = colors.get(s, None)
    ax.plot(steps, mu, linewidth=2, label=s, color=c)
    ax.fill_between(steps, mu - sigma, mu + sigma, alpha=0.20, color=c)
ax.set_xlabel("Growth step")
ax.set_ylabel("Solve time (ms)")
ax.set_title(f"Per-step solve time (mean ± 1σ, n={n_seeds})")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 2. Per-step solver iterations (mean ± std) ──────────────────────────────
ax = axes[0, 1]
for s in solvers:
    sub = mdf[mdf["solver"] == s]
    steps, mu, sigma, _ = pivot_stats(sub, "iters")
    c = colors.get(s, None)
    ax.plot(steps, mu, linewidth=2, label=s, color=c)
    ax.fill_between(steps, mu - sigma, mu + sigma, alpha=0.20, color=c)
ax.set_xlabel("Growth step")
ax.set_ylabel("Solver iterations")
ax.set_title(f"Per-step solver iterations (mean ± 1σ, n={n_seeds})")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 3. Cumulative solve time (mean ± std) ────────────────────────────────────
ax = axes[1, 0]
for s in solvers:
    sub = mdf[mdf["solver"] == s]
    _, _, _, piv = pivot_stats(sub, "solve_ms")
    cum = piv.cumsum(axis=0)
    steps = cum.index.values
    mu = cum.mean(axis=1).values
    sigma = cum.std(axis=1).values
    c = colors.get(s, None)
    ax.plot(steps, mu, linewidth=2, label=s, color=c)
    ax.fill_between(steps, mu - sigma, mu + sigma, alpha=0.20, color=c)
ax.set_xlabel("Growth step")
ax.set_ylabel("Cumulative solve time (ms)")
ax.set_title(f"Cumulative solve time (mean ± 1σ, n={n_seeds})")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 4. Concentration difference heatmap ──────────────────────────────────────
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

fig.suptitle(f"DLA Solver Comparison ({n_seeds} seeds)",
             fontsize=15, y=1.01)
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

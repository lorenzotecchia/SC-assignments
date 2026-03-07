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

sor_df = mdf[mdf["solver"] == "SOR"]
cg_df  = mdf[mdf["solver"] == "CG"]

sor_piv_ms    = sor_df.pivot(index="step", columns="seed", values="solve_ms")
sor_piv_iters = sor_df.pivot(index="step", columns="seed", values="iters")
cg_piv_ms     = cg_df.pivot(index="step", columns="seed", values="solve_ms")
cg_piv_iters  = cg_df.pivot(index="step", columns="seed", values="iters")

steps = sor_piv_ms.index.values

def mean_std(piv):
    return piv.mean(axis=1).values, piv.std(axis=1).values

sor_ms_mean, sor_ms_std = mean_std(sor_piv_ms)
cg_ms_mean,  cg_ms_std  = mean_std(cg_piv_ms)
sor_it_mean, sor_it_std = mean_std(sor_piv_iters)
cg_it_mean,  cg_it_std  = mean_std(cg_piv_iters)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# ── 1. Per-step solve time (mean ± std) ──────────────────────────────────────
ax = axes[0, 0]
ax.plot(steps, sor_ms_mean, linewidth=2, label="SOR")
ax.fill_between(steps, sor_ms_mean - sor_ms_std, sor_ms_mean + sor_ms_std,
                alpha=0.25)
ax.plot(steps, cg_ms_mean, linewidth=2, label="CG")
ax.fill_between(steps, cg_ms_mean - cg_ms_std, cg_ms_mean + cg_ms_std,
                alpha=0.25)
ax.set_xlabel("Growth step")
ax.set_ylabel("Solve time (ms)")
ax.set_title(f"Per-step solve time (mean ± 1σ, n={n_seeds})")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 2. Per-step solver iterations (mean ± std) ──────────────────────────────
ax = axes[0, 1]
ax.plot(steps, sor_it_mean, linewidth=2, label="SOR")
ax.fill_between(steps, sor_it_mean - sor_it_std, sor_it_mean + sor_it_std,
                alpha=0.25)
ax.plot(steps, cg_it_mean, linewidth=2, label="CG")
ax.fill_between(steps, cg_it_mean - cg_it_std, cg_it_mean + cg_it_std,
                alpha=0.25)
ax.set_xlabel("Growth step")
ax.set_ylabel("Solver iterations")
ax.set_title(f"Per-step solver iterations (mean ± 1σ, n={n_seeds})")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 3. Cumulative solve time (mean ± std) ────────────────────────────────────
ax = axes[1, 0]
sor_cum = sor_piv_ms.cumsum(axis=0)
cg_cum  = cg_piv_ms.cumsum(axis=0)
sor_cum_mean, sor_cum_std = mean_std(sor_cum)
cg_cum_mean,  cg_cum_std  = mean_std(cg_cum)
ax.plot(steps, sor_cum_mean, linewidth=2, label="SOR")
ax.fill_between(steps, sor_cum_mean - sor_cum_std,
                sor_cum_mean + sor_cum_std, alpha=0.25)
ax.plot(steps, cg_cum_mean, linewidth=2, label="CG")
ax.fill_between(steps, cg_cum_mean - cg_cum_std,
                cg_cum_mean + cg_cum_std, alpha=0.25)
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

fig.suptitle(f"DLA Solver Comparison: SOR vs Eigen-CG ({n_seeds} seeds)",
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

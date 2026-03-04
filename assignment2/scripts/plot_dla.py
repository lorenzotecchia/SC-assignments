import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ── Load growth log: step i j ────────────────────────────────────────
growth = np.loadtxt("output/dla_growth_log.txt", dtype=int)

# ── Load concentration snapshots ─────────────────────────────────────
snapshots = []
snap_steps = []
with open("output/dla_snapshots.txt") as f:
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

# ── Build cluster mask at each snapshot step ─────────────────────────
def build_cluster_at_step(step):
    mask = np.zeros((N, N), dtype=int)
    mask[0, N // 2] = 1  # seed
    for s, i, j in growth:
        if s <= step:
            mask[i, j] = 1
        else:
            break
    return mask

# ── Animation ────────────────────────────────────────────────────────
fig, (ax_cluster, ax_conc) = plt.subplots(1, 2, figsize=(12, 5))

cmap_cluster = plt.cm.binary
cmap_conc = plt.cm.viridis

im_cluster = ax_cluster.imshow(
    np.zeros((N, N)), cmap=cmap_cluster, origin="lower",
    extent=[0, 1, 0, 1], vmin=0, vmax=1,
)
ax_cluster.set_xlabel("x")
ax_cluster.set_ylabel("y")
ax_cluster.set_aspect("equal")
title_cluster = ax_cluster.set_title("DLA Cluster (step 0)")

im_conc = ax_conc.imshow(
    snapshots[0], cmap=cmap_conc, origin="lower",
    extent=[0, 1, 0, 1], vmin=0, vmax=1,
)
fig.colorbar(im_conc, ax=ax_conc, label=r"$c(x,y)$", shrink=0.8)
ax_conc.set_xlabel("x")
ax_conc.set_ylabel("y")
ax_conc.set_aspect("equal")
title_conc = ax_conc.set_title("Concentration (step 0)")

fig.tight_layout()

def update(frame_idx):
    step = snap_steps[frame_idx]
    cluster = build_cluster_at_step(step)
    im_cluster.set_data(cluster)
    title_cluster.set_text(f"DLA Cluster (step {step})")

    im_conc.set_data(snapshots[frame_idx])
    title_conc.set_text(f"Concentration (step {step})")
    return im_cluster, im_conc

anim = FuncAnimation(
    fig, update, frames=len(snapshots),
    interval=100, blit=False, repeat=True,
)

anim.save("output/plots/dla_animation.gif", writer="pillow", fps=10, dpi=100)
print(f"saved output/plots/dla_animation.gif ({len(snapshots)} frames)")
plt.close("all")

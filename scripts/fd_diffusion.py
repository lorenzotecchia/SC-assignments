import json

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

# TODO theoretical value does not act like simulated one :c


# function to compute theoretical value
def theoretical_value(y: np.ndarray, t: float, n: int, diffusion_constant: float):
    result = np.zeros_like(y)
    for i in range(int(n)):
        result = (
            result
            + erfc((1.0 - y + 2.0 * i) / (2.0 * np.sqrt(diffusion_constant * t)))
            - erfc((1.0 + y + 2.0 * i) / (2.0 * np.sqrt(diffusion_constant * t)))
        )
    return result


# open file with metadata used in cpp simulation
with open("output/fd_diffusion_params.txt") as f:
    meta = json.load(f)

n_rows = meta["x_num"]
n_columns = n_rows
t_save = meta["t_save"]
diffusion_constant = meta["diffusion_constant"]
t_delta_save = meta["t_delta_save"]

# extract data computed in cpp simulation
with open("output/fd_diffusion_data.txt", "r") as f:
    lines = [line for line in f if line.strip()]  # skip blank lines
data = np.loadtxt(lines)
data = data.reshape((t_save, n_rows, n_columns))

# time steps and y-coordinate arrays
time_data = np.array([i * t_delta_save for i in range(int(t_save))])
y = np.linspace(0, 1, n_rows)  # y-axis coordinates

# theoretical solution at each time step:
theory_concentration = np.zeros((int(t_save), n_rows))
theory_concentration[0][-1] = 1.0
for t in range(1, int(t_save)):
    theory_concentration[t] = theoretical_value(
        y, t * t_delta_save, 1000, diffusion_constant
    )

"""
fig, ax = plt.subplots()
im = ax.imshow(data[0], origin="lower", cmap="Purples", interpolation="gaussian")


def update(frame):
    im.set_data(data[frame])
    return [im]


ani = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=50)
plt.show()

"""

# Choose a column to track (e.g., middle column)
col_index = n_columns // 2

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))
ax2 = ax.twinx()
(line_theory,) = ax.plot([], [], lw=2, c="r", ls="-")
scatter_data = ax2.scatter([], [], s=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax.set_xlabel("y")
ax.set_ylabel("Concentration")
ax.set_title(f"Concentration along column {col_index}")
ax.grid(True, alpha=0.3)


# Update function for animation
def update(frame):
    line_theory.set_data(y, theory_concentration[frame])
    scatter_data.set_offsets(np.c_[y, data[frame, :, col_index]])
    ax.set_title(f"Time step {frame}")
    return (
        scatter_data,
        line_theory,
    )


# Create animation
ani = animation.FuncAnimation(fig, update, frames=data.shape[0], blit=True, interval=50)

# ── 3 static snapshots (early / mid / late) ───────────────────────────
n_frames = data.shape[0]
snap_frames = [1, n_frames // 2, n_frames - 1]
fig_snap, axes_snap = plt.subplots(1, 3, figsize=(14, 4))
fig_snap.suptitle("Diffusion — concentration snapshots", fontsize=13)
for ax_s, fr in zip(axes_snap, snap_frames):
    t_val = fr * t_delta_save
    ax_s.plot(y, theory_concentration[fr], lw=2, c="r", ls="-", label="theory")
    ax_s.scatter(y, data[fr, :, col_index], s=10, label="simulation")
    ax_s.set_xlim(0, 1)
    ax_s.set_ylim(0, 1)
    ax_s.set_xlabel("y")
    ax_s.set_ylabel("Concentration")
    ax_s.set_title(f"t = {t_val:.3f}")
    ax_s.legend(fontsize=8)
    ax_s.grid(True, alpha=0.3)
fig_snap.tight_layout()
fig_snap.savefig("output/plots/fd_diffusion_snapshots.png", dpi=150)

ani.save("output/plots/fd_diffusion.gif", writer="pillow", fps=20, dpi=80)
plt.close('all')

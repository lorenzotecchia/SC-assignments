import json
import logging

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.special import erfc

logging.getLogger("matplotlib.backends.backend_ps").setLevel(logging.ERROR)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


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

# Choose a column to track (e.g., middle column)
col_index = n_columns // 2
data_mid = data[:, :, col_index]

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
    scatter_data.set_offsets(np.c_[y, data_mid[frame]])
    ax.set_title(f"Time step {frame}")
    return (
        scatter_data,
        line_theory,
    )


# Create animation
ani = animation.FuncAnimation(fig, update, frames=data.shape[0], blit=True, interval=50)

# ── Static snapshots ──────────────────────────────────────────────────
times_to_plot = np.array([0, 0.001, 0.01, 0.1, 1.0])
indices_to_plot = np.clip(
    np.floor(times_to_plot / t_delta_save).astype(int), 0, int(t_save) - 1
)
colors = cm.viridis(np.linspace(0, 1, len(times_to_plot)))

fig_snap, ax_snap = plt.subplots(figsize=(8, 5))

for k, (idx, color) in enumerate(zip(indices_to_plot, colors)):
    t_val = idx * t_delta_save
    ax_snap.plot(y, theory_concentration[idx], lw=2, color=color, ls="-")
    ax_snap.scatter(y, data_mid[idx], s=10, color=color, label=f"t = {t_val:g}s")

handles_type = [
    Line2D([0], [0], color="k", lw=2, linestyle="-", label="Theory"),
    Line2D([0], [0], marker="o", linestyle="None", color="k", markersize=5, label="Simulation"),
]
handles_time = [
    Line2D([0], [0], marker="o", linestyle="-", color=colors[k], lw=2, markersize=5,
           label=f"t = {times_to_plot[k]:g}s")
    for k in range(len(times_to_plot))
]
ax_snap.legend(handles=handles_type + handles_time, loc="upper left", fontsize=9)
ax_snap.set_xlim(0, 1)
ax_snap.set_ylim(0, 1)
ax_snap.set_title("Diffusion — Concentration Snapshots", fontsize=14)
ax_snap.set_xlabel("$y$")
ax_snap.set_ylabel("Concentration $c(y,t)$")
ax_snap.grid(True, alpha=0.3)
fig_snap.tight_layout()
fig_snap.savefig("output/plots/fd_diffusion_snapshots.eps", format="eps")
fig_snap.savefig("output/plots/fd_diffusion_snapshots.png", dpi=150)
plt.close(fig_snap)

# ── Error plot ────────────────────────────────────────────────────────
error = np.abs(theory_concentration - data_mid).sum(axis=1)

fig_err, ax_err = plt.subplots(figsize=(8, 5))
# Skip t=0 to avoid log(0) issues
valid = time_data > 0
ax_err.semilogy(time_data[valid], error[valid], "o-", ms=3, lw=1.5, color="#2196F3")
ax_err.set_title("Diffusion — Absolute Error Over Time", fontsize=14)
ax_err.set_xlabel("Time $t$")
ax_err.set_ylabel("Error (sum of absolute differences)")
ax_err.grid(True, alpha=0.3)
fig_err.tight_layout()
fig_err.savefig("output/plots/fd_diffusion_error.eps", format="eps")
fig_err.savefig("output/plots/fd_diffusion_error.png", dpi=150)
plt.close(fig_err)

ani.save("output/plots/fd_diffusion.gif", writer="pillow", fps=20, dpi=80)
plt.close("all")

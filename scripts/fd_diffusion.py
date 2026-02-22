import json

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

# ── 3 static snapshots ────────────────────────────────────────────────
times_to_plot = np.array([0, 0.001, 0.01, 0.1, 1.0])
indices_to_plot = np.floor(times_to_plot / t_delta_save).astype(int)

plt.figure()

for k, idx in enumerate(indices_to_plot):
    plt.plot(y, theory_concentration[idx], lw=2, c="r", ls="-")
    plt.scatter(y, data_mid[idx], s=10)

    # annotate near where the curve crosses c = 0.5
    y_label_idx = np.argmin(np.abs(theory_concentration[idx] - 0.5))
    plt.text(
        y[y_label_idx] + 0.01,
        0.5 + 0.03,
        f"t={times_to_plot[k]:g}s",
        fontsize=8,
        color="k",
        va="bottom",
        ha="left",
        clip_on=False,
    )

handles = [
    Line2D([0], [0], color="r", lw=2, linestyle="-", label="Theory"),
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        color="C0",
        markersize=5,
        label="Simulation",
    ),
]
plt.legend(handles=handles, fontsize=8)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Diffusion — concentration snapshots")
plt.xlabel("y")
plt.ylabel("Concentration")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.subplots_adjust(right=0.82)
plt.savefig("output/plots/fd_diffusion_snapshots.png", dpi=150)
plt.close()

# ── 4 error plot ──────────────────────────────────────────────────────
error = np.abs(theory_concentration - data_mid).sum(axis=1)

plt.figure()
plt.scatter(time_data, error, s=10)
plt.title("Diffusion — absolute error over time")
plt.xlim(0, 1)
plt.xlabel("time")
plt.ylabel("error (abs)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output/plots/fd_diffusion_error.png", dpi=150)

ani.save("output/plots/fd_diffusion.gif", writer="pillow", fps=20, dpi=80)
plt.close("all")

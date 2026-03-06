import json
import logging

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.special import erfc

logging.getLogger("matplotlib.backends.backend_ps").setLevel(logging.ERROR)

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


# open file with metadata used in cpp simulation
with open("output/gray_scott_params.txt") as f:
    meta = json.load(f)

n_rows = meta["x_num"]
n_columns = n_rows
t_save = meta["t_save"]
t_delta_save = meta["t_delta_save"]

# extract data computed in cpp simulation
with open("output/gray_scott_u_data.txt", "r") as f:
    lines = [line for line in f if line.strip()]  # skip blank lines
u_data = np.loadtxt(lines)
u_data = u_data.reshape((t_save, n_rows, n_columns))

with open("output/gray_scott_v_data.txt", "r") as f:
    lines = [line for line in f if line.strip()]  # skip blank lines
v_data = np.loadtxt(lines)
v_data = v_data.reshape((t_save, n_rows, n_columns))

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("chemicals concentration - gray scott model", fontsize=14)

# initial heatmaps
im1 = ax1.imshow(u_data[0], cmap="viridis", animated=True, vmin=0, vmax=1)
im2 = ax2.imshow(v_data[0], cmap="magma", animated=True, vmin=0, vmax=1)

ax1.set_title("U concentration")
ax2.set_title("V concentration")


def update(frame):
    im1.set_array(u_data[frame])
    im2.set_array(v_data[frame])
    return im1, im2


ani = animation.FuncAnimation(fig, update, frames=len(u_data), interval=100, blit=True)

# save as mp4
ani.save("output/gray_scott_heatmaps.mp4", writer="ffmpeg", fps=10)

plt.show()

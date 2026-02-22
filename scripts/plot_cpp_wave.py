import logging
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger("matplotlib.backends.backend_ps").setLevel(logging.ERROR)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

x = np.loadtxt("output/x_data.txt")

cases = [
    ("output/case1_leapfrog_wave_data.txt", r"$\Psi(x,0) = \sin(2\pi x)$"),
    ("output/case2_leapfrog_wave_data.txt", r"$\Psi(x,0) = \sin(5\pi x)$"),
    (
        "output/case3_leapfrog_wave_data.txt",
        r"$\Psi(x,0) = \sin(5\pi x)$, $x \in (1/5, 2/5)$",
    ),
]

data = []
for path, _ in cases:
    data.append(np.loadtxt(path))

t2 = 4.0
t_frames = data[0].shape[0]
ymin = min(d.min() for d in data)
ymax = max(d.max() for d in data)

# ── Static snapshot plot ──────────────────────────────────────────────
# Show several time slices per case, with color varying by time.

snapshot_times = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
colors = cm.viridis(np.linspace(0, 1, len(snapshot_times)))

fig_snap, axes_snap = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig_snap.suptitle("Vibrating String — Snapshots at different times", fontsize=14)

for ax, wave_states, (_, title) in zip(axes_snap, data, cases):
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ymin * 1.1, ymax * 1.1)
    ax.set_xlabel("x")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)

    for t_snap, color in zip(snapshot_times, colors):
        frame_idx = int(round(t_snap / t2 * (t_frames - 1)))
        frame_idx = min(frame_idx, t_frames - 1)
        ax.plot(
            x, wave_states[frame_idx], color=color, lw=1.5, label=f"t = {t_snap:.2f}"
        )

axes_snap[0].set_ylabel(r"$\Psi(x, t)$")
axes_snap[2].legend(loc="upper right", fontsize=8)
fig_snap.tight_layout()
fig_snap.savefig("output/plots/cpp_wave_snapshots.eps", format="eps")
fig_snap.savefig("output/plots/cpp_wave_snapshots.png", dpi=150)

# ── Animation ─────────────────────────────────────────────────────────

fig_anim, axes_anim = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig_anim.suptitle("1D Wave Equation — Vibrating String", fontsize=14)

lines = []
for ax, wave_states, (_, title) in zip(axes_anim, data, cases):
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    (line,) = ax.plot([], [], lw=2)
    lines.append((line, wave_states))

axes_anim[0].set_ylabel(r"$\Psi(x, t)$")


def init():
    for line, _ in lines:
        line.set_data([], [])
    return [l for l, _ in lines]


def update(i):
    for line, wave_states in lines:
        line.set_data(x, wave_states[i])
    return [l for l, _ in lines]


ani = animation.FuncAnimation(
    fig=fig_anim, func=update, init_func=init, frames=t_frames, interval=10, blit=True
)

fig_anim.tight_layout()
ani.save("output/plots/cpp_wave.gif", writer="pillow", fps=30, dpi=80)
plt.close('all')

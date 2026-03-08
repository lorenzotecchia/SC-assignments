import json
import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

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


# helpers
def _suffix(f, k):
    return f"_f{f}_k{k}"


def load_metadata(f, k):
    with open(f"output/gray_scott_params{_suffix(f, k)}.txt") as fh:
        return json.load(fh)


def load_all_frames(species, f, k, n_frames, n_rows, n_cols):
    path = f"output/gray_scott_{species}_data{_suffix(f, k)}.txt"
    with open(path) as fh:
        lines = [line for line in fh if line.strip()]
    return np.loadtxt(lines).reshape((n_frames, n_rows, n_cols))


def load_last_frame(species, f, k, n_rows, n_cols):
    path = f"output/gray_scott_{species}_data{_suffix(f, k)}.txt"
    with open(path) as fh:
        all_lines = [line for line in fh if line.strip()]
    return np.loadtxt(all_lines[-n_rows:]).reshape((n_rows, n_cols))


# Comparison plots (U concentration, final snapshot)
# fixing one parameter and letting the other sweep
# Each figure combines two related comparisons as a 2x3 grid
comparison_figures = [
    {
        "rows": [
            {"params": [("0.014", "0.040"), ("0.014", "0.041"), ("0.014", "0.042")]},
            {"params": [("0.022", "0.051"), ("0.023", "0.051"), ("0.025", "0.051")]},
        ],
        "filename": "output/gray_scott_comparison_1.eps",
    },
    {
        "rows": [
            {"params": [("0.034", "0.060"), ("0.035", "0.060"), ("0.036", "0.060")]},
            {"params": [("0.035", "0.059"), ("0.035", "0.058"), ("0.035", "0.057")]},
        ],
        "filename": "output/gray_scott_comparison_2.eps",
    },
]

for fig_cfg in comparison_figures:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for row_idx, row_cfg in enumerate(fig_cfg["rows"]):
        for col_idx, (f, k) in enumerate(row_cfg["params"]):
            meta = load_metadata(f, k)
            n = meta["x_num"]
            frame = load_last_frame("u", f, k, n, n)

            ax = axes[row_idx, col_idx]
            ax.imshow(frame, cmap="inferno", vmin=0, vmax=1)
            ax.set_title(f"f = {f}, k = {k}", fontsize=24)

            is_bottom = row_idx == 1
            is_left = col_idx == 0

            if is_bottom:
                ax.set_xlabel("x")
            else:
                ax.tick_params(labelbottom=False)

            if is_left:
                ax.set_ylabel("y")
            else:
                ax.tick_params(labelleft=False)

    fig.subplots_adjust(hspace=0.15, wspace=0.05)
    fig.savefig(fig_cfg["filename"], format="eps", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_cfg['filename']}")


# Animations (U and V concentration over time, saved as .mp4)
# most interesting plots where chosen
animation_params = [
    ("0.025", "0.051"),
    ("0.035", "0.060"),
    ("0.035", "0.058"),
    ("0.014", "0.040"),
    ("0.014", "0.042"),
    ("0.022", "0.051"),
    ("0.023", "0.051"),
]

for f, k in animation_params:
    meta = load_metadata(f, k)
    n = meta["x_num"]
    t_save = meta["t_save"]
    t_delta_save = meta["t_delta_save"]

    print(f"Loading data for animation f={f}, k={k} ...")
    u_data = load_all_frames("u", f, k, t_save, n, n)
    v_data = load_all_frames("v", f, k, t_save, n, n)

    u_min, u_max = u_data.min(), u_data.max()
    v_min, v_max = v_data.min(), v_data.max()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"Gray-Scott  f = {f}, k = {k}", fontsize=14)

    im1 = ax1.imshow(u_data[0], cmap="viridis", animated=True, vmin=u_min, vmax=u_max)
    im2 = ax2.imshow(v_data[0], cmap="viridis", animated=True, vmin=v_min, vmax=v_max)

    ax1.set_title("U concentration")
    ax2.set_title("V concentration")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    time_text = fig.text(
        0.5, 0.01, "", ha="center", fontsize=11, transform=fig.transFigure
    )

    def _make_update(u_d, v_d, im_u, im_v, t_txt, dt):
        def _update(frame):
            im_u.set_array(u_d[frame])
            im_v.set_array(v_d[frame])
            t_txt.set_text(f"t = {frame * dt:.0f}")
            return im_u, im_v, t_txt

        return _update

    ani = animation.FuncAnimation(
        fig,
        _make_update(u_data, v_data, im1, im2, time_text, t_delta_save),
        frames=t_save,
        interval=50,
        blit=True,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    anim_path = f"output/gray_scott_animation_f{f}_k{k}.mp4"
    ani.save(anim_path, writer="ffmpeg", fps=10)
    plt.close(fig)
    print(f"Saved {anim_path}")

print("=== All plots and animations generated ===")

"""
Visualisation for the FD Kármán Vortex Street solver.

Usage
-----
# 1. Plot drag & lift time series
uv run python fd_ns/plot_fd.py forces

# 2. Plot a single vorticity snapshot
uv run python fd_ns/plot_fd.py vorticity fd_ns/output/vorticity_010000.txt

# 3. Animate all snapshots in output/
uv run python fd_ns/plot_fd.py animate

# 4. Save the animation to an mp4 (needs ffmpeg)
uv run python fd_ns/plot_fd.py animate --save karman.mp4
"""

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

def load_vorticity(path):
    """Load a vorticity txt file into a 2-D array.

    Grid shape is inferred from unique x/y values in the file, so the function
    works regardless of which Re/grid run produced the snapshot.
    """
    data = np.loadtxt(path)          # shape (N, 3): x  y  omega
    idx  = np.lexsort((data[:, 0], data[:, 1]))   # sort: y outer, x inner
    data = data[idx]
    x_vals = np.unique(data[:, 0])
    y_vals = np.unique(data[:, 1])
    ni, nj = len(x_vals), len(y_vals)
    omega  = data[:, 2].reshape(nj, ni)   # (nj, ni) — row=y, col=x
    return x_vals, y_vals, omega


def vorticity_clim(omega, pct=99):
    """Symmetric colour limits at the given percentile."""
    lim = np.percentile(np.abs(omega), pct)
    return -lim, lim


# ── sub-commands ──────────────────────────────────────────────────────────────

def cmd_forces(args):
    csv = Path(args.outdir) / "drag_lift.csv"
    if not csv.exists():
        sys.exit(f"Not found: {csv}")

    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    t, cd, cl = data[:, 0], data[:, 1], data[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, cd, lw=0.8, color="steelblue")
    axes[0].axhline(3.22, ls="--", color="gray", lw=0.8, label="Schäfer-Turek Cd≈3.22")
    axes[0].set_ylabel("C_D")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, lw=0.3)

    axes[1].plot(t, cl, lw=0.8, color="tomato")
    axes[1].set_ylabel("C_L")
    axes[1].set_xlabel("t  [s]")
    axes[1].grid(True, lw=0.3)

    fig.suptitle("FD Kármán Vortex Street — drag and lift", fontsize=11)
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved: {args.save}")
    else:
        plt.show()


def cmd_vorticity(args):
    path = Path(args.file)
    if not path.exists():
        sys.exit(f"Not found: {path}")

    x, y, omega = load_vorticity(path)
    vmin, vmax  = vorticity_clim(omega)

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.pcolormesh(x, y, omega, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       shading="auto")
    plt.colorbar(im, ax=ax, label="ω  [1/s]")

    # Draw cylinder outline
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(0.2 + 0.05*np.cos(theta), 0.2 + 0.05*np.sin(theta),
            "k-", lw=0.8)

    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    ax.set_title(f"Vorticity — {path.name}")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved: {args.save}")
    else:
        plt.show()


def cmd_animate(args):
    files = sorted(Path(args.outdir).glob("vorticity_*.txt"))
    if not files:
        sys.exit(f"No vorticity_*.txt files found in {args.outdir}")

    print(f"Found {len(files)} snapshots.")
    x, y, omega0 = load_vorticity(files[0])
    ref_shape = omega0.shape

    # Keep only files whose grid matches the first snapshot (handles mixed-run dirs).
    def _matches(f):
        try:
            _, _, om = load_vorticity(f)
            return om.shape == ref_shape
        except Exception:
            return False

    files = [f for f in files if _matches(f)]
    print(f"Using {len(files)} snapshots with grid {ref_shape[1]}×{ref_shape[0]}.")
    vmin, vmax   = vorticity_clim(omega0)

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.pcolormesh(x, y, omega0, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       shading="auto")
    plt.colorbar(im, ax=ax, label="ω  [1/s]")

    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(0.2 + 0.05*np.cos(theta), 0.2 + 0.05*np.sin(theta),
            "k-", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")
    title = ax.set_title(files[0].name)
    fig.tight_layout()

    def update(i):
        _, _, omega = load_vorticity(files[i])
        im.set_array(omega.ravel())
        title.set_text(files[i].name)
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(files),
                                  interval=args.interval, blit=True)

    if args.save:
        writer = animation.FFMpegWriter(fps=1000 // args.interval)
        ani.save(args.save, writer=writer, dpi=150)
        print(f"Saved: {args.save}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _add_common(sp):
    """Add flags shared by all subcommands."""
    sp.add_argument("--outdir", default="fd_ns/output", help="output directory")
    sp.add_argument("--save",   default="",             help="save to file instead of showing")


def main():
    p   = argparse.ArgumentParser(description="Visualise FD NS solver output")
    sub = p.add_subparsers(dest="cmd", required=True)

    sf = sub.add_parser("forces", help="Plot drag and lift time series")
    _add_common(sf)

    sv = sub.add_parser("vorticity", help="Plot a single vorticity snapshot")
    _add_common(sv)
    sv.add_argument("file", help="path to vorticity_NNNNNN.txt")

    sa = sub.add_parser("animate", help="Animate all vorticity snapshots")
    _add_common(sa)
    sa.add_argument("--interval", type=int, default=100,
                    help="frame interval in ms (default 100)")

    args = p.parse_args()
    if not hasattr(args, "interval"):
        args.interval = 100   # default for non-animate subcommands

    if args.cmd == "forces":
        cmd_forces(args)
    elif args.cmd == "vorticity":
        cmd_vorticity(args)
    elif args.cmd == "animate":
        cmd_animate(args)


if __name__ == "__main__":
    main()

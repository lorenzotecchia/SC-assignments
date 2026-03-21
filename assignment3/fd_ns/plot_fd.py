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

def load_vorticity(path, nx=440, ny=82):
    """Load a vorticity txt file into a 2-D array (x, y, omega)."""
    data = np.loadtxt(path)          # shape (N, 3): x  y  omega
    # Sort by y then x to get a regular grid order
    idx  = np.lexsort((data[:, 0], data[:, 1]))
    data = data[idx]
    # Inner cells: i=1..nx-2, j=1..ny-2  →  (nx-2)×(ny-2) points
    ni, nj = nx - 2, ny - 2
    omega = data[:, 2].reshape(nj, ni)   # shape (nj, ni) — row=y, col=x
    x     = data[:ni, 0]                 # x values along bottom row
    y     = data[::ni, 1]                # y values along left column
    return x, y, omega


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

    x, y, omega = load_vorticity(path, nx=args.nx, ny=args.ny)
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
    x, y, omega0 = load_vorticity(files[0], nx=args.nx, ny=args.ny)
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
        _, _, omega = load_vorticity(files[i], nx=args.nx, ny=args.ny)
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

def main():
    p = argparse.ArgumentParser(description="Visualise FD NS solver output")
    p.add_argument("--outdir",   default="fd_ns/output", help="output directory")
    p.add_argument("--nx",       type=int, default=440,  help="grid nx (default 440)")
    p.add_argument("--ny",       type=int, default=82,   help="grid ny (default 82)")
    p.add_argument("--save",     default="",             help="save to file instead of showing")
    p.add_argument("--interval", type=int, default=100,  help="animation frame interval ms")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("forces",    help="Plot drag and lift time series")

    sv = sub.add_parser("vorticity", help="Plot a single vorticity snapshot")
    sv.add_argument("file", help="path to vorticity_NNNNNN.txt")

    sub.add_parser("animate", help="Animate all vorticity snapshots")

    args = p.parse_args()

    if args.cmd == "forces":
        cmd_forces(args)
    elif args.cmd == "vorticity":
        cmd_vorticity(args)
    elif args.cmd == "animate":
        cmd_animate(args)


if __name__ == "__main__":
    main()

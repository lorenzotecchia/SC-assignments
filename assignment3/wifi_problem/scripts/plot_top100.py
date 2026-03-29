#!/usr/bin/env python3
"""Plot the top-100 candidate positions over the floor plan.

Usage:
  python3 scripts/plot_top100.py --top output/top100_annealing.csv --floor output/floor_plan.csv --out output/top100_scatter.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_top_candidates(floor_file, top_file, out_file):
    floor = np.loadtxt(floor_file, delimiter=",")

    floor_dx = 10.0 / floor.shape[0]
    xs = np.arange(floor.shape[0]) * floor_dx
    ys = np.arange(floor.shape[1]) * floor_dx

    data = np.genfromtxt(
        top_file, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    if data.size == 0:
        raise SystemExit(f"No points found in {top_file}")

    x = np.atleast_1d(data["x"]).astype(float)
    y = np.atleast_1d(data["y"]).astype(float)

    background_color = "#7ABABD"

    # Two-color colormap: 0 (wall) -> white, 1 (air) -> background_color
    cmap = mcolors.ListedColormap(["white", background_color])

    fig, ax = plt.subplots(figsize=(8, 6))

    # imshow with extent in meters; floor.T so x=horizontal, y=vertical
    x0 = xs[0] - floor_dx / 2
    x1 = xs[-1] + floor_dx / 2
    y0 = ys[0] - floor_dx / 2
    y1 = ys[-1] + floor_dx / 2

    ax.imshow(
        floor.T,
        origin="lower",
        extent=[x0, x1, y0, y1],
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=0,
    )

    ax.scatter(
        x,
        y,
        c="red",
        s=40,
        edgecolors="black",
        linewidth=0.8,
        zorder=3,
        label=f"Top {len(x)}",
    )

    # Measurement points (x, y, name)
    measurement_points = [
        (1.0, 5.0, "Living Room"),
        (2.0, 1.0, "Kitchen"),
        (9.0, 1.0, "Bathroom"),
        (9.0, 7.0, "Bedroom"),
    ]
    # Plot measurement points
    for x, y, name in measurement_points:
        ax.plot(x, y, "wo", markersize=10, markeredgecolor="black", markeredgewidth=1.5)
        ax.text(
            x,
            y + 0.3,
            name,
            ha="center",
            va="bottom",
            color="white",
            fontweight="bold",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("x (m)", fontsize=17)
    ax.set_ylabel("y (m)", fontsize=17)
    ax.set_title("Top candidate placements over floor plan", fontsize=17)
    ax.set_aspect("equal")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved scatter plot to {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--top",
        default="output/top100_annealing.csv",
        help="CSV with top candidates (columns: rank,x,y,score)",
    )
    p.add_argument(
        "--floor", default="output/floor_plan.csv", help="Floor plan CSV (0=wall,1=air)"
    )
    p.add_argument(
        "--out", default="output/top100_scatter.png", help="Output image path"
    )
    args = p.parse_args()

    plot_top_candidates(args.floor, args.top, args.out)

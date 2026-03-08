"""Plot DLA grid-size scaling benchmark results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
})

df = pd.read_csv("output/scaling.csv")
grid_sizes = sorted(df["N"].unique())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Panel 1: Per-step solve time vs step for each N ──────────────────────────
ax = axes[0]
for N in grid_sizes:
    sub = df[df["N"] == N]
    ax.plot(sub["step"], sub["solve_ms"], linewidth=1.5, label=f"N={N}")
ax.set_xlabel("Growth step")
ax.set_ylabel("Solve time (ms)")
ax.set_title("Per-step Solve Time vs Grid Size")
ax.legend()
ax.grid(True, alpha=0.3)

# ── Panel 2: Total solve time vs N (log-log) ────────────────────────────────
ax = axes[1]
totals = df.groupby("N")["solve_ms"].sum()
ax.loglog(totals.index, totals.values, "o-", linewidth=2, markersize=8)
for N, t in totals.items():
    ax.annotate(f"  {t:.0f} ms", (N, t), fontsize=10)
ax.set_xlabel("Grid size N")
ax.set_ylabel("Total solve time (ms)")
ax.set_title("Scaling: Total Solve Time vs N")
ax.grid(True, alpha=0.3, which="both")

fig.suptitle("DLA Grid-Size Scaling", fontsize=16, y=1.02)
fig.tight_layout()
fig.savefig("output/plots/scaling.eps", dpi=150, bbox_inches="tight")
print("saved output/plots/scaling.eps")
plt.close("all")

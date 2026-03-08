#!/usr/bin/env python3
"""Plot sweep CSV results: SOR (eta×omega) and RB-GS comparison."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

OUT_DIR = Path("output/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Read CSVs ────────────────────────────────────────────────────────
def read_sor():
    rows = []
    with open("output/sweep_eta_omega.csv") as f:
        for r in csv.DictReader(f):
            rows.append({
                "eta": float(r["eta"]),
                "omega": float(r["omega"]),
                "iters": int(r["total_iters"]),
                "time": float(r["total_time_ms"]),
            })
    return rows


def read_rbgs():
    rows = []
    with open("output/sweep_rbgs.csv") as f:
        for r in csv.DictReader(f):
            rows.append({
                "eta": float(r["eta"]),
                "iters": int(r["total_iters"]),
                "time": float(r["total_time_ms"]),
            })
    return rows


sor_data = read_sor()
rbgs_data = read_rbgs()

etas = sorted(set(r["eta"] for r in sor_data))
omegas = sorted(set(r["omega"] for r in sor_data))

# Build lookup
lookup_iters = {}
lookup_time = {}
for r in sor_data:
    lookup_iters[(r["eta"], r["omega"])] = r["iters"]
    lookup_time[(r["eta"], r["omega"])] = r["time"]

# ── Plot 1: SOR iterations vs omega, one line per eta ────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for eta in etas:
    iters = [lookup_iters[(eta, w)] for w in omegas]
    ax.plot(omegas, iters, "o-", label=rf"$\eta={eta:.1f}$")

# Theoretical optimal omega for 5-point Laplacian on N×N grid with
# Dirichlet BC: omega* = 2 / (1 + sin(pi/N))
N = 100
omega_star = 2.0 / (1.0 + np.sin(np.pi / N))
ax.axvline(omega_star, color="k", ls="--", lw=1.2, alpha=0.6,
           label=rf"$\omega^*_{{theory}}={omega_star:.4f}$")

ax.set_xlabel(r"$\omega$")
ax.set_ylabel("Total SOR iterations")
ax.set_title(r"SOR iterations vs $\omega$ for different $\eta$")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "sweep_sor_iters_vs_omega.eps", dpi=150)
print(f"saved {OUT_DIR / 'sweep_sor_iters_vs_omega.eps'}")
plt.close(fig)

# ── Plot 2: SOR time vs omega, one line per eta ─────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for eta in etas:
    times = [lookup_time[(eta, w)] for w in omegas]
    ax.plot(omegas, times, "s-", label=rf"$\eta={eta:.1f}$")

ax.axvline(omega_star, color="k", ls="--", lw=1.2, alpha=0.6,
           label=rf"$\omega^*_{{theory}}={omega_star:.4f}$")

ax.set_xlabel(r"$\omega$")
ax.set_ylabel("Total time (ms)")
ax.set_title(r"SOR wall time vs $\omega$ for different $\eta$")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "sweep_sor_time_vs_omega.eps", dpi=150)
print(f"saved {OUT_DIR / 'sweep_sor_time_vs_omega.eps'}")
plt.close(fig)

# ── Plot 3: SOR (best omega) vs RB-GS — iterations ──────────────────
best_sor = {}
for eta in etas:
    candidates = [r for r in sor_data if r["eta"] == eta]
    best_sor[eta] = min(candidates, key=lambda r: r["iters"])

rbgs_by_eta = {r["eta"]: r for r in rbgs_data}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

sor_iters = [best_sor[e]["iters"] for e in etas]
sor_times = [best_sor[e]["time"] for e in etas]
rbgs_iters = [rbgs_by_eta[e]["iters"] for e in etas]
rbgs_times = [rbgs_by_eta[e]["time"] for e in etas]

x = np.arange(len(etas))
w = 0.35

ax1.bar(x - w / 2, sor_iters, w, label="SOR (best ω)")
ax1.bar(x + w / 2, rbgs_iters, w, label="RB-GS")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{e:.1f}" for e in etas])
ax1.set_xlabel(r"$\eta$")
ax1.set_ylabel("Total iterations")
ax1.set_title("Iterations: SOR vs RB-GS")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

ax2.bar(x - w / 2, sor_times, w, label="SOR (best ω)")
ax2.bar(x + w / 2, rbgs_times, w, label="RB-GS")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{e:.1f}" for e in etas])
ax2.set_xlabel(r"$\eta$")
ax2.set_ylabel("Total time (ms)")
ax2.set_title("Wall time: SOR vs RB-GS")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(OUT_DIR / "sweep_sor_vs_rbgs.eps", dpi=150)
print(f"saved {OUT_DIR / 'sweep_sor_vs_rbgs.eps'}")
plt.close(fig)

# ── Plot 4: 3D surface — iterations(eta, omega) ─────────────────────
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402

OMEGA, ETA = np.meshgrid(omegas, etas)
ITERS = np.array([[lookup_iters[(e, w)] for w in omegas] for e in etas])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    OMEGA, ETA, ITERS / 1e3,
    cmap="viridis", edgecolor="k", linewidth=0.3, alpha=0.9,
)
fig.colorbar(surf, ax=ax, shrink=0.55, label="Iterations (×10³)")

ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\eta$")
ax.set_zlabel("Total iterations (×10³)")
ax.set_title(r"SOR iterations over $(\omega, \eta)$ grid")
ax.view_init(elev=25, azim=-50)

fig.tight_layout()
fig.savefig(OUT_DIR / "sweep_3d_iters.eps", dpi=150)
print(f"saved {OUT_DIR / 'sweep_3d_iters.eps'}")
plt.close(fig)

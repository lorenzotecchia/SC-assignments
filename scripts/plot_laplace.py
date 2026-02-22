import logging
import os
import sys

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

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.iterative_methods import SOR, Gauss_Seidel, N, epsilon, max_iters

# ── Load Jacobi data from C++ output ────────────────────────────────
jacobi_grid = np.loadtxt("output/laplace_solution.txt")
jacobi_deltas = np.loadtxt("output/laplace_deltas.txt")
N_jacobi = jacobi_grid.shape[0]

# ── Run Python solvers (same grid size as iterative_methods: N+1 points) ─
print("laplace: compiling numba jit...")
_ = Gauss_Seidel(5, 1e-2, 10)
_ = SOR(1.5, 5, 1e-2, 10)

c_gs, iter_gs, deltas_gs = Gauss_Seidel(N, epsilon, max_iters)
print(f"laplace gauss-seidel: {iter_gs} iters")

c_sor195, iter_sor195, deltas_sor195 = SOR(1.95, N, epsilon, max_iters)
print(f"laplace sor (omega=1.95): {iter_sor195} iters")

c_sor175, iter_sor175, deltas_sor175 = SOR(1.75, N, epsilon, max_iters)
print(f"laplace sor (omega=1.75): {iter_sor175} iters")

# ── Figure 1: Heatmaps side-by-side ─────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))

grids = [jacobi_grid, c_gs.T, c_sor195.T]
titles = [
    f"Jacobi (C++, {N_jacobi}x{N_jacobi})",
    f"Gauss-Seidel ({N+1}x{N+1})",
    f"SOR $\\omega$=1.95 ({N+1}x{N+1})",
]

for ax, grid, title in zip(axes, grids, titles):
    im = ax.imshow(grid, cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
    fig1.colorbar(im, ax=ax, label=r"$c(x,y)$", shrink=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")

fig1.suptitle("Steady-state Laplace solutions", fontsize=14, y=1.02)
fig1.tight_layout()
fig1.savefig("output/plots/laplace_heatmaps.eps", format="eps")

# ── Figure 2: Convergence delta vs iteration k ──────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.semilogy(
    range(1, len(jacobi_deltas) + 1),
    jacobi_deltas,
    label=f"Jacobi (C++, {len(jacobi_deltas)} iters)",
    color="#4CAF50",
)
ax2.semilogy(
    range(1, len(deltas_gs) + 1),
    deltas_gs,
    label=f"Gauss-Seidel ({len(deltas_gs)} iters)",
    color="#2196F3",
)
ax2.semilogy(
    range(1, len(deltas_sor195) + 1),
    deltas_sor195,
    label=f"SOR $\\omega$=1.95 ({len(deltas_sor195)} iters)",
    color="#FF5722",
)
ax2.semilogy(
    range(1, len(deltas_sor175) + 1),
    deltas_sor175,
    label=f"SOR $\\omega$=1.75 ({len(deltas_sor175)} iters)",
    color="#9C27B0",
)

ax2.set_xlabel("Iteration $k$")
ax2.set_ylabel(r"$\delta$ (max absolute change)")
ax2.set_title(r"Convergence: $\delta$ vs iteration $k$")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("output/plots/laplace_convergence.eps", format="eps")

# ── Figure 3: Cross-section vs analytical ────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 5))

# Jacobi: x-averaged profile (C++ grid is N_jacobi x N_jacobi)
y_jacobi = np.linspace(0, 1, N_jacobi)
c_mean_jacobi = np.mean(jacobi_grid, axis=1)

# GS and SOR: x-averaged profile (grid is (N+1) x (N+1))
y_py = np.linspace(0, 1, N + 1)
c_mean_gs = np.mean(c_gs, axis=0)
c_mean_sor = np.mean(c_sor195, axis=0)

ax3.plot(
    y_jacobi, c_mean_jacobi, "s-", ms=3, lw=1.5, label="Jacobi (C++)", color="#4CAF50"
)
ax3.plot(y_py, c_mean_gs, "o-", ms=3, lw=1.5, label="Gauss-Seidel", color="#2196F3")
ax3.plot(
    y_py, c_mean_sor, "^:", ms=3, lw=1.5, label="SOR $\\omega$=1.95", color="#FF5722"
)
ax3.plot(y_py, y_py, "--", lw=1.5, color="black", label="Analytical: $c = y$")

ax3.set_xlabel("$y$")
ax3.set_ylabel("$\\langle c \\rangle_x$")
ax3.set_title("Cross-section: x-averaged concentration vs analytical")
ax3.legend()
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig("output/plots/laplace_crosssection.eps", format="eps")
plt.close('all')

# ── K & L: Object visualisation ──────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

# Load new outputs
sor_base  = np.loadtxt("output/laplace_sor_baseline.txt")
sink_grid = np.loadtxt("output/laplace_sink.txt")
sink_mask = np.loadtxt("output/laplace_sink_mask.txt")
ins_grid  = np.loadtxt("output/laplace_insulator.txt")
ins_mask  = np.loadtxt("output/laplace_insulator_mask.txt")

# omega sweep: columns are [omega, iters_no_obj, iters_sink]
sweep = np.loadtxt("output/laplace_omega_sweep.txt", skiprows=1)
omega_vals, iters_empty, iters_sink = sweep[:, 0], sweep[:, 1], sweep[:, 2]

N = sink_grid.shape[0]

# ── Figure K-1: Concentration heatmaps (baseline vs sinks vs insulator) ──────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
datasets = [
    (sor_base, np.zeros((N, N)), "SOR baseline (no objects)"),
    (sink_grid, sink_mask,       "SOR with sinks (K)"),
    (ins_grid,  ins_mask,        "SOR with insulator (L)"),
]

# INSULATOR cells (mask==2) are never updated by the solver – their stored c=0
# is just the zero-initialisation, NOT a physical value.  Mask them as NaN so
# they render as gray and don't masquerade as sinks.
# SINK cells (mask==1) ARE physically at c=0 (Dirichlet BC): show their true
# value so the contour lines flow visibly into them.
INSULATOR = 2
cmap_obj = plt.cm.viridis.copy()
cmap_obj.set_bad(color="#808080", alpha=1.0)   # gray for insulator interior

x_coords = np.linspace(0, 1, N)
y_coords = np.linspace(0, 1, N)
field_levels = np.linspace(0.05, 0.95, 10)  # iso-concentration lines

for ax, (grid, mask, title) in zip(axes, datasets):
    # Only mask insulator cells; sink c=0 is real and should be visible.
    display_grid = np.where(mask == INSULATOR, np.nan, grid)

    im = ax.imshow(display_grid, cmap=cmap_obj, origin="lower",
                   extent=[0, 1, 0, 1], vmin=0, vmax=1)

    # Contour lines: reveal field bending around insulator and flux into sinks.
    ax.contour(x_coords, y_coords, display_grid, levels=field_levels,
               colors="white", alpha=0.55, linewidths=0.7)

    # White outline around each object region
    if mask.any():
        ax.contour(x_coords, y_coords, mask, levels=[0.5],
                   colors="white", linewidths=1.5)

    fig.colorbar(im, ax=ax, label=r"$c(x,y)$", shrink=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
    ax.set_aspect("equal")
fig.suptitle("Laplace steady-state solutions with objects", fontsize=13, y=1.02)
fig.tight_layout()
plt.savefig("output/plots/laplace_objects_heatmaps.eps", format="eps")
plt.close('all')

# ── Figure K-2: Omega sweep ───────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(omega_vals, iters_empty, "o-", label="No objects", color="#2196F3")
ax2.plot(omega_vals, iters_sink,  "s-", label="With sinks", color="#FF5722")
ax2.set_xlabel(r"$\omega$")
ax2.set_ylabel("Iterations to convergence")
ax2.set_title(r"Effect of $\omega$ on convergence (K)")
ax2.legend(); ax2.grid(True, alpha=0.3)
fig2.tight_layout()
plt.savefig("output/plots/laplace_omega_sweep.eps", format="eps")
plt.close('all')

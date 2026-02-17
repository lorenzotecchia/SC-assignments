import sys, os
import numpy as np
import matplotlib.pyplot as plt

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.iterative_methods import Gauss_Seidel, SOR, N, epsilon, max_iters

# ── Load Jacobi data from C++ output ────────────────────────────────
jacobi_grid = np.loadtxt("output/laplace_solution.txt")
jacobi_deltas = np.loadtxt("output/laplace_deltas.txt")
N_jacobi = jacobi_grid.shape[0]

# ── Run Python solvers (same grid size as iterative_methods: N+1 points) ─
print("Compiling Numba JIT functions...")
_ = Gauss_Seidel(5, 1e-2, 10)
_ = SOR(1.5, 5, 1e-2, 10)
print("Done.\n")

c_gs, iter_gs, deltas_gs = Gauss_Seidel(N, epsilon, max_iters)
print(f"Gauss-Seidel converged in {iter_gs} iterations")

c_sor195, iter_sor195, deltas_sor195 = SOR(1.95, N, epsilon, max_iters)
print(f"SOR (omega=1.95) converged in {iter_sor195} iterations")

c_sor175, iter_sor175, deltas_sor175 = SOR(1.75, N, epsilon, max_iters)
print(f"SOR (omega=1.75) converged in {iter_sor175} iterations")

# ── Figure 1: Heatmaps side-by-side ─────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))

grids = [jacobi_grid, c_gs, c_sor195]
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

# ── Figure 2: Convergence delta vs iteration k ──────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.semilogy(range(1, len(jacobi_deltas) + 1), jacobi_deltas,
             label=f"Jacobi (C++, {len(jacobi_deltas)} iters)", color="#4CAF50")
ax2.semilogy(range(1, len(deltas_gs) + 1), deltas_gs,
             label=f"Gauss-Seidel ({len(deltas_gs)} iters)", color="#2196F3")
ax2.semilogy(range(1, len(deltas_sor195) + 1), deltas_sor195,
             label=f"SOR $\\omega$=1.95 ({len(deltas_sor195)} iters)", color="#FF5722")
ax2.semilogy(range(1, len(deltas_sor175) + 1), deltas_sor175,
             label=f"SOR $\\omega$=1.75 ({len(deltas_sor175)} iters)", color="#9C27B0")

ax2.set_xlabel("Iteration $k$")
ax2.set_ylabel(r"$\delta$ (max absolute change)")
ax2.set_title(r"Convergence: $\delta$ vs iteration $k$")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

# ── Figure 3: Cross-section vs analytical ────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 5))

# Jacobi: x-averaged profile (C++ grid is N_jacobi x N_jacobi)
y_jacobi = np.linspace(0, 1, N_jacobi)
c_mean_jacobi = np.mean(jacobi_grid, axis=1)

# GS and SOR: x-averaged profile (grid is (N+1) x (N+1))
y_py = np.linspace(0, 1, N + 1)
c_mean_gs = np.mean(c_gs, axis=0)
c_mean_sor = np.mean(c_sor195, axis=0)

ax3.plot(y_jacobi, c_mean_jacobi, "s-", ms=3, lw=1.5,
         label="Jacobi (C++)", color="#4CAF50")
ax3.plot(y_py, c_mean_gs, "o-", ms=3, lw=1.5,
         label="Gauss-Seidel", color="#2196F3")
ax3.plot(y_py, c_mean_sor, "^:", ms=3, lw=1.5,
         label="SOR $\\omega$=1.95", color="#FF5722")
ax3.plot(y_py, y_py, "--", lw=1.5, color="black", label="Analytical: $c = y$")

ax3.set_xlabel("$y$")
ax3.set_ylabel("$\\langle c \\rangle_x$")
ax3.set_title("Cross-section: x-averaged concentration vs analytical")
ax3.legend()
ax3.grid(True, alpha=0.3)
fig3.tight_layout()

plt.show()

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("output/laplace_solution.txt")
N = data.shape[0]

x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

# ── 2D heatmap ────────────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(6, 5))
im = ax1.pcolormesh(x, y, data, shading="auto", cmap="hot")
fig1.colorbar(im, ax=ax1, label=r"$c(x, y)$")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Steady-state solution (Jacobi)")
ax1.set_aspect("equal")
fig1.tight_layout()

# ── Cross-section: c vs y at fixed x (should be linear) ──────────────

fig2, ax2 = plt.subplots(figsize=(6, 4))
mid_col = N // 2
ax2.plot(y, data[:, mid_col], "o-", ms=3, lw=1.5, label="Jacobi solution")
ax2.plot(y, y, "--", lw=1, color="red", label="Analytical: $c = y$")
ax2.set_xlabel("y")
ax2.set_ylabel(r"$c(x, y)$")
ax2.set_title(f"Cross-section at x = {x[mid_col]:.2f}")
ax2.legend()
fig2.tight_layout()

plt.show()

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

max_iters = 10000
epsilon = 1e-5
N = 50


def Gauss_Seidel(N, epsilon, max_iters):
    # grid with N+1 points
    c_grid = np.zeros((N + 1, N + 1))

    # boundaries for y
    c_grid[:, 0] = 0
    c_grid[:, N] = 1

    deltas = []

    for iter in range(max_iters):
        delta = 0

        for j in range(1, N):
            # periodic boundaries for x
            c_plus = np.roll(c_grid, 1, axis=0)
            c_minus = np.roll(c_grid, -1, axis=0)

            for i in range(N + 1):

                # remember old c for stopping criteria
                old_c = c_grid[i, j]

                c_grid[i, j] = 0.25 * (
                    c_plus[i, j] + c_minus[i, j] + c_grid[i, j + 1] + c_grid[i, j - 1]
                )

                # delta is max distance between new and old c
                delta = max(delta, np.abs(c_grid[i, j] - old_c))

        deltas.append(delta)

        # stopping criteria
        if delta < epsilon:
            break

    return c_grid, iter, deltas


def SOR(omega, N, epsilon, max_iters):
    # grid with N+1 points
    c_grid = np.zeros((N + 1, N + 1))

    # boundaries for y
    c_grid[:, 0] = 0
    c_grid[:, N] = 1

    deltas = []

    for iter in range(max_iters):
        delta = 0

        for j in range(1, N):
            # periodic boundaries for x
            c_plus = np.roll(c_grid, 1, axis=0)
            c_minus = np.roll(c_grid, -1, axis=0)

            for i in range(N + 1):

                # remember old c for stopping criteria
                old_c = c_grid[i, j]

                c_grid[i, j] = (
                    omega
                    * 0.25
                    * (
                        c_plus[i, j]
                        + c_minus[i, j]
                        + c_grid[i, j + 1]
                        + c_grid[i, j - 1]
                    )
                    + (1 - omega) * old_c
                )

                # delta is max distance between new and old c
                delta = max(delta, np.abs(c_grid[i, j] - old_c))

        deltas.append(delta)

        # stopping criteria
        if delta < epsilon:
            break

    return c_grid, iter, deltas


def optimal_omega(omega_min, omega_max, step):
    """Find optimal omega that minimizes number of iterations for different N using SOR"""

    # store optimal omegas and corresponding num of itretations for each N
    results = []

    for N in range(50, 250, 50):

        omegas = np.arange(omega_min, omega_max, step)
        iterations = np.zeros(len(omegas))

        for i, omega in enumerate(omegas):
            _, iter, _ = SOR(omega, N, epsilon, max_iters)
            iterations[i] = iter

        # index of smallest iteration
        opt_index = np.argmin(iterations)

        results.append(
            {"N": N, "omega": omegas[opt_index], "iterations": iterations[opt_index]}
        )

    return results


def compare_to_analytical(c_gauss, c_SOR):
    """Compare Gauss-Seidel and SOR solutions to analytical solutions"""

    # mean
    c_mean_gauss = np.mean(c_gauss, axis=0)
    c_mean_SOR = np.mean(c_SOR, axis=0)

    y = np.linspace(0, 1, N + 1)

    # mean absolute error
    mae_gauss = np.mean(np.abs(y - c_mean_gauss))
    mae_SOR = np.mean(np.abs(y - c_mean_SOR))

    # plot Gauss-Seidel, SOR, and analytical
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y, c_mean_gauss, "--", label="Gauss-Seidel")
    ax.plot(y, c_mean_SOR, ":", label="SOR")
    ax.plot(y, y, label="Analytical")
    ax.set_xlabel("$y$")
    ax.set_ylabel("Concentration")
    ax.set_title("Comparing Numerical and Analytical Solution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("output/plots/y_vs_conc.eps", format="eps")
    plt.close()

    return mae_gauss, mae_SOR


# ---- Numba implementations (modular indexing, no np.roll) ----


@njit
def Gauss_Seidel_numba(N, epsilon, max_iters):
    c_grid = np.zeros((N + 1, N + 1))
    c_grid[:, 0] = 0   # y=0: c=0
    c_grid[:, N] = 1   # y=1: c=1
    deltas = []
    for iter in range(max_iters):
        delta = 0.0
        for j in range(1, N):           # iterate over interior y
            for i in range(N + 1):      # iterate over all x
                i_plus = (i + 1) % (N + 1)   # periodic in x
                i_minus = (i - 1) % (N + 1)  # periodic in x
                old_c = c_grid[i, j]
                c_grid[i, j] = 0.25 * (
                    c_grid[i_plus, j]
                    + c_grid[i_minus, j]
                    + c_grid[i, j + 1]
                    + c_grid[i, j - 1]
                )
                delta = max(delta, abs(c_grid[i, j] - old_c))
        deltas.append(delta)
        if delta < epsilon:
            break
    return c_grid, iter, deltas


@njit
def SOR_numba(omega, N, epsilon, max_iters):
    c_grid = np.zeros((N + 1, N + 1))
    c_grid[:, 0] = 0   # y=0: c=0
    c_grid[:, N] = 1   # y=1: c=1
    deltas = []
    for iter in range(max_iters):
        delta = 0.0
        for j in range(1, N):           # iterate over interior y
            for i in range(N + 1):      # iterate over all x
                i_plus = (i + 1) % (N + 1)   # periodic in x
                i_minus = (i - 1) % (N + 1)  # periodic in x
                old_c = c_grid[i, j]
                c_grid[i, j] = (
                    omega
                    * 0.25
                    * (
                        c_grid[i_plus, j]
                        + c_grid[i_minus, j]
                        + c_grid[i, j + 1]
                        + c_grid[i, j - 1]
                    )
                    + (1 - omega) * old_c
                )
                delta = max(delta, abs(c_grid[i, j] - old_c))
        deltas.append(delta)
        if delta < epsilon:
            break
    return c_grid, iter, deltas


def optimal_omega_numba(omega_min, omega_max, step):
    """Find optimal omega that minimizes number of iterations for different N using SOR"""

    # store optimal omegas and corresponding num of itretations for each N
    results = []

    for N in range(50, 250, 50):

        omegas = np.arange(omega_min, omega_max, step)
        iterations = np.zeros(len(omegas))

        for i, omega in enumerate(omegas):
            _, iter, _ = SOR_numba(omega, N, epsilon, max_iters)
            iterations[i] = iter

        # index of smallest iteration
        opt_index = np.argmin(iterations)

        results.append(
            {"N": N, "omega": omegas[opt_index], "iterations": iterations[opt_index]}
        )

    return results


# ----------numba output -----------


if __name__ == "__main__":
    # ----------Gauss-Seidel -----------

    print("iterative: gauss-seidel...")
    c_gauss, iter_gauss, deltas_gauss = Gauss_Seidel_numba(N, epsilon, max_iters)
    print(f"iterative gauss-seidel: {iter_gauss} iters")

    plt.figure(figsize=(6, 5))
    plt.imshow(c_gauss)
    plt.colorbar(label="Concentration")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Numerical Solution using Gauss-Seidel")
    plt.savefig("output/plots/numerical_sol_gauss", format="eps")
    plt.close()

    print("iterative: sor...")
    omega = 1.95
    c_SOR, iter_SOR, deltas_SOR = SOR_numba(omega, N, epsilon, max_iters)
    print(f"iterative sor (omega={omega}): {iter_SOR} iters")

    plt.figure(figsize=(6, 5))
    plt.imshow(c_SOR, cmap="Purples")
    plt.colorbar(label="Concentration")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Numerical Solution using SOR")
    plt.savefig("output/plots/sol_sor.eps", format="eps")
    plt.close()

    omega_min = 1.7
    omega_max = 2
    step = 0.1
    results = optimal_omega_numba(omega_min, omega_max, step)
    for r in results:
        print(f"iterative optimal omega: N={r['N']}, omega={r['omega']}, iters={r['iterations']}")

    mae_gauss, mae_SOR = compare_to_analytical(c_gauss, c_SOR)
    print(f"iterative mae gauss-seidel: {mae_gauss:.6f}")
    print(f"iterative mae sor:          {mae_SOR:.6f}")

    omega = 1.75
    _, _, deltas_SOR_175 = SOR_numba(omega, N, epsilon, max_iters)

    fig, ax = plt.subplots(figsize=(8, 5))

    try:
        jacobi_deltas = np.loadtxt("output/laplace_deltas.txt")
        ax.semilogy(
            range(1, len(jacobi_deltas) + 1),
            jacobi_deltas,
            label=f"Jacobi (C++, {len(jacobi_deltas)} iters)",
            color="#4CAF50",
        )
    except FileNotFoundError:
        pass

    ax.semilogy(
        range(1, len(deltas_gauss) + 1),
        deltas_gauss,
        label=f"Gauss-Seidel ({len(deltas_gauss)} iters)",
        color="#2196F3",
    )
    ax.semilogy(
        range(1, len(deltas_SOR) + 1),
        deltas_SOR,
        "--",
        label=r"SOR $\omega$=1.95" + f" ({len(deltas_SOR)} iters)",
        color="#FF5722",
    )
    ax.semilogy(
        range(1, len(deltas_SOR_175) + 1),
        deltas_SOR_175,
        ":",
        label=r"SOR $\omega$=1.75" + f" ({len(deltas_SOR_175)} iters)",
        color="#9C27B0",
    )
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$\delta$ (max absolute change)")
    ax.set_title(r"Convergence: $\delta$ vs iteration $k$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("output/plots/convergence.eps", format="eps")
    plt.close()



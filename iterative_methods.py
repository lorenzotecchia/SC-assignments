import numpy as np
import matplotlib.pyplot as plt
from numba import njit


max_iters = 10000
epsilon = 1e-5
N = 50


def Gauss_Seidel(N, epsilon, max_iters):
    # grid with N+1 points
    c_grid = np.zeros((N+1, N+1))

    # boundaries for y
    c_grid[:,0] = 0
    c_grid[:,N] = 1

    deltas = []

    for iter in range(max_iters):
        delta = 0

        for j in range(1, N):
            # periodic boundaries for x
            c_plus = np.roll(c_grid, 1, axis=0)
            c_minus = np.roll(c_grid, -1, axis=0)

            for i in range(N+1):

                # remember old c for stopping criteria
                old_c = c_grid[i, j]

                c_grid[i, j] = 0.25 * (c_plus[i, j] + c_minus[i, j] + c_grid[i, j+1] + c_grid[i, j-1])

                # delta is max distance between new and old c
                delta = max(delta, np.abs(c_grid[i, j] - old_c))
            
        deltas.append(delta)

        # stopping criteria
        if delta < epsilon:
            break
    
    return c_grid, iter, deltas

def SOR(omega, N, epsilon, max_iters):
    # grid with N+1 points
    c_grid = np.zeros((N+1, N+1))

    # boundaries for y
    c_grid[:,0] = 0
    c_grid[:,N] = 1

    deltas = []

    for iter in range(max_iters):
        delta = 0

        for j in range(1, N):
            # periodic boundaries for x
            c_plus = np.roll(c_grid, 1, axis=0)
            c_minus = np.roll(c_grid, -1, axis=0)

            for i in range(N+1):

                # remember old c for stopping criteria
                old_c = c_grid[i, j]

                c_grid[i, j] = omega * 0.25 * (c_plus[i, j] + c_minus[i, j] + c_grid[i, j+1] + c_grid[i, j-1]) + (1 - omega) * old_c

                # delta is max distance between new and old c
                delta = max(delta, np.abs(c_grid[i, j] - old_c))

        deltas.append(delta)

        # stopping criteria
        if delta < epsilon:
            break
    
    return c_grid, iter, deltas

def optimal_omega(omega_min, omega_max, step):
    '''Find optimal omega that minimizes number of iterations for different N using SOR'''

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
        
        results.append({
            "N": N,
            "omega": omegas[opt_index],
            "iterations": iterations[opt_index]
        })

    return results


def compare_to_analytical(c_gauss, c_SOR):
    '''Compare Gauss-Seidel and SOR solutions to analytical solutions'''

    # mean
    c_mean_gauss = np.mean(c_gauss, axis=0)
    c_mean_SOR = np.mean(c_SOR, axis=0)

    y = np.linspace(0, 1, N+1)
    
    # mean absolute error
    mae_gauss = np.mean(np.abs(y - c_mean_gauss))
    mae_SOR = np.mean(np.abs(y - c_mean_SOR))
    
    # plot Gauss-Seidel, SOR, and analytical
    plt.plot(y, c_mean_gauss, "--", label="Gauss-Seidel")
    plt.plot(y, c_mean_SOR, ":", label="SOR")
    plt.plot(y, y, label="analytical")
    
    plt.legend()
    plt.xlabel("y")
    plt.ylabel("concentration")
    plt.title('Comparing Numerical and Analytical Solution')
    plt.show()

    return mae_gauss, mae_SOR

# ---- Numba implementations (modular indexing, no np.roll) ----

@njit
def Gauss_Seidel_numba(N, epsilon, max_iters):
    c_grid = np.zeros((N+1, N+1))
    c_grid[0,:] = 1
    c_grid[N,:] = 0
    deltas = []
    for iter in range(max_iters):
        delta = 0.0
        for i in range(1, N):
            for j in range(N+1):
                i_plus = (i + 1) % (N + 1)
                i_minus = (i - 1) % (N + 1)
                old_c = c_grid[i, j]
                c_grid[i, j] = 0.25 * (c_grid[i_plus, j] + c_grid[i_minus, j] + c_grid[i, j+1] + c_grid[i, j-1])
                delta = max(delta, abs(c_grid[i, j] - old_c))
        deltas.append(delta)
        if delta < epsilon:
            break
    return c_grid, iter, deltas


@njit
def SOR_numba(omega, N, epsilon, max_iters):
    c_grid = np.zeros((N+1, N+1))
    c_grid[0,:] = 1
    c_grid[N,:] = 0
    deltas = []
    for iter in range(max_iters):
        delta = 0.0
        for i in range(1, N):
            for j in range(N+1):
                i_plus = (i + 1) % (N + 1)
                i_minus = (i - 1) % (N + 1)
                old_c = c_grid[i, j]
                c_grid[i, j] = omega * 0.25 * (c_grid[i_plus, j] + c_grid[i_minus, j] + c_grid[i, j+1] + c_grid[i, j-1]) + (1 - omega) * old_c
                delta = max(delta, abs(c_grid[i, j] - old_c))
        deltas.append(delta)
        if delta < epsilon:
            break
    return c_grid, iter, deltas



def optimal_omega_numba(omega_min, omega_max, step):
    '''Find optimal omega that minimizes number of iterations for different N using SOR'''

    # store optimal omegas and corresponding num of itretations for each N
    results = []

    for N in range(50, 1050, 100):

        omegas = np.arange(omega_min, omega_max, step)
        iterations = np.zeros(len(omegas))

        for i, omega in enumerate(omegas):
            _, iter, _ = SOR_numba(omega, N, epsilon, max_iters)
            iterations[i] = iter

        # index of smallest iteration
        opt_index = np.argmin(iterations)
        
        results.append({
            "N": N,
            "omega": omegas[opt_index],
            "iterations": iterations[opt_index]
        })

    return results


#----------numba output -----------


#----------Gauss-Seidel -----------

print("\n########## H ##########\n")
c_gauss, iter_gauss, deltas_gauss = Gauss_Seidel_numba(N, epsilon, max_iters)
print("\n-------- Gauss-Seidel --------\n")
print("Solution using Gauss-Seidel\n", c_gauss)
print("\nNumber of iterations for Gauss-Seidel: ", iter_gauss)

# heatmap
plt.figure(figsize=(6,5))
plt.imshow(c_gauss)
plt.colorbar(label='Concentration')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution using Gauss-Seidel')
plt.show()

#-------------- SOR -------------
omega = 1.95
c_SOR, iter_SOR , deltas_SOR= SOR_numba(omega, N, epsilon, max_iters)
print("\n-------- SOR --------\n")
print("Solution using SOR\n", c_SOR )
print("\nNumber of iterations for SOR: ", iter_SOR)

# heatmap
plt.figure(figsize=(6,5))
plt.imshow(c_SOR, cmap='Purples')
plt.colorbar(label='Concentration')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution using SOR')
plt.show()

#-------------- optimal omega ------------
omega_min = 1.7
omega_max = 2
step = 0.1
results = optimal_omega_numba(omega_min, omega_max, step)
print("\n-------- Find optimal omega for SOR --------\n")
for r in results:
    print(f"For N={r["N"]}: optimal omega = {r["omega"]} with {r["iterations"]} iterations")


#-----------analytical comparison ------------

print("\n-------- Compare to analytical results --------\n")
mae_gauss, mae_SOR = compare_to_analytical(c_SOR, c_gauss)
print("Mean absolute error of Gauss-Seidel: ", mae_gauss)
print("Mean absolute error of SOR: ", mae_SOR)


print("\n########## I ##########\n")

# Show how delta depends on the number of iterations k

omega = 1.75
_, _, deltas_SOR_175= SOR_numba(omega, N, epsilon, max_iters)

plt.semilogx(deltas_gauss, label="Gauss-Seidel")
plt.semilogx(deltas_SOR, "--",label="SOR $\omega$=1.95")
plt.semilogx(deltas_SOR_175, ":", label="SOR $\omega$=1.75")
plt.legend()
plt.xlabel("Iterations k")
plt.ylabel("$\delta$")
plt.title("Behaviour of convergence measure $\delta$")
plt.show()





'''
print("\n########## H ##########\n")
c_gauss, iter_gauss, deltas_gauss = Gauss_Seidel(N, epsilon, max_iters)
print("\n-------- Gauss-Seidel --------\n")
print("Solution using Gauss-Seidel\n", c_gauss)
print("\nNumber of iterations for Gauss-Seidel: ", iter_gauss)

# heatmap
plt.figure(figsize=(6,5))
plt.imshow(c_gauss, cmap='RdPu')
plt.colorbar(label='Concentration')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution using Gauss-Seidel')
plt.show()


omega = 1.95
c_SOR, iter_SOR , deltas_SOR= SOR(omega, N, epsilon, max_iters)
print("\n-------- SOR --------\n")
print("Solution using SOR\n", c_SOR )
print("\nNumber of iterations for SOR: ", iter_SOR)

# heatmap
plt.figure(figsize=(6,5))
plt.imshow(c_SOR, cmap='RdPu')
plt.colorbar(label='Concentration')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Numerical Solution using SOR')
plt.show()


omega_min = 1.7
omega_max = 2
step = 0.1
results = optimal_omega(omega_min, omega_max, step)
print("\n-------- Find optimal omega for SOR --------\n")
for r in results:
    print(f"For N={r["N"]}: optimal omega = {r["omega"]} with {r["iterations"]} iterations")


print("\n-------- Compare to analytical results --------\n")
mae_gauss, mae_SOR = compare_to_analytical(c_SOR, c_gauss)
print("Mean absolute error of Gauss-Seidel: ", mae_gauss)
print("Mean absolute error of SOR: ", mae_SOR)


print("\n########## I ##########\n")

# Show how delta depends on the number of iterations k

omega = 1.75
_, _, deltas_SOR_175= SOR(omega, N, epsilon, max_iters)

plt.semilogx(deltas_gauss, label="Gauss-Seidel", color="royalblue")
plt.semilogx(deltas_SOR, label="SOR $\omega$=1.95", color="hotpink")
plt.semilogx(deltas_SOR_175, label="SOR $\omega$=1.75", color="purple")
plt.legend()
plt.xlabel("Iterations k")
plt.ylabel("$\delta$")
plt.title("Behaviour of convergence measure $\delta$")
plt.show()
'''

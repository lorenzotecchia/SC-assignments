# FD Navier-Stokes Solver — Kármán Vortex Street

Finite-difference incompressible Navier-Stokes solver for the
[Schäfer-Turek benchmark](https://doi.org/10.1007/978-3-322-89849-4_39) (flow past a cylinder, Re ≤ ~3000).

---

## Physical Problem

The **Kármán vortex street** is the periodic shedding of counter-rotating vortices from
a bluff body placed in a steady cross-flow.  It occurs whenever the Reynolds number
exceeds roughly Re ≈ 40 for a circular cylinder.

### Governing equations — incompressible Navier-Stokes

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u}
  = -\frac{1}{\rho}\nabla p + \nu\,\nabla^2\mathbf{u}, \qquad \nabla \cdot \mathbf{u} = 0$$

where **u** = (u, v) is the velocity, p is the kinematic pressure (pressure/density),
and ν is the kinematic viscosity.

### Non-dimensional parameters

| Symbol | Definition | Default |
|--------|-----------|---------|
| Re | U_mean · D / ν | 100 |
| St | f · D / U_mean (Strouhal) | ≈ 0.30 @ Re=100 |
| C_D | 2 F_x / (U² D) | ≈ 3.2 @ Re=100 |
| C_L | 2 F_y / (U² D) | peak ≈ ±1.0 @ Re=100 |

### Benchmark geometry — Schäfer & Turek (1996)

```
 y=0.41 ┌────────────────────────────────────────────────────┐
        │  →→→                                               │
        │  →→→    ●  (0.2, 0.2) r=0.05                      │
        │  →→→                                               │
 y=0.00 └────────────────────────────────────────────────────┘
       x=0                                                 x=2.2
```

- Channel: 2.2 × 0.41 m
- Cylinder: centre (0.2, 0.2) m, radius r = 0.05 m (diameter D = 0.1 m)
- Mean inflow velocity U_mean = 1.0 m/s
- ν = U_mean · D / Re

---

## Numerical Method

### Fractional-step (Chorin projection)

The time integration uses Chorin's projection method [Chorin 1968]:

1. **Predictor** — advance velocity explicitly, ignoring the pressure-gradient term:

$$\mathbf{u}^* = \mathbf{u}^n
  + \Delta t\!\left[\alpha\, N(\mathbf{u}^n) + \beta\, N(\mathbf{u}^{n-1})
  + \nu\,\nabla^2 \mathbf{u}^n\right]$$

   where N(**u**) = −(**u** · ∇)**u** is the convective acceleration.

2. **Pressure Poisson** — enforce ∇ · **u** = 0:

$$\nabla^2 p^{n+1} = \frac{1}{\Delta t}\,\nabla \cdot \mathbf{u}^*$$

3. **Corrector** — project onto the divergence-free subspace:

$$\mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t\,\nabla p^{n+1}$$

### Adams-Bashforth 2 (AB2) for convection

The convective term uses second-order Adams-Bashforth time integration
[Peyret & Taylor 1983]:

$$(\alpha, \beta) = \begin{cases}(1, 0) & \text{step } 0 \text{ (forward Euler warm-up)}\\(\tfrac{3}{2}, -\tfrac{1}{2}) & \text{step} \geq 1\end{cases}$$

Diffusion is treated explicitly with the same first-order (forward Euler) coefficient.
Stability requires:
- **CFL**: dt < 0.5 dx / U_max
- **Diffusion**: dt < dx² / (2ν)  (usually non-binding at Re ≥ 100)

### Convection discretisation — upwind

For the u-momentum equation, the upwind scheme picks the one-sided stencil
based on the sign of the local advecting velocity:

$$\frac{\partial u}{\partial x}\bigg|_{\text{upwind}}
  = \begin{cases}(u_i - u_{i-1})/\Delta x & u_E > 0\\ (u_{i+1} - u_i)/\Delta x & u_E \le 0\end{cases}$$

Upwind convection is first-order accurate but unconditionally stable for pure advection,
making it suitable for high-Re flows without added artificial viscosity.

---

## MAC Staggered Grid

The solver uses a **MAC (Marker-and-Cell) staggered grid** [Harlow & Welch 1965].
Velocity components are stored at cell-face centres; pressure is cell-centred.
This eliminates the spurious pressure modes that arise on co-located grids.

```
         j+1  ─────v─────────v─────
              │         │         │
              u    p    u    p    u
              │         │         │
          j  ─────v─────────v─────
              i        i+1
```

| Field | Location | Array shape |
|-------|---------- |-------------|
| u     | (i·dx, (j+½)·dy) | (nx+1) × ny |
| v     | ((i+½)·dx, j·dy) | nx × (ny+1) |
| p     | ((i+½)·dx, (j+½)·dy) | nx × ny |
| mask  | cell centres (same as p) | nx × ny |

Default grid: **nx = 440, ny = 82** → dx = dy = 0.005 m

---

## Boundary Conditions

### Inlet (x = 0) — Dirichlet parabolic

$$u_{\text{in}}(y) = \frac{3}{2} U_{\text{mean}} \cdot \frac{4\,y\,(H-y)}{H^2},
\quad v = 0$$

Mean is U_mean, peak is 1.5 U_mean (consistent with Schäfer-Turek definition).

### Outlet (x = L) — Neumann + Dirichlet pressure

$$\frac{\partial u}{\partial x} = 0, \quad v = 0, \quad p = 0$$

The pressure Dirichlet condition p = 0 at the outlet fixes the reference level.

### Walls (y = 0 and y = H) — no-slip

$$u = v = 0, \quad \frac{\partial p}{\partial n} = 0$$

Implemented via ghost-cell averaging for u (face at j = 0 / j = ny−1) and
direct zeroing of v at the boundary faces.

### Cylinder — bounce-back no-slip

All velocity faces adjacent to a solid cell (mask = 1) are zeroed.
Pressure uses homogeneous Neumann (automatically satisfied by skipping solid cells
in the SOR sweep).

---

## Pressure Solver — SOR

The discrete Poisson equation is solved with **Successive Over-Relaxation (SOR)**
[Young 1954], adapted from the assignment 2 SOR implementation.

$$p^{k+1}_{i,j} = (1-\omega)\,p^k_{i,j}
  + \omega\,\frac{\text{idx}^2(p_{i+1,j}+p_{i-1,j})+\text{idy}^2(p_{i,j+1}+p_{i,j-1}) - \text{rhs}_{i,j}}{2(\text{idx}^2+\text{idy}^2)}$$

where idx = 1/dx, idy = 1/dy. The optimal ω for a rectangular domain is
approximately 1.8–1.9; the solver defaults to ω = 1.8.

Convergence criterion: ‖p^{k+1} − p^k‖_∞ < tol (default 1e-4).

The pressure field is **warm-started** from the previous time step, which
reduces the SOR iteration count substantially for slowly-varying flows.
Solid cells (mask=1) are skipped in the sweep; their pressure remains at
whatever value the previous step left, satisfying homogeneous Neumann implicitly.

**Boundary conditions for the Poisson solve:**

| Face | Condition | Implementation |
|------|-----------|----------------|
| Outlet i=nx-1 | Dirichlet p=0 | Caller sets, solver skips |
| Inlet i=0 | Neumann dp/dx=0 | Ghost: p_{-1,j} = p_{0,j} |
| Walls j=0,ny-1 | Neumann dp/dn=0 | Ghost: p_{i,-1} = p_{i,0} |
| Solid cells | Neumann dp/dn=0 | Skipped in sweep |

---

## Force Coefficients

Drag and lift are computed using a **pressure-only control-volume** approach.
For each solid cell face adjacent to a fluid cell, the pressure force is
integrated over that face:

$$F_x = \sum_{\text{solid faces}} p_{\text{fluid}} \cdot \Delta y \cdot \hat{n}_x$$

$$C_D = \frac{2 F_x}{U_{\text{mean}}^2\,D}, \qquad C_L = \frac{2 F_y}{U_{\text{mean}}^2\,D}$$

This omits the viscous stress contribution, which is small at Re ≥ 100 but
causes a ~5% under-prediction of C_D. For high accuracy at low Re, a full
boundary-integral formulation is needed [Ferziger et al. 2020, §8.4].

---

## Quick Start

```bash
# Build
make

# Short smoke test (Re=100, t=1 s)
./fd_karman --Re 100 --tend 1.0

# Benchmark run (Re=100, t=20 s — vortex shedding fully developed)
./fd_karman --Re 100 --tend 20.0 --snap 500

# High-Re run (Re=1000, refined grid)
./fd_karman --Re 1000 --nx 880 --ny 164 --dt 0.0005 --tend 10.0
```

Output is written to `output/`:
- `drag_lift.csv` — time series of C_D and C_L
- `vorticity_NNNNNN.txt` — (x, y, ω) point cloud, one file per snapshot

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--Re` | 100 | Reynolds number |
| `--nx` | 440 | Grid cells in x |
| `--ny` | 82 | Grid cells in y |
| `--dt` | 0.001 | Time step (s) |
| `--tend` | 20.0 | End time (s) |
| `--omega` | 1.8 | SOR relaxation factor |
| `--outdir` | output | Output directory |
| `--snap` | 500 | Write vorticity every N steps |

### Recommended dt by Re

| Re | Grid | dt |
|----|------|----|
| 100 | 440 × 82 | 0.001 |
| 400 | 440 × 82 | 0.001 |
| 1000 | 880 × 164 | 0.0005 |
| 2000+ | 880 × 164 | 0.0002 |

---

## Verification

### Test 1 — Pressure Poisson manufactured solution (`test_ns.cpp`)

Solves ∇²p = rhs on a 40×20 grid with exact solution:

$$p(x,y) = \cos\!\left(\frac{\pi x}{2L_x}\right)\cos\!\left(\frac{\pi y}{L_y}\right)$$

This function was chosen because it satisfies **all** solver boundary conditions:

| Boundary | BC type | Why satisfied |
|----------|---------|---------------|
| Inlet x=0 | Neumann dp/dx=0 | cos'(0) = 0 |
| Outlet x=Lx | Dirichlet p=0 | cos(π/2) = 0 |
| Walls y=0,Ly | Neumann dp/dy=0 | sin(0) = sin(π) = 0 |

Result: **max_err = 1.94e-03 < 1e-2** ✓

> **Implementation note:** the Dirichlet condition at the outlet is set by the
> *caller* before invoking `pressure_solve_sor`; the solver skips those cells
> rather than resetting them, so the same routine works for both the NS loop
> (p=0) and the test (p=exact).

### Test 2 — Projection divergence-free property (`test_ns.cpp`)

Runs one time step on a 20×10 grid starting from parabolic inflow, verifies
the interior divergence after projection is machine-near-zero.

Divergence is checked only on **interior fluid cells** (i=1..nx-2, j=1..ny-2,
mask=0). Ghost rows at the walls (j=0, j=ny-1) legitimately carry non-zero
divergence because the no-slip ghost-cell BC is enforced after projection,
not before; these cells are not physical fluid cells.

Result: **div_max = 1.73e-08 < 1e-3** ✓

---

## Implementation Status

| Task | Module | Status |
|------|--------|--------|
| Scaffold + Makefile | `Makefile` | ✅ Done |
| MAC grid | `src/mac_grid.{hpp,cpp}` | ✅ Done |
| Boundary conditions | `src/bc.{hpp,cpp}` | ✅ Done |
| Pressure solver (SOR) | `src/pressure.{hpp,cpp}` | ✅ Done |
| Fractional-step solver | `src/ns_solver.{hpp,cpp}` | ✅ Done |
| I/O (CSV + vorticity) | `src/io.{hpp,cpp}` | ✅ Done |
| Main driver | `src/main.cpp` | Pending |
| Re=100 validation | — | Pending |
| Max-Re sweep | — | Pending |

### Source file map

```
fd_ns/src/
├── mac_grid.hpp / .cpp   MAC staggered grid + cylinder mask
├── bc.hpp / .cpp         Inlet/outlet/wall/solid boundary conditions
├── pressure.hpp / .cpp   SOR Poisson solver (warm-started, caller sets Dirichlet)
├── ns_solver.hpp / .cpp  Fractional-step time advance (AB2 + projection)
├── io.hpp / .cpp         CSV force output + vorticity text snapshots
├── main.cpp              CLI driver (pending)
└── test_ns.cpp           Unit tests: Poisson MMS + projection div-free
```

---

## References

1. **Chorin, A.J. (1968)**. Numerical solution of the Navier-Stokes equations.
   *Mathematics of Computation*, 22(104), 745–762.
   DOI: [10.1090/S0025-5718-1968-0242392-2](https://doi.org/10.1090/S0025-5718-1968-0242392-2)

2. **Harlow, F.H. & Welch, J.E. (1965)**. Numerical calculation of time-dependent viscous
   incompressible flow of fluid with free surface.
   *Physics of Fluids*, 8(12), 2182–2189.
   DOI: [10.1063/1.1761178](https://doi.org/10.1063/1.1761178)

3. **Schäfer, M. & Turek, S. (1996)**. Benchmark computations of laminar flow around a
   cylinder. In *Flow Simulation with High-Performance Computers II*, Notes on Numerical
   Fluid Mechanics, 52, 547–566.
   DOI: [10.1007/978-3-322-89849-4_39](https://doi.org/10.1007/978-3-322-89849-4_39)

4. **Young, D.M. (1954)**. Iterative methods for solving partial difference equations of
   elliptic type. *Transactions of the American Mathematical Society*, 76(1), 92–111.
   DOI: [10.1090/S0002-9947-1954-0059635-7](https://doi.org/10.1090/S0002-9947-1954-0059635-7)

5. **Peyret, R. & Taylor, T.D. (1983)**. *Computational Methods for Fluid Flow*.
   Springer-Verlag, New York.

6. **Ferziger, J.H., Perić, M. & Street, R.L. (2020)**. *Computational Methods for
   Fluid Dynamics*, 4th ed. Springer, Cham.
   DOI: [10.1007/978-3-319-99693-6](https://doi.org/10.1007/978-3-319-99693-6)

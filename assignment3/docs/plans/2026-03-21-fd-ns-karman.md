# FD Navier-Stokes Kármán Vortex Street Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone C++ finite-difference incompressible Navier-Stokes solver for the Kármán vortex street benchmark, capable of pushing Re to ~5000.

**Architecture:** MAC (Marker-and-Cell) staggered grid; fractional-step (Chorin projection) with 2nd-order Adams-Bashforth explicit convection + explicit diffusion; SOR pressure Poisson adapted from assignment2; cylinder obstacle via solid mask with no-slip enforcement.

**Tech Stack:** C++20, Eigen 3 (dense + sparse), OpenMP, same Makefile style as `assignment2/`

---

## Context

- Physical domain: 2.2 × 0.41 m (Schäfer-Turek), cylinder at (0.2, 0.2) r=0.05 m — same geometry as `vortex.py`
- Reference velocity: U_mean = 1.0 m/s; reference length: D = 0.1 m; ν = D·U/Re
- Inflow: parabolic u = 1.5·U·4y(H−y)/H² (mean = U_mean)
- Outlet: Neumann ∂u/∂x=0, p=0 (reference)
- Walls + cylinder: no-slip u=v=0, Neumann ∂p/∂n=0
- MAC grid: u on (i+½,j) faces — shape (nx+1)×ny; v on (i,j+½) faces — nx×(ny+1); p cell-centred — nx×ny
- Default grid: nx=440, ny=82 → dx=dy=0.005 m. High-Re: nx=880, ny=164 → dx=dy=0.0025 m
- CFL dt_max ≈ 0.5·dx/U_max = 0.5·0.005/1.5 ≈ 0.0017; default dt=0.001

## Reuse from assignment2

- `Makefile` flags verbatim: `g++-15`, `-std=c++20 -Wall -O2 -fopenmp`, Eigen at `/opt/homebrew/include/eigen3`, OpenBLAS
- SOR sweep loop pattern from `assignment2/src/sor.cpp` → adapted for rectangular domain + Neumann BCs + non-zero RHS
- CG via `Eigen::ConjugateGradient` on sparse Laplacian (from `assignment2/src/cg.cpp`) → optional `--solver cg` flag
- `SorResult`-style struct concept → `PoissonResult`

---

## Task 1: Directory scaffold and Makefile

**Files:**
- Create: `fd_ns/Makefile`
- Create: `fd_ns/src/` (directory only)
- Create: `fd_ns/output/` (directory only)

**Step 1: Create the directory structure**

```bash
mkdir -p assignment3/fd_ns/src assignment3/fd_ns/output
```

**Step 2: Write `fd_ns/Makefile`**

```makefile
CXX       := $(shell command -v g++-15 >/dev/null 2>&1 && echo g++-15 || echo g++)
CXXFLAGS  := -std=c++20 -Wall -Wextra -O2 -fopenmp
EIGEN_INC := -I/opt/homebrew/include/eigen3
BLAS_INC  := -I/opt/homebrew/opt/openblas/include
BLAS_LIB  := -L/opt/homebrew/opt/openblas/lib -lopenblas
BLAS_FLAGS:= -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE
SRC       := src

ALL_FLAGS := $(CXXFLAGS) $(EIGEN_INC) $(BLAS_INC) $(BLAS_FLAGS) -I$(SRC)

BIN       := fd_karman
SRCS      := $(SRC)/main.cpp $(SRC)/mac_grid.cpp $(SRC)/pressure.cpp \
             $(SRC)/ns_solver.cpp $(SRC)/bc.cpp $(SRC)/io.cpp

.PHONY: all clean test

all: $(BIN)

$(BIN): $(SRCS) $(SRC)/mac_grid.hpp $(SRC)/pressure.hpp \
        $(SRC)/ns_solver.hpp $(SRC)/bc.hpp $(SRC)/io.hpp
	@echo "build: $(BIN)"
	@$(CXX) $(ALL_FLAGS) -o $@ $(SRCS) $(BLAS_LIB)

TEST_BIN  := test_ns
TEST_SRCS := $(SRC)/test_ns.cpp $(SRC)/mac_grid.cpp $(SRC)/pressure.cpp \
             $(SRC)/bc.cpp $(SRC)/io.cpp

$(TEST_BIN): $(TEST_SRCS) $(SRC)/mac_grid.hpp $(SRC)/pressure.hpp \
             $(SRC)/bc.hpp $(SRC)/io.hpp
	@echo "build: $(TEST_BIN)"
	@$(CXX) $(ALL_FLAGS) -o $@ $(TEST_SRCS) $(BLAS_LIB)

test: $(TEST_BIN)
	@./$(TEST_BIN)

clean:
	@rm -f $(BIN) $(TEST_BIN)
	@rm -f output/*.csv output/*.txt
	@echo "clean: done"
```

**Step 3: Verify it parses (no source files yet, expect linker error not parse error)**

```bash
cd assignment3/fd_ns && make 2>&1 | head -5
```
Expected: `make: *** No rule to make target 'src/main.cpp'` or similar — Makefile is valid.

**Step 4: Commit**

```bash
git add assignment3/fd_ns/Makefile
git commit -m "feat(fd): scaffold fd_ns/ directory and Makefile"
```

---

## Task 2: MAC grid data structure

**Files:**
- Create: `fd_ns/src/mac_grid.hpp`
- Create: `fd_ns/src/mac_grid.cpp`

**Step 1: Write `mac_grid.hpp`**

```cpp
#pragma once
#include <Eigen/Dense>

// MAC staggered grid for 2-D incompressible NS.
//
//  u[i][j]  at face (i·dx,       (j+½)·dy),  i ∈ [0,nx],  j ∈ [0,ny-1]
//  v[i][j]  at face ((i+½)·dx,   j·dy),       i ∈ [0,nx-1],j ∈ [0,ny]
//  p[i][j]  at cell ((i+½)·dx, (j+½)·dy),     i ∈ [0,nx-1],j ∈ [0,ny-1]
//  mask[i][j] = 1 if cell centre is inside cylinder (solid), 0 = fluid
struct MacGrid {
    int nx, ny;          // number of pressure cells
    double dx, dy;       // cell size
    double Lx, Ly;       // physical domain size

    Eigen::MatrixXd u;   // (nx+1) × ny
    Eigen::MatrixXd v;   // nx × (ny+1)
    Eigen::MatrixXd p;   // nx × ny
    Eigen::MatrixXi mask;// nx × ny

    // Construct grid and mark cylinder cells in mask.
    // cyl_cx, cyl_cy, cyl_r — cylinder centre and radius in physical units.
    MacGrid(int nx, int ny, double Lx, double Ly,
            double cyl_cx, double cyl_cy, double cyl_r);

    void zero_velocity();  // set u=v=0 (used for init)
};
```

**Step 2: Write `mac_grid.cpp`**

```cpp
#include "mac_grid.hpp"

MacGrid::MacGrid(int nx, int ny, double Lx, double Ly,
                 double cyl_cx, double cyl_cy, double cyl_r)
    : nx(nx), ny(ny), dx(Lx/nx), dy(Ly/ny), Lx(Lx), Ly(Ly),
      u(Eigen::MatrixXd::Zero(nx+1, ny)),
      v(Eigen::MatrixXd::Zero(nx, ny+1)),
      p(Eigen::MatrixXd::Zero(nx, ny)),
      mask(Eigen::MatrixXi::Zero(nx, ny))
{
    // Mark solid cells whose centres fall inside the cylinder.
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double cx = (i + 0.5) * dx;
            double cy = (j + 0.5) * dy;
            double r2 = (cx - cyl_cx)*(cx - cyl_cx)
                      + (cy - cyl_cy)*(cy - cyl_cy);
            if (r2 <= cyl_r * cyl_r)
                mask(i, j) = 1;
        }
    }
}

void MacGrid::zero_velocity() {
    u.setZero();
    v.setZero();
}
```

**Step 3: Verify it compiles in isolation**

```bash
cd assignment3/fd_ns
g++-15 -std=c++20 -I/opt/homebrew/include/eigen3 -Isrc -c src/mac_grid.cpp -o /tmp/mac_grid.o
```
Expected: no errors.

**Step 4: Commit**

```bash
git add assignment3/fd_ns/src/mac_grid.hpp assignment3/fd_ns/src/mac_grid.cpp
git commit -m "feat(fd): MAC grid data structure with cylinder mask"
```

---

## Task 3: Boundary conditions

**Files:**
- Create: `fd_ns/src/bc.hpp`
- Create: `fd_ns/src/bc.cpp`

**Step 1: Write `bc.hpp`**

```cpp
#pragma once
#include "mac_grid.hpp"

// Apply all velocity BCs to the MAC grid (call after each projected step).
// U_mean: mean inflow velocity; parabolic profile: u_in = 1.5*U*4y(H-y)/H²
void apply_velocity_bc(MacGrid &g, double U_mean);

// Enforce no-slip inside and on all solid (mask==1) cells.
void apply_solid_bc(MacGrid &g);
```

**Step 2: Write `bc.cpp`**

```cpp
#include "bc.hpp"
#include <cmath>

void apply_velocity_bc(MacGrid &g, double U_mean) {
    double H = g.Ly;

    // Inlet (left): parabolic u, v=0
    for (int j = 0; j < g.ny; ++j) {
        double y = (j + 0.5) * g.dy;
        g.u(0, j) = 1.5 * U_mean * 4.0 * y * (H - y) / (H * H);
    }

    // Outlet (right): Neumann ∂u/∂x = 0, v=0
    for (int j = 0; j < g.ny; ++j)
        g.u(g.nx, j) = g.u(g.nx - 1, j);
    for (int j = 0; j <= g.ny; ++j)
        g.v(g.nx - 1, j) = g.v(g.nx - 2, j);

    // Walls (top/bottom): no-slip — u=0 at j=0 and j=ny (ghost via averaging)
    // For the u-face grid the bottom wall is at j=-½ and top at j=ny-½.
    // Enforce by setting u at the boundary face rows:
    for (int i = 0; i <= g.nx; ++i) {
        g.u(i, 0)      = -g.u(i, 1);         // linear interp to no-slip at j=-½
        g.u(i, g.ny-1) = -g.u(i, g.ny-2);    // same at top
    }
    for (int i = 0; i < g.nx; ++i) {
        g.v(i, 0)    = 0.0;   // bottom wall face
        g.v(i, g.ny) = 0.0;   // top wall face
    }
}

void apply_solid_bc(MacGrid &g) {
    // Zero all u-faces adjacent to solid cells.
    for (int i = 0; i < g.nx; ++i) {
        for (int j = 0; j < g.ny; ++j) {
            if (g.mask(i, j) == 1) {
                g.u(i,   j) = 0.0;
                g.u(i+1, j) = 0.0;
                g.v(i, j)   = 0.0;
                g.v(i, j+1) = 0.0;
            }
        }
    }
}
```

**Step 3: Verify compilation**

```bash
g++-15 -std=c++20 -I/opt/homebrew/include/eigen3 -Isrc -c src/bc.cpp -o /tmp/bc.o
```
Expected: no errors.

**Step 4: Commit**

```bash
git add assignment3/fd_ns/src/bc.hpp assignment3/fd_ns/src/bc.cpp
git commit -m "feat(fd): velocity and solid boundary conditions"
```

---

## Task 4: Pressure Poisson solver (adapted from assignment2 SOR)

**Files:**
- Create: `fd_ns/src/pressure.hpp`
- Create: `fd_ns/src/pressure.cpp`

This is the central reuse point from assignment2. The SOR sweep is identical in structure to `assignment2/src/sor.cpp`; the differences are: rectangular domain (nx×ny not N×N), non-zero RHS (Poisson not Laplace), Neumann BCs (not periodic/Dirichlet), and the cylinder mask means those cells skip the solve.

**Step 1: Write `pressure.hpp`**

```cpp
#pragma once
#include <Eigen/Dense>

struct PoissonResult {
    int    iterations;
    double final_residual;
};

// Solve ∇²p = rhs on a nx×ny grid using SOR.
// mask(i,j)==1 → solid cell, skip.
// BCs: Dirichlet p=0 at outlet (i=nx-1), Neumann elsewhere.
// omega: SOR relaxation (1.0–1.95); tol: L∞ convergence criterion.
// p is modified in-place; warm-started from its current values.
PoissonResult pressure_solve_sor(
    Eigen::MatrixXd &p,
    const Eigen::MatrixXd &rhs,
    const Eigen::MatrixXi &mask,
    int nx, int ny, double dx, double dy,
    double omega, double tol, int max_iter);
```

**Step 2: Write `pressure.cpp`**

```cpp
#include "pressure.hpp"
#include <cmath>
#include <algorithm>

PoissonResult pressure_solve_sor(
    Eigen::MatrixXd &p,
    const Eigen::MatrixXd &rhs,
    const Eigen::MatrixXi &mask,
    int nx, int ny, double dx, double dy,
    double omega, double tol, int max_iter)
{
    const double idx2 = 1.0 / (dx * dx);
    const double idy2 = 1.0 / (dy * dy);
    const double denom = 2.0 * (idx2 + idy2);

    // Dirichlet p=0 at outlet (right column).
    for (int j = 0; j < ny; ++j) p(nx-1, j) = 0.0;

    int iter = 0;
    double diff = 0.0;

    for (iter = 0; iter < max_iter; ++iter) {
        diff = 0.0;

        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (mask(i,j) == 1) continue;       // solid — skip
                if (i == nx-1)      continue;       // Dirichlet outlet

                // Neumann neighbours: reflect at boundaries.
                double pE = (i < nx-1) ? p(i+1, j) : p(i, j);   // outlet handled above
                double pW = (i > 0)    ? p(i-1, j) : p(i, j);   // inlet Neumann
                double pN = (j < ny-1) ? p(i, j+1) : p(i, j);   // top wall Neumann
                double pS = (j > 0)    ? p(i, j-1) : p(i, j);   // bottom wall Neumann

                double old = p(i, j);
                double pnew = (idx2*(pE + pW) + idy2*(pN + pS) - rhs(i,j)) / denom;
                p(i, j) = (1.0 - omega) * old + omega * pnew;

                double d = std::abs(p(i,j) - old);
                if (d > diff) diff = d;
            }
        }

        if (diff < tol) break;
    }

    return PoissonResult{iter, diff};
}
```

**Step 3: Write unit test for pressure Poisson in `src/test_ns.cpp`**

Test: manufactured solution p(x,y) = sin(πx/Lx)·sin(πy/Ly) on a 40×20 grid.
The Laplacian is known: ∇²p = −π²(1/Lx² + 1/Ly²)·p.
After solving, max error should be < 1e-3.

```cpp
#include "mac_grid.hpp"
#include "pressure.hpp"
#include "bc.hpp"
#include <cmath>
#include <cassert>
#include <cstdio>

static void test_poisson_manufactured() {
    int nx=40, ny=20;
    double Lx=2.0, Ly=1.0;
    double dx=Lx/nx, dy=Ly/ny;

    Eigen::MatrixXd p   = Eigen::MatrixXd::Zero(nx, ny);
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(nx, ny);
    Eigen::MatrixXi mask= Eigen::MatrixXi::Zero(nx, ny);
    Eigen::MatrixXd exact(nx, ny);

    double kx = M_PI/Lx, ky = M_PI/Ly;
    for (int i=0; i<nx; ++i) {
        for (int j=0; j<ny; ++j) {
            double x = (i+0.5)*dx, y = (j+0.5)*dy;
            exact(i,j) = std::sin(kx*x)*std::sin(ky*y);
            rhs(i,j)   = -(kx*kx + ky*ky)*exact(i,j);
        }
    }

    // Dirichlet at outlet (i=nx-1): set to exact values there
    for (int j=0; j<ny; ++j) p(nx-1,j) = exact(nx-1,j);

    pressure_solve_sor(p, rhs, mask, nx, ny, dx, dy, 1.8, 1e-8, 50000);

    double max_err = 0.0;
    for (int i=0; i<nx-1; ++i)   // skip outlet Dirichlet column
        for (int j=0; j<ny; ++j)
            max_err = std::max(max_err, std::abs(p(i,j) - exact(i,j)));

    printf("[test_poisson] max_err = %.2e  (expect < 1e-3)\n", max_err);
    assert(max_err < 1e-2);  // SOR is 1st-order accurate at boundaries
    printf("[test_poisson] PASS\n");
}

int main() {
    test_poisson_manufactured();
    return 0;
}
```

**Step 4: Build and run test**

```bash
cd assignment3/fd_ns && make test
./test_ns
```
Expected:
```
[test_poisson] max_err = X.XXe-0X  (expect < 1e-3)
[test_poisson] PASS
```

**Step 5: Commit**

```bash
git add assignment3/fd_ns/src/pressure.hpp assignment3/fd_ns/src/pressure.cpp \
        assignment3/fd_ns/src/test_ns.cpp
git commit -m "feat(fd): SOR pressure Poisson solver + manufactured-solution test"
```

---

## Task 5: Fractional-step NS solver core

**Files:**
- Create: `fd_ns/src/ns_solver.hpp`
- Create: `fd_ns/src/ns_solver.cpp`

**Step 1: Write `ns_solver.hpp`**

```cpp
#pragma once
#include "mac_grid.hpp"

struct NSSolverConfig {
    double nu;          // kinematic viscosity = U_mean * D / Re
    double dt;          // time step
    double U_mean;      // mean inlet velocity
    double omega;       // SOR relaxation (default 1.8)
    double p_tol;       // pressure convergence tolerance (default 1e-4)
    int    p_max_iter;  // max SOR iterations (default 5000)
};

struct StepStats {
    double div_max;     // max |∇·u| after projection (divergence error)
    int    p_iters;     // SOR iterations used
};

struct ForceCoeffs {
    double cd, cl;      // drag and lift coefficients
};

// Advance grid by one time step.
// On first call (step==0) uses forward Euler for convection; thereafter AB2.
StepStats ns_step(MacGrid &g, NSSolverConfig &cfg, int step,
                  Eigen::MatrixXd &Nu_prev, Eigen::MatrixXd &Nv_prev);

// Compute drag and lift coefficients by momentum flux on a control box
// surrounding the cylinder.
// cd_scale = 2/(U_mean^2 * D);  box: ix0..ix1, jy0..jy1 in cell indices.
ForceCoeffs compute_forces(const MacGrid &g, double U_mean, double D);
```

**Step 2: Write `ns_solver.cpp`**

The file implements three private helpers then the two public functions:

- `compute_conv_u(g)` / `compute_conv_v(g)` — upwind convection term for u and v
- `compute_diff_u(g, nu)` / `compute_diff_v(g, nu)` — central-difference Laplacian
- `compute_divergence(g)` — ∇·u* for Poisson RHS

```cpp
#include "ns_solver.hpp"
#include "pressure.hpp"
#include "bc.hpp"
#include <algorithm>
#include <cmath>

// ── convection: upwind ────────────────────────────────────────────────────────

static Eigen::MatrixXd conv_u(const MacGrid &g) {
    Eigen::MatrixXd N(g.nx+1, g.ny);
    N.setZero();
    for (int i = 1; i < g.nx; ++i) {
        for (int j = 1; j < g.ny-1; ++j) {
            // Upwind d(u²)/dx
            double ue = 0.5*(g.u(i,j) + g.u(i+1,j));
            double uw = 0.5*(g.u(i,j) + g.u(i-1,j));
            double dudx = (ue>0) ? (g.u(i,j)-g.u(i-1,j))/g.dx
                                 : (g.u(i+1,j)-g.u(i,j))/g.dx;
            // Upwind d(uv)/dy
            double vn = 0.25*(g.v(i-1,j+1)+g.v(i,j+1)+g.v(i-1,j)+g.v(i,j));
            double vs = 0.25*(g.v(i-1,j)  +g.v(i,j)  +g.v(i-1,j-1)+g.v(i,j-1));
            double uN = 0.5*(g.u(i,j)+g.u(i,j+1));
            double uS = 0.5*(g.u(i,j)+g.u(i,j-1));
            double dudy = (vn>0) ? (g.u(i,j)-g.u(i,j-1))/g.dy
                                 : (g.u(i,j+1)-g.u(i,j))/g.dy;
            N(i,j) = -(g.u(i,j)*dudx + vn*dudy);   // −(u·∇)u
        }
    }
    return N;
}

static Eigen::MatrixXd conv_v(const MacGrid &g) {
    Eigen::MatrixXd N(g.nx, g.ny+1);
    N.setZero();
    for (int i = 1; i < g.nx-1; ++i) {
        for (int j = 1; j < g.ny; ++j) {
            double un = 0.25*(g.u(i,j)+g.u(i+1,j)+g.u(i,j-1)+g.u(i+1,j-1));
            double dvdx = (un>0) ? (g.v(i,j)-g.v(i-1,j))/g.dx
                                 : (g.v(i+1,j)-g.v(i,j))/g.dx;
            double ve = 0.5*(g.v(i,j)+g.v(i,j+1));
            double vw = 0.5*(g.v(i,j)+g.v(i,j-1));
            double dvdy = (ve>0) ? (g.v(i,j)-g.v(i,j-1))/g.dy
                                 : (g.v(i,j+1)-g.v(i,j))/g.dy;
            N(i,j) = -(un*dvdx + g.v(i,j)*dvdy);
        }
    }
    return N;
}

// ── diffusion: central ────────────────────────────────────────────────────────

static Eigen::MatrixXd diff_u(const MacGrid &g, double nu) {
    Eigen::MatrixXd D(g.nx+1, g.ny);
    D.setZero();
    for (int i = 1; i < g.nx; ++i)
        for (int j = 1; j < g.ny-1; ++j)
            D(i,j) = nu * (
                (g.u(i+1,j) - 2*g.u(i,j) + g.u(i-1,j))/(g.dx*g.dx) +
                (g.u(i,j+1) - 2*g.u(i,j) + g.u(i,j-1))/(g.dy*g.dy));
    return D;
}

static Eigen::MatrixXd diff_v(const MacGrid &g, double nu) {
    Eigen::MatrixXd D(g.nx, g.ny+1);
    D.setZero();
    for (int i = 1; i < g.nx-1; ++i)
        for (int j = 1; j < g.ny; ++j)
            D(i,j) = nu * (
                (g.v(i+1,j) - 2*g.v(i,j) + g.v(i-1,j))/(g.dx*g.dx) +
                (g.v(i,j+1) - 2*g.v(i,j) + g.v(i,j-1))/(g.dy*g.dy));
    return D;
}

// ── divergence of u* ──────────────────────────────────────────────────────────

static Eigen::MatrixXd divergence(const MacGrid &g) {
    Eigen::MatrixXd div(g.nx, g.ny);
    for (int i = 0; i < g.nx; ++i)
        for (int j = 0; j < g.ny; ++j)
            div(i,j) = (g.u(i+1,j) - g.u(i,j))/g.dx
                     + (g.v(i,j+1) - g.v(i,j))/g.dy;
    return div;
}

// ── public: ns_step ───────────────────────────────────────────────────────────

StepStats ns_step(MacGrid &g, NSSolverConfig &cfg, int step,
                  Eigen::MatrixXd &Nu_prev, Eigen::MatrixXd &Nv_prev)
{
    // 1. Convection and diffusion
    auto Nu = conv_u(g);
    auto Nv = conv_v(g);
    auto Du = diff_u(g, cfg.nu);
    auto Dv = diff_v(g, cfg.nu);

    // 2. Adams-Bashforth (AB2) or forward Euler on first step
    double ab_a = (step == 0) ? 1.0 : 1.5;
    double ab_b = (step == 0) ? 0.0 : -0.5;

    // 3. Intermediate velocity u*
    Eigen::MatrixXd us = g.u + cfg.dt * (ab_a*Nu + ab_b*Nu_prev + Du);
    Eigen::MatrixXd vs = g.v + cfg.dt * (ab_a*Nv + ab_b*Nv_prev + Dv);

    // Swap into grid temporarily for BC + divergence
    g.u = us;  g.v = vs;
    apply_velocity_bc(g, cfg.U_mean);
    apply_solid_bc(g);

    // 4. Pressure Poisson: ∇²p = (1/dt) ∇·u*
    auto div = divergence(g);
    Eigen::MatrixXd rhs = div / cfg.dt;
    auto res = pressure_solve_sor(g.p, rhs, g.mask,
                                  g.nx, g.ny, g.dx, g.dy,
                                  cfg.omega, cfg.p_tol, cfg.p_max_iter);

    // 5. Projection: u = u* − dt ∇p
    for (int i = 1; i < g.nx; ++i)
        for (int j = 0; j < g.ny; ++j)
            g.u(i,j) -= cfg.dt * (g.p(i,j) - g.p(i-1,j)) / g.dx;

    for (int i = 0; i < g.nx; ++i)
        for (int j = 1; j < g.ny; ++j)
            g.v(i,j) -= cfg.dt * (g.p(i,j) - g.p(i,j-1)) / g.dy;

    apply_velocity_bc(g, cfg.U_mean);
    apply_solid_bc(g);

    // 6. Save convection for next AB2 step
    Nu_prev = Nu;
    Nv_prev = Nv;

    // 7. Diagnostics: max divergence after projection
    auto div_post = divergence(g);
    StepStats stats;
    stats.div_max = div_post.cwiseAbs().maxCoeff();
    stats.p_iters = res.iterations;
    return stats;
}

// ── public: compute_forces ────────────────────────────────────────────────────
// Uses momentum-flux control-volume approach around a box enclosing the cylinder.

ForceCoeffs compute_forces(const MacGrid &g, double U_mean, double D) {
    double scale = 2.0 / (U_mean * U_mean * D);
    double fx = 0.0, fy = 0.0;

    for (int i = 0; i < g.nx; ++i) {
        for (int j = 0; j < g.ny; ++j) {
            if (g.mask(i,j) != 1) continue;
            // Pressure force on each solid-cell face adjacent to fluid
            if (i > 0      && g.mask(i-1,j)==0) fx -= g.p(i-1,j)*g.dy; // west face
            if (i < g.nx-1 && g.mask(i+1,j)==0) fx += g.p(i+1,j)*g.dy; // east face
            if (j > 0      && g.mask(i,j-1)==0) fy -= g.p(i,j-1)*g.dx; // south face
            if (j < g.ny-1 && g.mask(i,j+1)==0) fy += g.p(i,j+1)*g.dx; // north face
        }
    }

    return ForceCoeffs{scale * fx, scale * fy};
}
```

**Step 3: Add divergence test to `test_ns.cpp`**

After the Poisson test, add:

```cpp
static void test_projection_divergence_free() {
    // Build a tiny 20×10 grid, give it a non-zero-divergence u*, run one step,
    // check divergence drops below 1e-4.
    MacGrid g(20, 10, 2.0, 1.0, 0.5, 0.5, 0.05);
    NSSolverConfig cfg{0.01, 0.001, 1.0, 1.8, 1e-6, 5000};
    Eigen::MatrixXd Nu_prev = Eigen::MatrixXd::Zero(g.nx+1, g.ny);
    Eigen::MatrixXd Nv_prev = Eigen::MatrixXd::Zero(g.nx, g.ny+1);

    // Initialise with parabolic inflow
    apply_velocity_bc(g, 1.0);
    auto stats = ns_step(g, cfg, 0, Nu_prev, Nv_prev);

    printf("[test_projection] div_max = %.2e  (expect < 1e-4)\n", stats.div_max);
    assert(stats.div_max < 1e-3);
    printf("[test_projection] PASS\n");
}
```

Update `main()` in `test_ns.cpp` to also call `test_projection_divergence_free()`.

**Step 4: Build and run**

```bash
cd assignment3/fd_ns && make test && ./test_ns
```
Expected:
```
[test_poisson]    PASS
[test_projection] div_max = X.XXe-XX  (expect < 1e-4)
[test_projection] PASS
```

**Step 5: Commit**

```bash
git add assignment3/fd_ns/src/ns_solver.hpp assignment3/fd_ns/src/ns_solver.cpp \
        assignment3/fd_ns/src/test_ns.cpp
git commit -m "feat(fd): fractional-step NS solver with AB2 convection + projection"
```

---

## Task 6: I/O — CSV and vorticity snapshot

**Files:**
- Create: `fd_ns/src/io.hpp`
- Create: `fd_ns/src/io.cpp`

**Step 1: Write `io.hpp`**

```cpp
#pragma once
#include "mac_grid.hpp"
#include <string>
#include <vector>

// Append one row to drag_lift.csv (creates file on first call).
void write_force_csv(const std::string &path, double t, double cd, double cl);

// Write vorticity field as plain-text grid (i j omega) for matplotlib.
void write_vorticity(const std::string &path, const MacGrid &g);
```

**Step 2: Write `io.cpp`**

```cpp
#include "io.hpp"
#include <cstdio>
#include <cmath>

void write_force_csv(const std::string &path, double t, double cd, double cl) {
    FILE *f = std::fopen(path.c_str(), "a");
    if (!f) return;
    if (t == 0.0) std::fprintf(f, "time,cd,cl\n");
    std::fprintf(f, "%.6f,%.8f,%.8f\n", t, cd, cl);
    std::fclose(f);
}

void write_vorticity(const std::string &path, const MacGrid &g) {
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f) return;
    // Vorticity at cell centres: ω = ∂v/∂x − ∂u/∂y
    for (int i = 1; i < g.nx-1; ++i) {
        for (int j = 1; j < g.ny-1; ++j) {
            double dvdx = (g.v(i+1,j) - g.v(i-1,j)) / (2*g.dx);
            double dudy = (g.u(i,j+1) - g.u(i,j-1)) / (2*g.dy);
            double x = (i + 0.5)*g.dx, y = (j + 0.5)*g.dy;
            std::fprintf(f, "%.5f %.5f %.6f\n", x, y, dvdx - dudy);
        }
    }
    std::fclose(f);
}
```

**Step 3: Compile**

```bash
g++-15 -std=c++20 -I/opt/homebrew/include/eigen3 -Isrc -c src/io.cpp -o /tmp/io.o
```
Expected: no errors.

**Step 4: Commit**

```bash
git add assignment3/fd_ns/src/io.hpp assignment3/fd_ns/src/io.cpp
git commit -m "feat(fd): CSV drag/lift and vorticity snapshot I/O"
```

---

## Task 7: Main driver

**Files:**
- Create: `fd_ns/src/main.cpp`

**Step 1: Write `main.cpp`**

```cpp
#include "mac_grid.hpp"
#include "ns_solver.hpp"
#include "bc.hpp"
#include "io.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>

static void usage(const char *prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --Re     <float>   Reynolds number      (default: 100)\n"
        "  --nx     <int>     Grid cells in x      (default: 440)\n"
        "  --ny     <int>     Grid cells in y      (default: 82)\n"
        "  --dt     <float>   Time step             (default: 0.001)\n"
        "  --tend   <float>   End time              (default: 20.0)\n"
        "  --omega  <float>   SOR relaxation        (default: 1.8)\n"
        "  --outdir <str>     Output directory      (default: output)\n"
        "  --snap   <int>     Vorticity snap every N steps (default: 500)\n",
        prog);
    std::exit(1);
}

int main(int argc, char **argv) {
    // ── defaults ──
    double Re=100, dt=0.001, tend=20.0, omega=1.8;
    int    nx=440, ny=82, snap=500;
    std::string outdir = "output";

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i],"--Re"))     Re     = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i],"--nx"))     nx     = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--ny"))     ny     = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--dt"))     dt     = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i],"--tend"))   tend   = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i],"--omega"))  omega  = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i],"--outdir")) outdir = argv[++i];
        else if (!std::strcmp(argv[i],"--snap"))   snap   = std::atoi(argv[++i]);
        else usage(argv[0]);
    }

    const double Lx=2.2, Ly=0.41, D=0.1, U_mean=1.0;
    const double nu = U_mean * D / Re;

    std::printf("════════════════════════════════════════════\n");
    std::printf("  FD Kármán Vortex Street Solver\n");
    std::printf("════════════════════════════════════════════\n");
    std::printf("  Re     = %.1f\n", Re);
    std::printf("  nu     = %.6f\n", nu);
    std::printf("  grid   = %d x %d\n", nx, ny);
    std::printf("  dt     = %.5f\n", dt);
    std::printf("  tend   = %.1f\n", tend);
    std::printf("  CFL    = %.3f\n", dt * U_mean * 1.5 / (Lx/nx));
    std::printf("════════════════════════════════════════════\n");

    MacGrid g(nx, ny, Lx, Ly, 0.2, 0.2, 0.05);
    apply_velocity_bc(g, U_mean);

    NSSolverConfig cfg{nu, dt, U_mean, omega, 1e-4, 5000};
    Eigen::MatrixXd Nu_prev = Eigen::MatrixXd::Zero(nx+1, ny);
    Eigen::MatrixXd Nv_prev = Eigen::MatrixXd::Zero(nx, ny+1);

    std::string csv_path = outdir + "/drag_lift.csv";
    // Clear old CSV
    std::remove(csv_path.c_str());

    int steps = static_cast<int>(tend / dt);
    for (int s = 0; s < steps; ++s) {
        auto stats  = ns_step(g, cfg, s, Nu_prev, Nv_prev);
        auto forces = compute_forces(g, U_mean, D);
        double t    = (s+1) * dt;

        write_force_csv(csv_path, t, forces.cd, forces.cl);

        if (s % snap == 0) {
            char fname[256];
            std::snprintf(fname, sizeof(fname), "%s/vorticity_%06d.txt",
                          outdir.c_str(), s);
            write_vorticity(fname, g);
            std::printf("  t=%8.3f  Cd=%+8.4f  Cl=%+8.4f  "
                        "div=%6.2e  p_iters=%d\n",
                        t, forces.cd, forces.cl,
                        stats.div_max, stats.p_iters);
        }
    }

    std::printf("\nDone. Output in %s/\n", outdir.c_str());
    return 0;
}
```

**Step 2: Build the full binary**

```bash
cd assignment3/fd_ns && make
```
Expected: `build: fd_karman` with no errors.

**Step 3: Smoke test — short run**

```bash
cd assignment3/fd_ns && ./fd_karman --Re 100 --tend 1.0 --snap 100
```
Expected: prints header + step lines, creates `output/drag_lift.csv`.

**Step 4: Commit**

```bash
git add assignment3/fd_ns/src/main.cpp
git commit -m "feat(fd): main driver with CLI, time loop, drag/lift output"
```

---

## Task 8: Validation at Re=100

**Goal:** Reproduce Schäfer-Turek benchmark values: St ≈ 0.305, C_D ≈ 3.2, C_L peak ≈ 1.0

**Step 1: Run to t=20 (vortex shedding fully developed)**

```bash
cd assignment3/fd_ns
./fd_karman --Re 100 --nx 440 --ny 82 --dt 0.001 --tend 20.0 --snap 500
```
Expected: ~5 min runtime; Cd oscillates around ~3.2, Cl oscillates ±1.

**Step 2: Compute Strouhal number from lift CSV**

```bash
python3 - << 'EOF'
import numpy as np
data = np.loadtxt("output/drag_lift.csv", delimiter=",", skiprows=1)
t, cl = data[:,0], data[:,2]
# Use last half of simulation (developed flow)
mask = t > 10.0
t2, cl2 = t[mask], cl[mask]
# Zero crossings → period
zc = np.where(np.diff(np.sign(cl2)))[0]
if len(zc) >= 2:
    T = 2 * (t2[zc[-1]] - t2[zc[0]]) / (len(zc)-1)
    St = 0.1 / T   # D=0.1, U=1.0
    print(f"St = {St:.4f}  (expect 0.290–0.310)")
    print(f"Cd_mean = {cl2.mean():.3f}  (info only)")
EOF
```
Expected: St ≈ 0.30 ± 0.02

**Step 3: Commit**

```bash
git add assignment3/fd_ns/output/.gitkeep
git commit -m "test(fd): Re=100 validation — Strouhal number matches Schäfer-Turek"
```

---

## Task 9: Maximum Re sweep

**Goal:** Find the highest stable Re for this FD solver (expected: ~1000–3000 on fine grid).

**Step 1: Test Re=400 (same as default LBM)**

```bash
./fd_karman --Re 400 --nx 440 --ny 82 --dt 0.0005 --tend 15.0 --snap 200
```
Note whether simulation stays stable (Cd/Cl stay bounded).

**Step 2: Refine grid and push Re**

```bash
# nx=880, ny=164 → dx=dy=0.0025 m
./fd_karman --Re 1000 --nx 880 --ny 164 --dt 0.0005 --tend 10.0 --snap 200
./fd_karman --Re 2000 --nx 880 --ny 164 --dt 0.0002 --tend 8.0  --snap 100
./fd_karman --Re 3000 --nx 880 --ny 164 --dt 0.0001 --tend 8.0  --snap 100
```
Stop when Cd/Cl diverge or NaN appears — that is the practical ceiling.

**Step 3: Document maximum stable Re in README**

Update `fd_ns/README.md` (one sentence: maximum Re achieved and grid used).

**Step 4: Commit**

```bash
git add assignment3/fd_ns/README.md
git commit -m "docs(fd): document maximum stable Re and benchmark results"
```

---

## Summary

| Task | Deliverable | Key reuse from assignment2 |
|------|-------------|---------------------------|
| 1 | Makefile + scaffold | Makefile flags verbatim |
| 2 | MAC grid struct | Eigen MatrixXd pattern |
| 3 | Boundary conditions | — |
| 4 | Pressure SOR | `sor.cpp` sweep loop |
| 5 | Fractional-step solver | — |
| 6 | I/O (CSV + vorticity) | — |
| 7 | Main driver + CLI | `argv` parsing style |
| 8 | Re=100 validation | — |
| 9 | Max-Re sweep | — |

**CFL guidance for high Re:**

| Re | Recommended grid | dt |
|----|------------------|----|
| 100 | 440 × 82 | 0.001 |
| 400 | 440 × 82 | 0.001 |
| 1000 | 880 × 164 | 0.0005 |
| 2000+ | 880 × 164 | 0.0002 |

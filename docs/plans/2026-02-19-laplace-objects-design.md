# Design: Laplace Solver with Objects — Assignment Set 1, K & L

**Date:** 2026-02-19
**Assignments:** K (2 pts) — Sinks, L (1 pt) — Insulators
**Language:** C++ (extending existing `src/laplace.cpp`)

---

## Problem Summary

Extend the existing 2D Laplace/SOR solver to support interior objects:

- **K:** Objects as **sinks** — concentration fixed at 0 inside the object (interior Dirichlet BC). Study effects on iteration count and optimal ω.
- **L:** Objects as **insulators** — no diffusion flux through the object (Neumann zero-flux BC, ∂c/∂n = 0). Study changes in time evolution and final steady state.

---

## Theory

### Laplace Equation

Steady-state diffusion (eq. 10 in assignment):

```
∇²c = 0
```

Discretised with the 5-point stencil (eq. 11):

```
c(i,j) = (1/4) * [c(i+1,j) + c(i-1,j) + c(i,j+1) + c(i,j-1)]
```

### SOR Update (eq. from assignment section 1.6)

```
c_new(i,j) = (ω/4) * [c(i+1,j) + c(i-1,j) + c(i,j+1) + c(i,j-1)]
             + (1 - ω) * c_old(i,j)
```

SOR converges for 0 < ω < 2. Optimal ω ∈ (1.7, 2.0) for this domain; exact value depends on grid size N.

### K — Interior Dirichlet (Sink)

A sink cell has a fixed value c = 0. During the SOR sweep, if cell (i,j) is a sink:
- Skip the update entirely
- Ensure c(i,j) = 0 remains enforced

This is equivalent to an additional boundary condition inside the domain.

**References:**
- Leveque, R.J. *Finite Difference Methods for Ordinary and Partial Differential Equations*, SIAM, 2007. Chapter 2.
- Assignment document, Section 1.6 and hint for K.

### L — Interior Neumann (Insulator, ∂c/∂n = 0)

An insulating cell blocks all flux through it. At the interface between a regular cell (i,j) and an insulating neighbour, the zero-flux condition is enforced via the ghost-node method:

- Replace each insulating neighbour's contribution in the stencil with the current cell's own value c(i,j).
- This makes the finite difference approximation of the normal derivative zero: (c(i,j) - c(i,j)) / δx = 0.

Modified stencil for cell (i,j) adjacent to insulators:

```
c_n = (mask(i+1,j) == INSULATOR) ? c(i,j) : c(i+1,j)
c_s = (mask(i-1,j) == INSULATOR) ? c(i,j) : c(i-1,j)
c_e = (mask(i,j+1) == INSULATOR) ? c(i,j) : c(i,j+1)
c_w = (mask(i,j-1) == INSULATOR) ? c(i,j) : c(i,j-1)
c_new(i,j) = ω/4 * (c_n + c_s + c_e + c_w) + (1-ω) * c(i,j)
```

**Expected final state:** Concentration isocontours wrap around the insulator and are perpendicular to its surface. No zero-concentration shadow (contrast with sinks).

**References:**
- Strikwerda, J.C. *Finite Difference Schemes and Partial Differential Equations*, SIAM, 2004. Chapter 1.
- Ghost-node / image-point method for Neumann BCs: standard in computational PDE literature.

---

## Architecture

### Files Modified

| File | Change |
|---|---|
| `src/laplace.hpp` | Add `CellType` enum, `Rect` struct, `fill_mask()` declaration, `SorResult` struct, `sor_solve()` declaration |
| `src/laplace.cpp` | Implement `fill_mask()` and `sor_solve()` |
| `src/laplace_main.cpp` | Add sink and insulator examples; write outputs for each case |
| `scripts/plot_laplace.py` | Add plots for object cases (heatmaps with object outlines) |
| `Makefile` | Ensure new outputs are handled |

Existing Jacobi code is untouched.

---

## Data Structures

```cpp
// Cell classification
enum class CellType { NORMAL = 0, SINK = 1, INSULATOR = 2 };

// Axis-aligned rectangle for object placement (grid indices, inclusive)
struct Rect { int x0, y0, x1, y1; };

// Result from SOR solve
struct SorResult {
    Eigen::MatrixXd solution;
    int iterations;
    double final_residual;
    std::vector<double> deltas;
};

// Fill a mask matrix with rectangles of a given type
void fill_mask(Eigen::MatrixXi& mask, std::vector<Rect> rects, CellType type);

// SOR solver with mask support
SorResult sor_solve(int N, double omega, double tolerance, int max_iter,
                    const Eigen::MatrixXi& mask,
                    const std::function<void(Eigen::MatrixXd&)>& apply_boundary);
```

The mask is an `Eigen::MatrixXi` of the same dimensions as the solution grid. Value 0/1/2 maps to NORMAL/SINK/INSULATOR.

---

## SOR Loop Logic

```
for each interior row i (1 to N-2):
    for each interior col j (0 to N-2, with periodic wrap):
        if mask(i,j) == SINK:
            c(i,j) = 0.0   // enforce, skip SOR
            continue
        if mask(i,j) == INSULATOR:
            continue        // not solved, not modified
        // Resolve each neighbour (replace insulator with own value)
        cn = mask(i+1,j)==INSULATOR ? c(i,j) : c(i+1,j)
        cs = mask(i-1,j)==INSULATOR ? c(i,j) : c(i-1,j)
        ce = mask(i,jp) ==INSULATOR ? c(i,j) : c(i,jp)
        cw = mask(i,jm) ==INSULATOR ? c(i,j) : c(i,jm)
        c_new = omega/4 * (cn+cs+ce+cw) + (1-omega)*c(i,j)
        delta = max(delta, |c_new - c(i,j)|)
        c(i,j) = c_new
```

SOR is in-place (single grid), unlike Jacobi which needs two grids.

---

## Outputs

Each case writes to `output/`:

| File | Content |
|---|---|
| `laplace_sink_solution.txt` | Concentration field with sink objects |
| `laplace_sink_deltas.txt` | Convergence history for sink case |
| `laplace_insulator_solution.txt` | Concentration field with insulator objects |
| `laplace_insulator_deltas.txt` | Convergence history for insulator case |
| `laplace_omega_sweep.txt` | Iteration counts for ω ∈ [1.0, 1.99] (for K's ω analysis) |

---

## Analysis to Report (per assignment)

**K:**
- Concentration heatmap with sink outline overlaid
- Iteration count: baseline vs. with sink(s)
- ω sweep plot: iteration count vs. ω for no-object and with-object cases
- Interpretation: how does the sink affect optimal ω and the concentration field?

**L:**
- Concentration heatmap with insulator outline overlaid
- Comparison of final state: sink vs. insulator (different shadow/wrapping behaviour)
- Qualitative answer: what changes in time evolution and final state?

---

## C++ Concepts Introduced

| Concept | Where used |
|---|---|
| `enum class` | `CellType` — strongly typed, no accidental int comparison |
| `struct` | `Rect`, `SorResult` — plain data aggregates |
| `std::vector<Rect>` | Object list passed to `fill_mask()` |
| `Eigen::MatrixXi` | Integer mask matrix, same API as `MatrixXd` |
| In-place update | SOR modifies single grid; contrast with Jacobi's two-grid swap |
| Ternary operator | `mask(i+1,j)==INSULATOR ? c(i,j) : c(i+1,j)` — concise neighbour selection |

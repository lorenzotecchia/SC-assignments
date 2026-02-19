# Laplace Objects (Sinks & Insulators) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the C++ Laplace solver with SOR + interior objects: sinks (c=0, assignment K) and insulators (∂c/∂n=0, assignment L).

**Architecture:** Add `CellType` enum, `Rect` struct, `fill_mask()`, and `sor_solve()` to the existing `laplace.hpp/.cpp`. Extend `laplace_main.cpp` with three scenarios (baseline, sinks, insulator). Extend `scripts/plot_laplace.py` to visualise object results.

**Tech Stack:** C++20, Eigen3 (`/opt/homebrew/include/eigen3`), g++-15 with -fopenmp, Python/matplotlib for plots.

---

## Grid conventions (READ THIS FIRST)

The existing code uses an **N × N** Eigen matrix where:
- **Row index i** = y-coordinate: row 0 = bottom (c=0), row N-1 = top (c=1)
- **Col index j** = x-coordinate: periodic (j=0 wraps to j=N-2)
- Interior rows: i = 1 … N-2
- Interior cols: j = 0 … N-2, with `jm = (j-1+(N-1)) % (N-1)` and `jp = (j+1) % (N-1)`
- After every sweep, copy: `grid(i, N-1) = grid(i, 0)` for all interior i (enforces periodicity)

SOR is **in-place** (one grid), unlike Jacobi (two grids). Each cell update immediately influences subsequent cells in the same sweep.

---

## Task 1: Add types and declarations to `laplace.hpp`

**Files:**
- Modify: `src/laplace.hpp`

**What to add** — paste these additions at the top of the file (after the existing `#pragma once` and includes):

```cpp
#pragma once
#include <Eigen/Dense>
#include <functional>
#include <vector>

// ── Existing struct (unchanged) ─────────────────────────────────────────────
struct JacobiResult {
  Eigen::MatrixXd solution;
  int iterations;
  double final_residual;
  std::vector<double> deltas;
};

JacobiResult
jacobi_solve(int N, double tolerance, int max_iter,
             const std::function<void(Eigen::MatrixXd &)> &apply_boundary);

// ── New additions for K & L ──────────────────────────────────────────────────

// Cell type: NORMAL is solved by SOR; SINK is forced to 0; INSULATOR blocks flux.
enum class CellType { NORMAL = 0, SINK = 1, INSULATOR = 2 };

// Axis-aligned rectangle in grid-index coordinates (inclusive on all sides).
struct Rect {
  int x0, y0, x1, y1;
};

// Fill all cells inside each rectangle with the given CellType.
void fill_mask(Eigen::MatrixXi &mask, std::vector<Rect> rects, CellType type);

// Result from the SOR solver (mirrors JacobiResult).
struct SorResult {
  Eigen::MatrixXd solution;
  int iterations;
  double final_residual;
  std::vector<double> deltas;
};

// SOR solver with object mask support.
// omega: relaxation parameter (1 < omega < 2 for over-relaxation).
// mask:  N×N integer matrix; 0=NORMAL, 1=SINK, 2=INSULATOR.
SorResult
sor_solve(int N, double omega, double tolerance, int max_iter,
          const Eigen::MatrixXi &mask,
          const std::function<void(Eigen::MatrixXd &)> &apply_boundary);
```

**Step 1: Open the file and replace its entire content** with the block above.

**Step 2: Build to check the header compiles**

```bash
g++-15 -std=c++20 -I/opt/homebrew/include/eigen3 -Isrc -c src/laplace.cpp -o /dev/null
```

Expected: no errors (the .cpp doesn't implement the new functions yet, so you'll get linker errors if you try to link — that's fine for now).

**Step 3: Commit**

```bash
git add src/laplace.hpp
git commit -m "add CellType, Rect, SorResult, sor_solve declarations to laplace.hpp"
```

---

## Task 2: Implement `fill_mask()` in `laplace.cpp`

**Files:**
- Modify: `src/laplace.cpp`

**What to add** — append after the closing `}` of `jacobi_solve`:

```cpp
void fill_mask(Eigen::MatrixXi &mask, std::vector<Rect> rects, CellType type) {
  // Cast enum to int so Eigen can store it in MatrixXi
  int val = static_cast<int>(type);
  for (const Rect &r : rects) {
    for (int i = r.y0; i <= r.y1; ++i) {
      for (int j = r.x0; j <= r.x1; ++j) {
        mask(i, j) = val;
      }
    }
  }
}
```

**C++ note — `static_cast<int>(type)`:** `enum class` values cannot be implicitly converted to `int`. `static_cast` makes the conversion explicit and safe. Eigen's `MatrixXi` stores `int`, so we need this conversion.

**Step 1: Add the function** to `src/laplace.cpp`.

**Step 2: Build to verify it compiles**

```bash
g++-15 -std=c++20 -I/opt/homebrew/include/eigen3 -Isrc -c src/laplace.cpp -o /dev/null
```

Expected: compiles with no errors.

**Step 3: Commit**

```bash
git add src/laplace.cpp
git commit -m "implement fill_mask() for sink/insulator object placement"
```

---

## Task 3: Implement `sor_solve()` in `laplace.cpp`

**Files:**
- Modify: `src/laplace.cpp`

**What to add** — append after `fill_mask()`:

```cpp
SorResult
sor_solve(int N, double omega, double tolerance, int max_iter,
          const Eigen::MatrixXi &mask,
          const std::function<void(Eigen::MatrixXd &)> &apply_boundary) {

  // One grid (SOR is in-place, unlike Jacobi which needs two).
  Eigen::MatrixXd grid = Eigen::MatrixXd::Zero(N, N);
  apply_boundary(grid);

  // Enforce sinks to 0 in the initial state.
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      if (mask(i, j) == static_cast<int>(CellType::SINK))
        grid(i, j) = 0.0;

  int iter = 0;
  double diff = 0.0;
  std::vector<double> deltas;
  deltas.reserve(max_iter);

  for (iter = 0; iter < max_iter; ++iter) {
    diff = 0.0;

    for (int i = 1; i < N - 1; ++i) {
      for (int j = 0; j < N - 1; ++j) {
        // ── Skip non-normal cells ────────────────────────────────────────────
        if (mask(i, j) == static_cast<int>(CellType::SINK)) {
          grid(i, j) = 0.0; // enforce sink (in case BCs reset it)
          continue;
        }
        if (mask(i, j) == static_cast<int>(CellType::INSULATOR)) {
          continue; // insulator is not solved
        }

        // ── Periodic x-neighbours ───────────────────────────────────────────
        int jm = (j - 1 + (N - 1)) % (N - 1); // left  (wraps)
        int jp = (j + 1) % (N - 1);            // right (wraps)

        // ── Neighbour values: replace insulator with own value (zero-flux) ──
        // This enforces ∂c/∂n = 0 at the insulator surface.
        double c_north = (mask(i + 1, j) == static_cast<int>(CellType::INSULATOR))
                             ? grid(i, j) : grid(i + 1, j);
        double c_south = (mask(i - 1, j) == static_cast<int>(CellType::INSULATOR))
                             ? grid(i, j) : grid(i - 1, j);
        double c_east  = (mask(i, jp) == static_cast<int>(CellType::INSULATOR))
                             ? grid(i, j) : grid(i, jp);
        double c_west  = (mask(i, jm) == static_cast<int>(CellType::INSULATOR))
                             ? grid(i, j) : grid(i, jm);

        // ── SOR update ───────────────────────────────────────────────────────
        // omega=1 → Gauss-Seidel; 1<omega<2 → over-relaxation (faster).
        double old_val = grid(i, j);
        double avg = 0.25 * (c_north + c_south + c_east + c_west);
        grid(i, j) = omega * avg + (1.0 - omega) * old_val;

        double local_diff = std::abs(grid(i, j) - old_val);
        if (local_diff > diff) diff = local_diff;
      }
    }

    // Enforce periodicity: last column mirrors first column.
    for (int i = 1; i < N - 1; ++i)
      grid(i, N - 1) = grid(i, 0);

    deltas.push_back(diff);
    if (diff < tolerance) break;
  }

  return SorResult{grid, iter, diff, std::move(deltas)};
}
```

**C++ notes:**
- `std::move(deltas)` transfers ownership of the vector instead of copying it — avoids an O(N) copy at return.
- The ternary `condition ? a : b` evaluates to `a` if condition is true, else `b`. Used here to select the insulator mirror value in one line.
- `++iter` (pre-increment) is preferred over `iter++` in C++ loops — same effect but skips a useless temporary copy.

**Step 1: Add the function** to `src/laplace.cpp`.

**Step 2: Full build to check it links**

```bash
make build-laplace
```

Expected: binary `laplace_sim` created with no errors.

**Step 3: Commit**

```bash
git add src/laplace.cpp
git commit -m "implement sor_solve() with sink and insulator mask support"
```

---

## Task 4: Extend `laplace_main.cpp` — three scenarios

**Files:**
- Modify: `src/laplace_main.cpp`

Replace the entire file with:

```cpp
#include "laplace.hpp"
#include <fstream>
#include <iostream>

// Helper: write a MatrixXd to a text file (space-separated, one row per line).
static void write_matrix(const std::string &path, const Eigen::MatrixXd &m) {
  std::ofstream out(path);
  out << m << "\n";
}

// Helper: write a vector of doubles to a text file (one value per line).
static void write_deltas(const std::string &path,
                         const std::vector<double> &deltas) {
  std::ofstream out(path);
  for (double d : deltas) out << d << "\n";
}

int main() {
  const int    N         = 50;
  const double tolerance = 1e-5;
  const int    max_iter  = 100000;
  const double omega     = 1.95; // good default for N=50

  // Boundary conditions: bottom row c=0, top row c=1, periodic in x.
  auto apply_boundary = [](Eigen::MatrixXd &grid) {
    int n = grid.rows();
    grid.row(0).setZero();
    grid.row(n - 1).setConstant(1.0);
  };

  // ── 1. Baseline Jacobi (existing, unchanged) ────────────────────────────
  auto jres = jacobi_solve(N, tolerance, max_iter, apply_boundary);
  std::cout << "Jacobi: " << jres.iterations << " iters, "
            << "residual=" << jres.final_residual << "\n";
  write_matrix("output/laplace_solution.txt", jres.solution);
  write_deltas("output/laplace_deltas.txt", jres.deltas);

  // ── 2. Baseline SOR (no objects) ────────────────────────────────────────
  Eigen::MatrixXi mask_empty = Eigen::MatrixXi::Zero(N, N);
  auto sres = sor_solve(N, omega, tolerance, max_iter, mask_empty, apply_boundary);
  std::cout << "SOR (no objects): " << sres.iterations << " iters, "
            << "residual=" << sres.final_residual << "\n";
  write_matrix("output/laplace_sor_baseline.txt", sres.solution);
  write_deltas("output/laplace_sor_baseline_deltas.txt", sres.deltas);

  // ── 3. K: Sinks ─────────────────────────────────────────────────────────
  // Two rectangular sinks: a large one and a smaller offset one.
  Eigen::MatrixXi mask_sink = Eigen::MatrixXi::Zero(N, N);
  fill_mask(mask_sink,
            {{10, 10, 20, 20},   // Rect{x0=10, y0=10, x1=20, y1=20}
             {30, 25, 45, 35}},  // second rectangle
            CellType::SINK);

  auto kres = sor_solve(N, omega, tolerance, max_iter, mask_sink, apply_boundary);
  std::cout << "SOR (sinks): " << kres.iterations << " iters, "
            << "residual=" << kres.final_residual << "\n";
  write_matrix("output/laplace_sink.txt", kres.solution);
  write_deltas("output/laplace_sink_deltas.txt", kres.deltas);

  // Write mask so Python can draw object outlines.
  {
    std::ofstream out("output/laplace_sink_mask.txt");
    out << mask_sink << "\n";
  }

  // Omega sweep for K: measure iteration count vs omega with sinks present.
  {
    std::ofstream out("output/laplace_omega_sweep.txt");
    out << "omega iterations_no_obj iterations_sink\n";
    for (double w = 1.0; w < 2.0; w += 0.05) {
      auto r_empty = sor_solve(N, w, tolerance, max_iter, mask_empty, apply_boundary);
      auto r_sink  = sor_solve(N, w, tolerance, max_iter, mask_sink,  apply_boundary);
      out << w << " " << r_empty.iterations << " " << r_sink.iterations << "\n";
    }
    std::cout << "Omega sweep written.\n";
  }

  // ── 4. L: Insulator ─────────────────────────────────────────────────────
  Eigen::MatrixXi mask_ins = Eigen::MatrixXi::Zero(N, N);
  fill_mask(mask_ins,
            {{20, 15, 30, 35}},  // one rectangle as insulator
            CellType::INSULATOR);

  auto lres = sor_solve(N, omega, tolerance, max_iter, mask_ins, apply_boundary);
  std::cout << "SOR (insulator): " << lres.iterations << " iters, "
            << "residual=" << lres.final_residual << "\n";
  write_matrix("output/laplace_insulator.txt", lres.solution);
  write_deltas("output/laplace_insulator_deltas.txt", lres.deltas);
  {
    std::ofstream out("output/laplace_insulator_mask.txt");
    out << mask_ins << "\n";
  }

  return 0;
}
```

**C++ notes for the new patterns here:**
- `static void write_matrix(...)` — `static` on a file-scope function means "private to this translation unit" (this .cpp file). It won't conflict with any same-named function in another file.
- `{{10, 10, 20, 20}, {30, 25, 45, 35}}` — brace-initializer list for `std::vector<Rect>`. Each inner `{...}` constructs a `Rect` by initializing its fields in order: `x0, y0, x1, y1`.
- `[](Eigen::MatrixXd &grid) { ... }` — a **lambda**, i.e. an anonymous function defined inline. It's passed as `apply_boundary` to both solvers.

**Step 1: Replace `src/laplace_main.cpp`** with the code above.

**Step 2: Build**

```bash
make build-laplace
```

Expected: `laplace_sim` binary created, no errors.

**Step 3: Run and check output files exist**

```bash
./laplace_sim
ls output/laplace_sink.txt output/laplace_insulator.txt output/laplace_omega_sweep.txt
```

Expected output (approximate):
```
Jacobi: XXXX iters, residual=...
SOR (no objects): YYY iters, residual=...
SOR (sinks): ZZZ iters, residual=...
Omega sweep written.
SOR (insulator): WWW iters, residual=...
```
SOR should converge in far fewer iterations than Jacobi.

**Validation check — sinks must be 0:**
Open `output/laplace_sink.txt` and spot-check that the rows/cols corresponding to the sink rectangles contain `0`. (You can use Python: `import numpy as np; g=np.loadtxt('output/laplace_sink.txt'); print(g[10:21, 10:21])` — should be all zeros.)

**Step 4: Commit**

```bash
git add src/laplace_main.cpp
git commit -m "extend laplace_main: SOR baseline, K sinks with omega sweep, L insulator"
```

---

## Task 5: Extend `scripts/plot_laplace.py` for K & L

**Files:**
- Modify: `scripts/plot_laplace.py`

Append the following **after** the existing `plt.show()` at the end of the file:

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load new outputs ─────────────────────────────────────────────────────────
sor_base    = np.loadtxt("output/laplace_sor_baseline.txt")
sink_grid   = np.loadtxt("output/laplace_sink.txt")
sink_mask   = np.loadtxt("output/laplace_sink_mask.txt")
ins_grid    = np.loadtxt("output/laplace_insulator.txt")
ins_mask    = np.loadtxt("output/laplace_insulator_mask.txt")

# omega sweep: columns are [omega, iters_no_obj, iters_sink]
sweep = np.loadtxt("output/laplace_omega_sweep.txt", skiprows=1)
omega_vals, iters_empty, iters_sink = sweep[:, 0], sweep[:, 1], sweep[:, 2]

N = sink_grid.shape[0]

# ── Figure K-1: Concentration heatmaps (baseline vs sinks vs insulator) ─────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
datasets = [
    (sor_base, np.zeros((N, N)), "SOR baseline (no objects)"),
    (sink_grid, sink_mask,       "SOR with sinks (K)"),
    (ins_grid,  ins_mask,        "SOR with insulator (L)"),
]
for ax, (grid, mask, title) in zip(axes, datasets):
    im = ax.imshow(grid, cmap="viridis", origin="lower", extent=[0, 1, 0, 1],
                   vmin=0, vmax=1)
    # Overlay object outlines (white contour where mask != 0)
    if mask.any():
        ax.contour(mask, levels=[0.5], colors="white", linewidths=1.5,
                   extent=[0, 1, 0, 1], origin="lower")
    fig.colorbar(im, ax=ax, label=r"$c(x,y)$", shrink=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title)
    ax.set_aspect("equal")
fig.suptitle("Laplace steady-state solutions with objects", fontsize=13, y=1.02)
fig.tight_layout()
plt.savefig("output/plots/laplace_objects_heatmaps.eps", format="eps")
plt.show()

# ── Figure K-2: Omega sweep ──────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(omega_vals, iters_empty, "o-", label="No objects", color="#2196F3")
ax2.plot(omega_vals, iters_sink,  "s-", label="With sinks", color="#FF5722")
ax2.set_xlabel(r"$\omega$")
ax2.set_ylabel("Iterations to convergence")
ax2.set_title(r"Effect of $\omega$ on convergence (K)")
ax2.legend(); ax2.grid(True, alpha=0.3)
fig2.tight_layout()
plt.savefig("output/plots/laplace_omega_sweep.eps", format="eps")
plt.show()
```

**Step 1: Append the code above** to `scripts/plot_laplace.py`.

**Step 2: Run the plot script**

```bash
python3 scripts/plot_laplace.py
```

Expected: three windows appear in sequence (existing plots, then new heatmap comparison, then omega sweep). White outlines should appear on the object cells. Sink grid should show a clearly zero region at the sink positions. Insulator grid should show concentration "wrapping around" the object.

**Step 3: Commit**

```bash
git add scripts/plot_laplace.py
git commit -m "extend plot_laplace: K/L heatmaps with object outlines and omega sweep plot"
```

---

## Task 6: Update Makefile to track new outputs

**Files:**
- Modify: `Makefile`

The `run-laplace` target currently only depends on `laplace_solution.txt`. Add the new outputs so `make laplace` reruns correctly:

Find this block in the Makefile:
```make
$(OUT_DIR)/laplace_solution.txt: $(LAPLACE_BIN) | $(OUT_DIR)
	./$(LAPLACE_BIN)

run-laplace: $(OUT_DIR)/laplace_solution.txt
```

Replace with:
```make
LAPLACE_OUTPUTS := $(OUT_DIR)/laplace_solution.txt \
                   $(OUT_DIR)/laplace_sink.txt \
                   $(OUT_DIR)/laplace_insulator.txt \
                   $(OUT_DIR)/laplace_omega_sweep.txt

$(LAPLACE_OUTPUTS): $(LAPLACE_BIN) | $(OUT_DIR)
	./$(LAPLACE_BIN)

run-laplace: $(LAPLACE_OUTPUTS)
```

**Step 1: Edit the Makefile** as shown above.

**Step 2: Verify full pipeline works**

```bash
make clean && make laplace
```

Expected: compiles → runs `laplace_sim` (prints iteration counts) → runs `plot_laplace.py` (shows all figures). No errors.

**Step 3: Commit**

```bash
git add Makefile
git commit -m "update Makefile: track all laplace output files for K and L"
```

---

## Verification checklist

After all tasks are done, confirm:

- [ ] `make build-laplace` succeeds
- [ ] `./laplace_sim` prints 4 lines (Jacobi, SOR baseline, SOR sinks, SOR insulator)
- [ ] `output/laplace_sink.txt` exists and sink region cells are 0
- [ ] `output/laplace_insulator.txt` exists and concentration wraps around the insulator
- [ ] `output/laplace_omega_sweep.txt` has ~20 rows of omega/iteration data
- [ ] `python3 scripts/plot_laplace.py` produces heatmap and omega-sweep figures
- [ ] `make clean && make laplace` runs the full pipeline end to end

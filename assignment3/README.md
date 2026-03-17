# Assignment 3

Kármán vortex street simulations for the 2026 Scientific Computing course.
Two independent solvers — a GPU-accelerated Lattice Boltzmann Method and a finite-element Navier-Stokes solver — both targeting the same benchmark flow.

---

## How to run

Install dependencies with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

### LBM solver (Taichi, interactive GUI)

```bash
# Kármán Vortex Street — default Re=400
uv run python taichi_lbm.py 0

# Kármán Vortex Street — custom Reynolds number
uv run python taichi_lbm.py 0 200

# Lid-driven Cavity Flow — Re=1000
uv run python taichi_lbm.py 1
```

Press `ESC` or close the window to exit.

### NGSolve FEM solver (headless, batch output)

```bash
# default Re=100, dt=0.0005, tend=8.0
uv run python vortex.py

# custom parameters
uv run python vortex.py --Re 200 --dt 0.0005 --tend 8.0 --order 2 --maxh 0.07
uv run python vortex.py --help   # full option list
```

Output is written to `karman_output/` (VTK time series, drag/lift CSV, and plots).
Open the result in ParaView:

```bash
paraview karman_output/karman.pvd
```

---

## Repository structure

```
.
├── taichi_lbm.py       # D2Q9 LBM solver — GPU via Taichi, interactive visualisation
├── vortex.py           # Taylor-Hood FEM solver — NGSolve, headless batch run
├── data/
│   └── ghia1982.dat    # Ghia et al. (1982) reference data for lid-driven cavity validation
├── docs/
│   └── reference.md    # upstream Taichi LBM README (original source / theory)
├── karman_output/      # FEM solver output: *.vtu, *.pvd, drag_lift.csv, drag.png, lift.png
├── pyproject.toml
└── uv.lock
```

### `taichi_lbm.py` — Lattice Boltzmann Method

| Case | `flow_case` | Grid | Re |
|---|---|---|---|
| Kármán Vortex Street | `0` | 801 × 201 | configurable (default 400) |
| Lid-driven Cavity | `1` | 256 × 256 | 1000 |

### `vortex.py` — NGSolve FEM (Navier-Stokes)

Taylor-Hood P(k)/P(k-1) elements, IMEX-1 time integration, Schäfer-Turek benchmark geometry (2.2 × 0.41 m channel, cylinder diameter 0.1 m at (0.2, 0.2)).

| Option | Default | Description |
|---|---|---|
| `--Re` | 100 | Reynolds number |
| `--dt` | 0.0005 | Time step (s) |
| `--tend` | 8.0 | End time (s) |
| `--order` | 2 | FE polynomial order for velocity |
| `--maxh` | 0.07 | Max mesh element size |
| `--vtk-interval` | 50 | Write VTK every N steps |

---

## Dependencies

- **Python ≥ 3.12** managed with [uv](https://github.com/astral-sh/uv)
- `taichi ≥ 1.7.4` — GPU kernel compilation (CUDA / Metal / Vulkan)
- `ngsolve ≥ 6.2` — finite element framework
- `matplotlib ≥ 3.10`
- **ParaView** — required to visualise the VTK output of `vortex.py` (`karman_output/*.pvd`). Install from [paraview.org](https://www.paraview.org/download/) or via your system package manager (`brew install paraview`, `apt install paraview`, etc.). Not managed by uv.

---

## Contributors

[![git-fame](https://git-fame.cdcl.ml/gh/lorenzotecchia/SC-assignments)](https://github.com/lorenzotecchia/SC-assignments)

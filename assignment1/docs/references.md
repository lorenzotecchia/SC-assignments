# References — Laplace/Diffusion Simulations with Sinks & Insulators

---

## 1. Most Relevant: Embedded Objects, Sinks & Insulators with Code

These sources deal directly with placing internal objects (fixed-value sinks or
zero-flux insulators) inside a 2D finite-difference Laplace/Poisson grid.

| Source | Why it is relevant |
|--------|-------------------|
| [Elliptic PDEs — Barba Group (Essential Skills RRC)](https://barbagroup.github.io/essential_skills_RRC/laplace/1/) | Full Python Jupyter tutorial: 2D Laplace with Jacobi iteration, ghost-point Neumann BCs, convergence. The closest match to this project's structure. |
| [GitHub — gpollo/laplace: Numerical solver of the 2D Laplace equation](https://github.com/gpollo/laplace) | C++ Laplace solver with configurable boundary regions, directly comparable to the `laplace.cpp` + mask approach here. |
| [GitHub — zaman13/Poisson-solver-2D](https://github.com/zaman13/Poisson-solver-2D) | Python finite-difference solver that explicitly supports **interior Dirichlet/Neumann conditions** (embedded objects inside the domain), with code examples. |
| [Difference Approximations of the Neumann Problem — LLNL (PDF)](https://computing.llnl.gov/sites/default/files/KPY_Neumann.pdf) | Rigorous derivation of ghost-point and one-sided difference approximations for Neumann BCs — exactly the `c_north = grid(i,j)` self-reflection trick in `sor_solve()`. |
| [Insulated Boundary Conditions — Ohio University Lecture 38 (PDF)](http://www.ohiouniversityfaculty.com/youngt/IntNumMeth/lecture38.pdf) | Step-by-step finite-difference stencil for an insulated (zero-flux) boundary, including corner cells. Directly maps to the insulator mask handling in the code. |
| [How is an insulated boundary handled in FD? — Vaia / Cengel](https://www.vaia.com/en-us/textbooks/physics/heat-and-mass-transfer-fundamentals-and-applications-5-edition/chapter-5/problem-16-how-is-an-insulated-boundary-handled-in-finite-di/) | Short textbook answer: zero-flux Neumann as mirror/ghost-point substitution. |
| [2D Steady-State FD: Chapter 4 — VisualSlope (PDF)](https://www.visualslope.com/Library/FDM-for-heat-transfering.pdf) | Textbook treatment of 2D conduction FD including internal nodes, surface BCs, and corner stencils. |
| [Boundary Value Problems — MOOC notebook](https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/03_FiniteDifferences/03_03_BoundaryValueProblems.html) | Interactive Python notebook: Dirichlet and Neumann discretisation side by side, with numpy code. |
| [Solving a discrete BVP — SciPy Cookbook](https://scipy-cookbook.readthedocs.io/items/discrete_bvp.html) | Shows how to assemble the finite-difference matrix for Laplace with mixed BCs using `scipy.sparse`. |
| [Stanford CS205b Lecture 16 — Incompressible Flow / Laplace (PDF)](https://web.stanford.edu/class/cs205b/lectures/lecture16.pdf) | Covers MAC grid, staggered-grid Laplace, and how to handle solid (insulator-like) obstacles in a computational domain. |

---

## 2. Iterative Solvers (Jacobi / Gauss-Seidel / SOR)

| Source | Notes |
|--------|-------|
| [Successive over-relaxation — Wikipedia](https://en.wikipedia.org/wiki/Successive_over-relaxation) | Derivation of optimal ω, spectral radius, convergence bound. |
| [Gauss–Seidel method — Wikipedia](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method) | |
| [Iterative Methods for 2D Laplace — MOOC notebook](https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/05_IterativeMethods/05_01_Iteration_and_2D.html) | Python: Jacobi → GS → SOR on 2D grid, convergence plots. |
| [Iterative Methods — Mathematics LibreTexts (Chasnov)](https://math.libretexts.org/Bookshelves/Scientific_Computing_Simulations_and_Modeling/Scientific_Computing_(Chasnov)/I:_Numerical_Methods/7:_Iterative_Methods) | Textbook level, includes optimal ω formula. |
| [NTNU Iterative Methods for Elliptic PDEs](https://leifh.folk.ntnu.no/teaching/tkt4140/._main057.html) | Jacobi / GS / SOR with worked equations and convergence criteria. |
| [Neumann BC implementation for SOR — NTNU](https://folk.ntnu.no/leifh/teaching/tkt4140/._main056.html) | How to modify the SOR update stencil at a Neumann boundary. |
| [COMSOL SOR algorithm reference](https://doc.comsol.com/5.5/doc/com.comsol.help.comsol/comsol_ref_solver.27.129.html) | Industrial reference for SOR with mixed BCs. |

---

## 3. Boundary Conditions — Theory

| Source | Notes |
|--------|-------|
| [Neumann boundary condition — Wikipedia](https://en.wikipedia.org/wiki/Neumann_boundary_condition) | Canonical reference for zero-flux (insulator) BCs. |
| [Robin boundary condition — Wikipedia](https://en.wikipedia.org/wiki/Robin_boundary_condition) | Mixed Dirichlet + Neumann; useful if the sink has partial absorption. |
| [UBC Finite Difference Methods notes (PDF)](https://personal.math.ubc.ca/~gustaf/M31611/fd.pdf) | How Dirichlet and Neumann modify the linear system and stencil. |
| [NTNU Numerical PDE notes (PDF)](https://wiki.math.ntnu.no/_media/tma4125/2023v/numpde.pdf) | Comprehensive: BCs, stability, convergence for elliptic and parabolic PDEs. |
| [Finite Difference Methods for Poisson — UC Irvine (PDF)](https://www.math.uci.edu/~chenlong/226/FDM.pdf) | Error analysis, ghost cells, second-order Neumann. |

---

## 4. Background: Laplace / Diffusion Equation

| Source | Notes |
|--------|-------|
| [MIT OCW — Heat and wave equations in 2D/3D (PDF)](https://ocw.mit.edu/courses/18-303-linear-partial-differential-equations-fall-2006/5faa7f7c21719c38fb701bc79ef4a29f_pde3d.pdf) | Analytical solutions and separation of variables. |
| [CMU FDM for Laplace Equation (PDF)](https://www.andrew.cmu.edu/course/24-681/handouts/lectures/fdm_for_laplace_equation.pdf) | Finite difference derivation for 2D Laplace, matrix form. |
| [University of Warwick — Diffusion equation FD notes (PDF)](https://warwick.ac.uk/fac/cross_fac/complexity/study/msc_and_phd/co906/co906online/lecturenotes_2009/chap3.pdf) | FTCS, stability, boundary conditions. |
| [2D Heat equation with FD — UT Austin (PDF)](https://www-udc.ig.utexas.edu/external/becker/teaching/557/problem_sets/problem_set_fd_2dheat.pdf) | Problem set with worked solutions. |
| [NASA Introduction to Numerical Methods in Heat Transfer (PDF)](https://ntrs.nasa.gov/api/citations/20200006182/downloads/Introduction%20to%20Numerical%20Methods%20in%20Heat%20Transfer.pdf) | Engineering perspective: sinks, insulators, convective BCs. |
| [Monte Carlo 2D Laplace simulation — IIETA](https://iieta.org/journals/mmep/paper/10.18280/mmep.120118) | Alternative (stochastic) approach to the same problem. |

---

## 5. Code Repositories

| Repo | Language | What it shows |
|------|----------|---------------|
| [barbagroup/essential\_skills\_RRC](https://github.com/barbagroup/essential_skills_RRC/blob/master/docs/laplace/1.md) | Python / NumPy | Jacobi 2D Laplace, Neumann ghost points |
| [gpollo/laplace](https://github.com/gpollo/laplace) | C++ | Configurable region BCs, close to this project |
| [zaman13/Poisson-solver-2D](https://github.com/zaman13/Poisson-solver-2D) | Python | Interior Dirichlet + Neumann objects in a 2D grid |
| [wiseodd/laplace-numpy (Gist)](https://gist.github.com/wiseodd/c08d5a2b02b1957a16f886ab7044032d) | Python / NumPy | Minimal Jacobi Laplace implementation |
| [2D FDM steady-state — MATLAB File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/55058-2d-heat-equation-using-finite-difference-method-with-steady-state-solution) | MATLAB | Steady-state 2D FDM with BCs |

# DLA Solver Optimisation — Resources & References

Collected references for each optimisation stage of the DLA Laplace solver.
Covers iterative solvers, multicore parallelism, GPU computing, and multigrid methods.
All code references target modern C++ (C++17 and onwards).

---

## 1. Eigen Sparse Solvers (Conjugate Gradient & BiCGSTAB)

### Documentation
- **Eigen Iterative Solvers Module** — API reference for `ConjugateGradient`, `BiCGSTAB`, preconditioners, and `solveWithGuess()` (warm-start):
  https://eigen.tuxfamily.org/dox/group__IterativeLinearSolvers__Module.html

- **`Eigen::ConjugateGradient`** — class reference; for best performance use `Lower|Upper` template param which also enables OpenMP internally:
  https://eigen.tuxfamily.org/dox/classEigen_1_1ConjugateGradient.html

- **`Eigen::BiCGSTAB`** — alternative for non-symmetric systems (useful if you modify the Laplacian with advection terms later):
  https://eigen.tuxfamily.org/dox/classEigen_1_1BiCGSTAB.html

- **Eigen Sparse Matrix Tutorial** — how to build a `SparseMatrix` efficiently using triplets:
  https://eigen.tuxfamily.org/dox/group__TutorialSparse.html

- **Eigen and Multi-Threading** — how Eigen uses OpenMP internally for sparse solvers:
  https://eigen.tuxfamily.org/dox/TopicMultiThreading.html

### Key idea for DLA
The 2D Laplacian on an N×N grid with periodic-x and Dirichlet-y boundary conditions
produces a sparse symmetric positive-definite matrix with ~5 nonzeros per row (N²×N²
total size). Building it once and using `solveWithGuess()` with the previous solution
as initial guess is the easiest speedup — CG converges in far fewer iterations than SOR
because it minimises the error in the Krylov subspace.

### Textbook
- **Shewchuk, J.R. (1994)** — *"An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"*, CMU Technical Report.
  https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
  — The classic accessible introduction. Covers CG, preconditioning, convergence.

---

## 2. Red-Black SOR with OpenMP

### Theory
- **Wikipedia: Successive Over-Relaxation** — includes red-black ordering, optimal ω formula, and convergence analysis:
  https://en.wikipedia.org/wiki/Successive_over-relaxation

- **Young, D.M. (1971)** — *Iterative Solution of Large Linear Systems*. Academic Press.
  — The definitive treatment of SOR theory, optimal relaxation parameters, and ordering strategies.

### Red-Black SOR concept
Split the grid into a checkerboard pattern (red/black cells). In each SOR sweep:
1. Update all **red** cells in parallel (each red cell only reads from black neighbours)
2. Synchronise
3. Update all **black** cells in parallel (each black cell only reads from updated red neighbours)

This eliminates data races and is trivially parallelisable with `#pragma omp parallel for`.

### OpenMP references
- **OpenMP 5.2 Specification** — the authoritative reference:
  https://www.openmp.org/specifications/

- **OpenMP Compilers & Tools** — which compilers support what (GCC 15 has full OpenMP 5.0+):
  https://www.openmp.org/resources/openmp-compilers-tools/

- **LLNL OpenMP Tutorial** — practical introduction with C/C++ examples:
  https://hpc-tutorials.llnl.gov/openmp/

- **Tim Mattson's OpenMP video course** (YouTube/Intel) — free, covers `parallel for`, reductions, scheduling:
  https://www.youtube.com/playlist?list=PLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG

### Example code
- **`pgomur/lid-driven-cavity-2D`** — Fortran CFD with red-black SOR + OpenMP parallelisation, HDF5 output. Good structural reference:
  https://github.com/pgomur/lid-driven-cavity-2D

### C++17 Parallel STL (alternative to raw OpenMP)
- **`std::execution::par`** — C++17 parallel execution policies. With GCC + TBB, algorithms like `std::for_each(std::execution::par, ...)` run on a thread pool:
  https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag

- Compile with: `g++ -std=c++20 -O2 -ltbb` (needs Intel TBB installed via `brew install tbb`)

---

## 3. GPU Computing on Apple Silicon (Metal)

Since the target machine is an Apple M4 Pro with Metal 3, CUDA is not available.
The options are Metal compute shaders (native) or portable frameworks.

### Apple Metal
- **metal-cpp** — Apple's official header-only C++ bindings for Metal. Zero overhead, direct mapping of Objective-C Metal API to C++:
  https://developer.apple.com/metal/cpp/

- **Metal Shading Language Specification** — reference for writing `.metal` compute kernels:
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf

- **"Performing Calculations on a GPU"** — Apple's tutorial on dispatching compute kernels:
  https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu

- **`larsgeb/m1-gpu-cpp`** — **Highly relevant**: scientific C++ on Apple Silicon using metal-cpp. Includes 2D Laplacian stencil kernels (5-point and 9-point), wave propagation, and benchmarks vs CPU. CMake-based, with Python bindings. Accompanies the paper [arXiv:2206.01791](https://arxiv.org/abs/2206.01791):
  https://github.com/larsgeb/m1-gpu-cpp

  Key examples:
  - `03-2DKernels/` — `laplacian2d` Metal kernel (exactly what we need for Jacobi iteration)
  - `05-WavePropagation/` — staggered-grid FD on GPU

### Why Jacobi (not SOR) on GPU
SOR is inherently sequential (each cell depends on already-updated neighbours in the same sweep). Jacobi iteration reads only from the *previous* iteration's values, making it embarrassingly parallel — ideal for GPU dispatch. On GPU, the extra iterations Jacobi needs are more than compensated by massive parallelism.

### CUDA (reference, not usable on Mac)
These are useful for understanding GPU programming concepts even if you'll use Metal:
- **`CoffeeBeforeArch/cuda_programming`** — YouTube-linked CUDA crash course with progressive examples (vector add → matrix mul → stencils):
  https://github.com/CoffeeBeforeArch/cuda_programming

- **`olcf-tutorials/vector_addition_cuda`** — minimal CUDA example, good for understanding kernel launch, memory management:
  https://github.com/olcf-tutorials/vector_addition_cuda

---

## 4. Portable GPU Frameworks (Cross-Platform)

### Kokkos (C++20)
Performance-portable parallel programming model. Write once, run on CUDA, HIP, SYCL, OpenMP, threads.
Well-suited for stencil operations on structured grids.

- **Repository**: https://github.com/kokkos/kokkos
- **Video lectures** (comprehensive tutorial series): https://kokkos.org/kokkos-core-wiki/tutorials-and-examples/video-lectures.html
- **Programming guide**: https://kokkos.org/kokkos-core-wiki/programmingguide.html
- **API reference**: https://kokkos.org/kokkos-core-wiki/API/core-index.html

### AdaptiveCpp / hipSYCL (C++17 SYCL)
Independent SYCL implementation supporting CPUs + GPUs from Intel, NVIDIA, AMD.
Single-source C++ with automatic GPU offloading.

- **Repository**: https://github.com/AdaptiveCpp/AdaptiveCpp
- **SYCL specification**: https://www.khronos.org/sycl/

### NVIDIA stdexec (C++20 Senders)
Reference implementation of P2300 `std::execution` (accepted into C++26). Includes GPU scheduling via `nvexec`.

- **Repository**: https://github.com/NVIDIA/stdexec
- Requires nvc++ for GPU; CPU-only with GCC 11+ or Clang 16+.

---

## 5. Multigrid Methods

### Theory
- **Wikipedia: Multigrid Method** — good overview of V-cycle, F-cycle, W-cycle, smoothing, restriction, prolongation:
  https://en.wikipedia.org/wiki/Multigrid_method

- **Briggs, W.L., Henson, V.E. & McCormick, S.F. (2000)** — *A Multigrid Tutorial* (2nd ed.). SIAM.
  — The standard introductory text. Covers geometric multigrid for Poisson/Laplace in detail.
  https://www.siam.org/publications/books/a-multigrid-tutorial-second-edition-se07/

- **Trottenberg, U., Oosterlee, C.W. & Schüller, A. (2001)** — *Multigrid*. Academic Press.
  — Comprehensive reference covering algebraic and geometric multigrid.

### Why multigrid for DLA
Multigrid solves ∇²c = 0 in O(N²) work (one pass over the grid), vs O(N² log N) for CG
or O(N³) for SOR. For DLA specifically, only one cell changes per growth step, so:
- The correction to the field is very localised
- A V-cycle with 2-3 levels and warm-start should converge in 1-3 cycles
- This is the biggest theoretical speedup available

### Implementation references
- **`cfinch/multigrid-poisson`** — Simple C multigrid Poisson solver:
  https://github.com/cfinch/multigrid-poisson

- The V-cycle pseudocode from the Wikipedia article is a direct template for implementation.

---

## 6. Profiling & Benchmarking Tools

### Built-in (what we use)
- `std::chrono::high_resolution_clock` — per-section timing in the DLA main loop.
  Output: `output/dla_profile.txt` with per-step breakdown.

### External tools
- **Instruments.app** (macOS) — Apple's profiler. Use the "Time Profiler" template to find hotspots:
  `xcrun xctrace record --template 'Time Profiler' --launch ./dla_sim`

- **`perf`** (Linux) — standard performance counters, flamegraphs.

- **Eigen's internal timers** — `EIGEN_RUNTIME_NO_MALLOC` macro to detect unexpected allocations in hot loops.

---

## 7. Books (General HPC / Scientific Computing in C++)

- **Hager, G. & Wellein, G. (2010)** — *Introduction to High Performance Computing for Scientists and Engineers*. CRC Press.
  — Covers cache optimisation, OpenMP, MPI, and GPU computing with practical C/C++ examples.

- **Kirk, D.B. & Hwu, W.-M.W. (2016)** — *Programming Massively Parallel Processors* (3rd ed.). Morgan Kaufmann.
  — The standard GPU programming textbook. CUDA-focused but concepts transfer to Metal/SYCL.

- **Sanders, J. & Kandrot, E. (2010)** — *CUDA by Example*. Addison-Wesley.
  — Beginner-friendly GPU programming introduction.

- **Josuttis, N. (2019)** — *C++17 — The Complete Guide*. Self-published.
  — Covers parallel STL, execution policies, structured bindings, `std::optional`, etc.

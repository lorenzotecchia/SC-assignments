# Assignment Set 2 — Reference & Reading Material

## 1. Clarification: The Growth Probability p_g and the η Parameter

### The Continuous (Deterministic) DLA Model (Lecture Slides 9–11)

The slides first present DLA in its **continuous mathematical form**, consisting of two coupled equations:

1. **Diffusion** of nutrient concentration outside the cluster:
   ```
   ∂c(x,t)/∂t = D ∇²c(x,t),    x ∈ {outside the cluster}
   ```

2. **Boundary growth** — the cluster surface advances proportionally to the incoming flux:
   ```
   v_n = −(D / ρ₀) ∂c/∂n,       x ∈ {boundary of the cluster}
   ```

In this formulation there is **no p_g** and **no η**. Growth is deterministic and proportional to the local concentration gradient. The problem: the cluster boundary is fractal, making ∂c/∂n undefined (the normal `n` doesn't exist on a fractal surface).

### The Stochastic DLA Algorithm (Lecture Slide 19 + Assignment Sheet)

To handle the fractal boundary, we switch to a **stochastic, lattice-based** approach. The continuous gradient ∂c/∂n is replaced by the local concentration c_{i,j} at discrete candidate sites, and growth becomes probabilistic:

```
                    (c_{i,j})^η
p_g(i,j) = ─────────────────────────────
            Σ_{candidates} (c_{i,j})^η
```

**Key points:**
- **p_g** is a **normalized probability** — all candidate probabilities sum to 1.
- At each growth step, **exactly one candidate** is selected with probability p_g (using e.g. `numpy.choice(candidates, p=probabilities)`).
- **η (eta)** controls the morphology:
  - `η = 0` → all candidates equally likely → compact **Eden cluster**
  - `η = 1` → classical DLA (probability proportional to local concentration)
  - `η > 1` → growth biased strongly toward highest-concentration tips → more dendritic / lightning-like patterns
  - `η < 1` → growth more uniform → more compact clusters

### Why p_g Is Essential

The assignment requires you to:
1. Solve the **Laplace equation** (∇²c = 0, steady-state diffusion) using SOR from Set 1
2. Identify growth candidates (non-cluster neighbours of cluster sites)
3. Compute **p_g** for each candidate using the formula above
4. Select one candidate stochastically
5. Add it to the cluster, then re-solve the Laplace equation
6. Repeat for many growth steps

Without p_g, you have no mechanism to select which candidate grows. The η parameter lets you explore how DLA morphology transitions from compact (Eden) to dendritic to needle-like.

---

## 2. Theoretical Background

### 2.1 Diffusion-Limited Aggregation (DLA)

**Foundational paper:**
- Witten, T.A. & Sander, L.M. (1981). *"Diffusion-Limited Aggregation, a Kinetic Critical Phenomenon."* Physical Review Letters, 47(19), 1400–1403.
  [doi:10.1103/PhysRevLett.47.1400](https://doi.org/10.1103/PhysRevLett.47.1400)
  — The original paper defining DLA as random walkers aggregating on a seed.

**Review / textbook references:**
- Vicsek, T. (1992). *Fractal Growth Phenomena* (2nd ed.). World Scientific.
  — Comprehensive treatment of DLA, fractal dimensions, and growth models.
- Barabási, A.-L. & Stanley, H.E. (1995). *Fractal Concepts in Surface Growth.* Cambridge University Press.
  — Chapter on Eden model and DLA, discusses η-like parameters and cluster morphology.
- Meakin, P. (1998). *Fractals, Scaling and Growth Far from Equilibrium.* Cambridge University Press.
  — Extensive coverage of DLA, Laplacian growth, and stochastic models.

**Wikipedia overview:**
- [Diffusion-limited aggregation](https://en.wikipedia.org/wiki/Diffusion-limited_aggregation)
- [Eden growth model](https://en.wikipedia.org/wiki/Eden_growth_model) — the η = 0 limit of the growth probability formula.

### 2.2 Laplacian Growth and the η Parameter

The growth probability formula with η generalises several growth models:
- **Niemeyer, Pietronero & Wiesmann (1984)**: *"Fractal Dimension of Dielectric Breakdown."* Physical Review Letters, 52(12), 1033–1036. [doi:10.1103/PhysRevLett.52.1033](https://doi.org/10.1103/PhysRevLett.52.1033)
  — Introduces the **dielectric breakdown model (DBM)**, which uses exactly the formula `p ∝ (∇φ)^η`. This is the origin of the η parameter in the DLA growth probability. The assignment's formula is a lattice discretisation of their model.
- **Paterson, L. (1984)**: *"Diffusion-Limited Aggregation and Two-Fluid Displacements in Porous Media."* Physical Review Letters, 52(18), 1621–1624.
  — Connection between DLA and viscous fingering.

### 2.3 Random Walk / Monte Carlo DLA (Assignment 2.2)

- **Nittmann, J. & Stanley, H.E. (1986)**: *"Tip splitting without interfacial tension and dendritic growth patterns arising from molecular anisotropy."* Nature, 321, 663–668.
  — Introduces stubbornness / sticking probability p_s concept.
- Landau, D.P. & Binder, K. (2000). *A Guide to Monte Carlo Simulations in Statistical Physics.* Cambridge University Press. Chapter on DLA and random walk methods.

### 2.4 Gray-Scott Reaction-Diffusion Model (Assignment 2.3)

**Original paper:**
- Pearson, J.E. (1993). *"Complex Patterns in a Simple System."* Science, 261(5118), 189–192.
  [doi:10.1126/science.261.5118.189](https://doi.org/10.1126/science.261.5118.189)

**Companion experimental paper:**
- Lee, K.J., McCormick, W.D., Ouyang, Q., & Swinney, H.L. (1993). *"Pattern Formation by Interacting Chemical Fronts."* Science, 261(5118), 192–194.

**Interactive / visual resources:**
- [MIT Gray-Scott project page](https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/) — parameter exploration, pattern gallery
- [Xmorphia (Caltech)](http://www.ccsf.caltech.edu/ismap/image.html) — parameter space explorer for Gray-Scott patterns
- Wikipedia: [Reaction-diffusion system](https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system)

---

## 3. Code Examples and Implementations

### 3.1 DLA Code Examples

**C++ implementations:**
- [`markstock/dla-nd`](https://github.com/markstock/dla-nd) — C, arbitrary-dimensional DLA simulator with adaptive tree subdivision, stubbornness (η-like) and stickiness parameters. Builds with CMake + libpng. Referenced in the lecture slides.
- [`3onier/Diffusion-limited-aggregation`](https://github.com/3onier/Diffusion-limited-aggregation) — C++ university project implementing DLA on a lattice, similar scope to this assignment.

**Python implementations (useful for prototyping / comparison):**
- [`ksenia007/dlaCluster`](https://github.com/ksenia007/dlaCluster) — Python DLA with GIF output and fractal dimension calculation. Good reference for the random-walk Monte Carlo approach.
- [`MatiXOfficial/dla-simulation`](https://github.com/MatiXOfficial/dla-simulation) — Python DLA simulator.
- [`058f9cf1/dla`](https://github.com/058f9cf1/dla) — Python DLA cluster simulation.

### 3.2 Gray-Scott Code Examples

**C/C++ implementations:**
- [`fogleman/GrayScott`](https://github.com/fogleman/GrayScott) — **C++ with OpenCL** implementation of Gray-Scott reaction-diffusion. Clean, well-structured code.
- [`nirajan-mandal/Gray-Scott-Reaction-Diffusion-Model`](https://github.com/nirajan-mandal/Gray-Scott-Reaction-Diffusion-Model) — **C++ with MPI** parallelisation of Gray-Scott. Relevant if you want to explore parallelism.
- [`DarkPhoenix42/ReDi`](https://github.com/DarkPhoenix42/ReDi) — **C++ with SDL2** visualisation of Gray-Scott.
- [`datavorous/Gray-Scott-Reaction-Diffusion-Model`](https://github.com/datavorous/Gray-Scott-Reaction-Diffusion-Model) — **C** implementation with Raylib visualisation.
- [`rodrigosetti/reaction-diffusion`](https://github.com/rodrigosetti/reaction-diffusion) — Simple **C** implementation of Gray-Scott.

### 3.3 SOR / Laplace Solver (needed for DLA)

Your Set 1 SOR implementation is the foundation. Key optimisation for DLA:
- **Warm-start**: initialise each SOR solve with the previous solution (the cluster changes by only one cell per step, so the concentration field barely changes).
- **Initial condition**: start with the analytical linear gradient c(y) = y before any growth has occurred.

---

## 4. Summary of Assignment Set 2 Tasks

| Task | Points | Description |
|------|--------|-------------|
| **2.1 A** | 4 | Implement Laplacian DLA with p_g growth probability, explore η on 100×100 grid, optimise SOR ω |
| **2.1 B** | 1 | Optimise diffusion solve time, try larger grids, optionally parallelise |
| **2.2 C** | 2 | Monte Carlo DLA via random walkers (no Laplace solve needed) |
| **2.2 D** | 1 | Add sticking probability p_s to Monte Carlo DLA, explore its effect |
| **2.3 E** | 3 | Implement Gray-Scott reaction-diffusion in 2D, explore (f, k) parameter space |

---

## 5. Key Equations Quick Reference

### DLA Growth Probability
```
p_g(i,j) = c_{i,j}^η / Σ_{candidates} c_{k,l}^η
```

### Laplace Equation (steady-state diffusion)
```
∇²c = 0   →   c_{i,j} = ¼(c_{i+1,j} + c_{i-1,j} + c_{i,j+1} + c_{i,j-1})
```

### SOR Iteration
```
c_{i,j}^{new} = (1−ω)·c_{i,j}^{old} + ω·¼(c_{i+1,j} + c_{i-1,j} + c_{i,j+1} + c_{i,j-1})
```

### Gray-Scott Model
```
∂u/∂t = D_u ∇²u − uv² + f(1 − u)
∂v/∂t = D_v ∇²v + uv² − (f + k)v
```
Suggested initial parameters: δt=1, δx=1, D_u=0.16, D_v=0.08, f=0.035, k=0.060.


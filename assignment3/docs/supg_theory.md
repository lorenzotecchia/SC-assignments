# SUPG Stabilisation for High-Reynolds-Number FEM

## 1. Why standard Galerkin fails at high Re

The incompressible Navier-Stokes equations in non-dimensional form are:

```
∂u/∂t + (u·∇)u − (1/Re)∇²u + ∇p = 0
∇·u = 0
```

At high Reynolds number the convective term `(u·∇)u` dominates the viscous term `(1/Re)∇²u`.
The ratio of these two scales is the **element Péclet number**:

```
Pe_h = |u| h / (2ν)   where h = local element size, ν = 1/Re
```

When Pe_h >> 1 (high Re or coarse mesh) the standard Galerkin formulation produces
**spurious node-to-node oscillations** across convection layers. This is the FEM analogue
of the central-difference instability in finite differences: both methods fail to respect
the upwind direction of information propagation.

The classical remedy in FD is **upwind differencing** (first-order, dissipative).
In FEM the elegant solution is **SUPG** — Streamline-Upwind Petrov-Galerkin.

---

## 2. Petrov-Galerkin idea

In the standard Galerkin method trial and test functions live in the same space:

```
∫ (u·∇u) · v dx   (standard, v = test function)
```

Petrov-Galerkin breaks this symmetry by enriching the test function:

```
v  →  v + τ (u·∇)v
```

The added term `τ (u·∇)v` is the **streamline derivative** of the test function —
it points in the direction of convection and provides upwinding without destroying
consistency: if u satisfies the PDE, the extra term vanishes identically.

---

## 3. SUPG for the unsteady convection–diffusion and NS equations

For the time-dependent momentum equation, the SUPG-stabilised weak form adds the
following elemental integral to the standard Galerkin convection term:

```
∑_K  ∫_K  τ_K  (u·∇u) · (u·∇v)  dΩ
```

where the sum is over all mesh elements K.

This can be read as: weight the momentum residual `(u·∇u)` by the streamline
derivative of the test function `(u·∇v)`, scaled by τ_K.

The pressure and continuity equations are not modified (Taylor-Hood already
satisfies the inf-sup condition; PSPG pressure stabilisation is a separate
technique needed only for equal-order elements).

---

## 4. The stabilisation parameter τ

The parameter τ controls the strength of stabilisation. For **unsteady** flows
the standard choice (Tezduyar & Osawa 2000, building on Franca, Frey & Hughes 1992) is:

```
τ_K = [ (2/Δt)²  +  (2|u|/h_K)²  +  (4ν/h_K²)² ]^{-1/2}
```

The three terms inside the square root represent the three competing time scales
at element level:

| Term | Scale | Dominant when |
|------|-------|---------------|
| `(2/Δt)²` | temporal | unsteady, large Δt |
| `(2\|u\|/h)²` | convective | high Re, coarse mesh |
| `(4ν/h²)²` | diffusive | low Re, fine mesh |

When convection dominates (high Re, Pe_h >> 1):

```
τ ≈ h / (2|u|)
```

which is the classical **upwind length** h/2 divided by the advection speed —
identical in structure to the FD upwind bias.

When time accuracy matters (small Δt):

```
τ ≈ Δt / 2
```

The formula naturally blends between these limits.

---

## 5. Implementation in `vortex.py`

The SUPG term is added **explicitly** (evaluated at the current time level u^n),
consistent with the IMEX-1 time integrator already in place.

At each time step, after computing the Galerkin residual `r = (A + C(u^n)) u^n`,
the SUPG contribution is assembled and added:

```python
# τ as a symbolic NGSolve CoefficientFunction (per element)
h        = specialcf.mesh_size
vel_norm = Norm(velocity)           # |u^n|
tau_supg = 1 / sqrt((2/dt)**2 + (2*vel_norm/h)**2 + (4*nu/h**2)**2 + ε)

# SUPG linear form: τ (u^n·∇u^n) · (u^n·∇v) dx
supg_lf += tau_supg * InnerProduct(Grad(velocity)*velocity,
                                    Grad(v)*velocity) * dx

# In the time loop:
supg_lf.Assemble()          # re-evaluate with current u^n
res += supg_lf.vec          # add to Galerkin residual
gfu.vec -= dt * inv * res   # IMEX-1 update (unchanged)
```

Key implementation notes:
- `velocity = gfu.components[0]` is a lazy reference; `Assemble()` uses the current values.
- `mstar` (the implicit operator) is **not** modified — the stabilisation is purely explicit.
- Enable with `--supg` flag; disabled by default to preserve Re=100 benchmark fidelity.

---

## 6. Effect on maximum achievable Re

Without SUPG, the FEM solver becomes unstable when the element Péclet number exceeds
roughly Pe_h ≈ 1, i.e., when:

```
Re  >  h / (2 * L_ref)  ×  (L / h)  =  L / (2 * L_ref)
```

For the Schäfer-Turek geometry (L=2.2, L_ref=D=0.1) with maxh=0.07:
Pe_h = 1  ⟹  Re ≈ 2.2 / (2 × 0.07) ≈ 16

This is why the baseline solver needs very small dt or fine mesh at Re=100.
SUPG removes this hard Péclet constraint: the solver can handle Pe_h >> 1
(convection-dominated), allowing Re up to 500–1000 on the same mesh,
limited only by temporal resolution and the accuracy of the linear solver.

---

## 7. References

- **Hughes, Brooks (1979)** — "A multi-dimensional upwind scheme with no crosswind diffusion". *Finite Element Methods for Convection Dominated Flows*, AMD vol. 34. *(Original SUPG paper.)*
- **Brooks & Hughes (1982)** — "Streamline Upwind/Petrov-Galerkin formulations for convection dominated flows with particular emphasis on the incompressible Navier-Stokes equations". *Comput. Methods Appl. Mech. Engrg.* **32**, 199–259.
- **Franca, Frey & Hughes (1992)** — "Stabilized finite element methods: I. Application to the advective-diffusive model". *Comput. Methods Appl. Mech. Engrg.* **95**, 253–276. *(τ formula derivation.)*
- **Tezduyar & Osawa (2000)** — "Finite element stabilization parameters computed from element matrices and vectors". *Comput. Methods Appl. Mech. Engrg.* **190**, 411–430. *(Unsteady τ formula used here.)*
- **Schäfer & Turek (1996)** — benchmark geometry and reference drag/lift values.

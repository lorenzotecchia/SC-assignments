"""
Kármán Vortex Street — NGSolve FEM Implementation
===================================================
Solves the incompressible Navier-Stokes equations using a Taylor-Hood
(P3/P2) finite element pair with IMEX time integration.

Geometry follows the assignment specification (Figure 4):
  - Channel: 2.2 m x H m  (H = 0.41 m per Schäfer-Turek benchmark)
  - Cylinder center: (0.2, 0.2), radius 0.05 m
  - Inlet (left), outlet (right), no-slip walls (top/bottom + cylinder)

Output:
  - VTK files for ParaView visualisation  (karman_output/*.vtu + .pvd)
  - Drag and lift coefficient time series (drag_lift.csv)
  - Matplotlib plots of drag/lift vs time  (drag.png, lift.png)

Usage:
  python3 karman_ngsolve.py              # default Re ≈ 100
  python3 karman_ngsolve.py --Re 200     # higher Reynolds number
  python3 karman_ngsolve.py --help       # all options
"""

import argparse
import os
import time as walltime

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for terminal use
import matplotlib.pyplot as plt
from netgen.occ import *
from ngsolve import *

# ─────────────────────────────────────────────
# 0. Command-line arguments
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Kármán vortex street with NGSolve")
parser.add_argument(
    "--Re", type=float, default=100, help="Reynolds number (default: 100)"
)
parser.add_argument(
    "--dt", type=float, default=0.0005, help="Time step (default: 0.0005)"
)
parser.add_argument("--tend", type=float, default=8.0, help="End time (default: 8.0)")
parser.add_argument(
    "--order", type=int, default=2, help="Polynomial order k for velocity (default: 2)"
)
parser.add_argument(
    "--maxh", type=float, default=0.07, help="Max mesh size (default: 0.07)"
)
parser.add_argument(
    "--vtk-interval", type=int, default=50, help="Write VTK every N steps (default: 50)"
)
parser.add_argument(
    "--outdir", type=str, default="karman_output", help="Output directory"
)
args = parser.parse_args()

# derived quantities
L_ref = 0.1  # cylinder diameter
U_mean = 1.0  # mean inflow velocity
H = 0.41  # channel height
nu_visc = U_mean * L_ref / args.Re  # kinematic viscosity

print(f"═══════════════════════════════════════════════")
print(f"  Kármán Vortex Street — NGSolve FEM Solver")
print(f"═══════════════════════════════════════════════")
print(f"  Re       = {args.Re}")
print(f"  ν        = {nu_visc:.6f}")
print(f"  dt       = {args.dt}")
print(f"  tend     = {args.tend}")
print(f"  order    = {args.order}")
print(f"  maxh     = {args.maxh}")
print(f"  output   = {args.outdir}/")
print(f"═══════════════════════════════════════════════\n")

os.makedirs(args.outdir, exist_ok=True)

# ─────────────────────────────────────────────
# 1. Geometry & Mesh
# ─────────────────────────────────────────────
print("[1/6] Building geometry and mesh ...")

# OCC geometry: channel rectangle minus cylinder
shape = Rectangle(2.2, H).Circle(0.2, 0.2, 0.05).Reverse().Face()

# label boundaries
shape.edges.name = "cyl"
shape.edges.Min(X).name = "inlet"
shape.edges.Max(X).name = "outlet"
shape.edges.Min(Y).name = "wall"
shape.edges.Max(Y).name = "wall"

mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=args.maxh)).Curve(3)
print(f"       Mesh: {mesh.ne} elements, {mesh.nv} vertices")

# ─────────────────────────────────────────────
# 2. Finite Element Spaces
# ─────────────────────────────────────────────
print("[2/6] Setting up Taylor-Hood FE spaces ...")

k = args.order
V = VectorH1(mesh, order=k, dirichlet="wall|cyl|inlet")
Q = H1(mesh, order=k - 1)
X = V * Q

(u, p), (v, q) = X.TnT()

ndof = X.ndof
print(f"       V order={k}, Q order={k - 1}, total DOFs = {ndof}")

# ─────────────────────────────────────────────
# 3. Forms & Initial Conditions (Stokes solve)
# ─────────────────────────────────────────────
print("[3/6] Assembling forms and solving Stokes for initial data ...")

# grid function for the solution
gfu = GridFunction(X)
velocity, pressure = gfu.components

# parabolic inflow: u_x = 6 y (H-y) / H^2  so that mean = 1
uin = CoefficientFunction((1.5 * 4 * y * (H - y) / (H * H), 0))
velocity.Set(uin, definedon=mesh.Boundaries("inlet"))

# Stokes bilinear form
stokes = (nu_visc * InnerProduct(grad(u), grad(v)) - div(u) * q - div(v) * p) * dx

a = BilinearForm(X)
a += stokes
a.Assemble()

f = LinearForm(X)
f.Assemble()

# solve Stokes for initial conditions
res = f.vec - a.mat * gfu.vec
inv_stokes = a.mat.Inverse(X.FreeDofs())
gfu.vec.data += inv_stokes * res
print("       Stokes initial solve complete.")

# ─────────────────────────────────────────────
# 4. Time-stepping setup (IMEX-1)
# ─────────────────────────────────────────────
print("[4/6] Setting up IMEX time integrator ...")

dt = args.dt

# M* = M + dt * A_stokes  (implicit part)
mstar = BilinearForm(X)
mstar += InnerProduct(u, v) * dx + dt * stokes
mstar.Assemble()
inv = mstar.mat.Inverse(X.FreeDofs())

# convection form (nonassemble — applied as operator)
conv = BilinearForm(X, nonassemble=True)
conv += (Grad(u) * u) * v * dx

print("       IMEX setup complete.")

# ─────────────────────────────────────────────
# 5. Drag & Lift test functions
# ─────────────────────────────────────────────
# scaling factor: 2 / (U_mean^2 * L_ref)
cd_scale = 2.0 / (U_mean**2 * L_ref)

drag_x_test = GridFunction(X)
drag_x_test.components[0].Set(
    CoefficientFunction((-cd_scale, 0)), definedon=mesh.Boundaries("cyl")
)
drag_y_test = GridFunction(X)
drag_y_test.components[0].Set(
    CoefficientFunction((0, -cd_scale)), definedon=mesh.Boundaries("cyl")
)

# ─────────────────────────────────────────────
# 6. VTK output setup
# ─────────────────────────────────────────────
print("[5/6] Configuring VTK output ...")

vel_mag = Norm(velocity)
vorticity = grad(velocity)[1, 0] - grad(velocity)[0, 1]  # dv/dx - du/dy

vtk = VTKOutput(
    mesh,
    coefs=[velocity, pressure, vel_mag, vorticity],
    names=["velocity", "pressure", "velocity_mag", "vorticity"],
    filename=os.path.join(args.outdir, "karman"),
    subdivision=2,
    floatsize="single",
)

# write initial state
vtk.Do(time=0.0)

# ─────────────────────────────────────────────
# 7. Time loop
# ─────────────────────────────────────────────
print("[6/6] Starting time integration ...\n")

t = 0.0
step = 0
time_vals = []
drag_vals = []
lift_vals = []
wall_start = walltime.time()

with TaskManager():
    while t < args.tend - 0.5 * dt:
        # IMEX step: residual = (A + C(u)) * u
        res = (a.mat + conv.mat) * gfu.vec
        gfu.vec.data -= dt * inv * res

        t += dt
        step += 1

        # record drag and lift
        cd = InnerProduct(res, drag_x_test.vec)
        cl = InnerProduct(res, drag_y_test.vec)
        time_vals.append(t)
        drag_vals.append(cd)
        lift_vals.append(cl)

        # VTK output
        if step % args.vtk_interval == 0:
            vtk.Do(time=t)
            elapsed = walltime.time() - wall_start
            print(
                f"  t = {t:8.4f} / {args.tend}  |  "
                f"C_D = {cd:+10.5f}  C_L = {cl:+10.5f}  |  "
                f"wall: {elapsed:6.1f}s"
            )

elapsed = walltime.time() - wall_start
print(f"\n  Simulation finished.  Total wall time: {elapsed:.1f}s")
print(f"  Steps: {step},  VTK frames: {step // args.vtk_interval}")

# ─────────────────────────────────────────────
# 8. Post-processing: save drag/lift data + plots
# ─────────────────────────────────────────────
print("\nPost-processing ...")

# CSV
csv_path = os.path.join(args.outdir, "drag_lift.csv")
np.savetxt(
    csv_path,
    np.column_stack([time_vals, drag_vals, lift_vals]),
    header="time,drag,lift",
    delimiter=",",
    comments="",
)
print(f"  Saved: {csv_path}")

# Drag plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time_vals, drag_vals, linewidth=0.8)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Drag coefficient $C_D$")
ax.set_title(f"Drag coefficient — Re = {args.Re}")
ax.grid(True, alpha=0.3)
fig.tight_layout()
drag_png = os.path.join(args.outdir, "drag.png")
fig.savefig(drag_png, dpi=150)
print(f"  Saved: {drag_png}")
plt.close(fig)

# Lift plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time_vals, lift_vals, linewidth=0.8, color="tab:orange")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Lift coefficient $C_L$")
ax.set_title(f"Lift coefficient — Re = {args.Re}")
ax.grid(True, alpha=0.3)
fig.tight_layout()
lift_png = os.path.join(args.outdir, "lift.png")
fig.savefig(lift_png, dpi=150)
print(f"  Saved: {lift_png}")
plt.close(fig)

print(f"\nDone!  Open in ParaView:")
print(f"  paraview {os.path.join(args.outdir, 'karman.pvd')}")

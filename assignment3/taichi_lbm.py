# Fluid solver based on lattice boltzmann method using taichi language
# Author : Wang (hietwll@gmail.com)

import argparse
import sys

import matplotlib
import numpy as np
import taichi as ti
import taichi.math as tm
from matplotlib import cm

ti.init(arch=ti.gpu)


@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        name,  # name of the flow case
        nx,  # domain size
        ny,
        niu,  # viscosity of fluid
        bc_type,  # [left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
        bc_value,  # if bc_type = 0, we need to specify the velocity in bc_value
        cy=0,  # whether to place a cylindrical obstacle
        cy_para=[0.0, 0.0, 0.0],  # location and radius of the cylinder
        C_smag=0.0,  # Smagorinsky constant (0 = disabled)
    ):
        self.name = name
        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny
        self.niu = niu
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau
        self.C_smag = C_smag
        self.rho = ti.field(float, shape=(nx, ny))
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))
        self.mask = ti.field(float, shape=(nx, ny))
        self.f_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, float, shape=(nx, ny))
        self.tau_eff = ti.field(
            float, shape=(nx, ny)
        )  # per-cell effective relaxation time
        self.w = (
            ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        )
        self.e = ti.types.matrix(9, 2, int)(
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]
        )
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))
        self.cy = cy
        self.cy_para = tm.vec3(cy_para)

    @ti.func  # compute equilibrium distribution function
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel
    def init(self):
        self.vel.fill(0)
        self.rho.fill(1)
        self.mask.fill(0)
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)
            if self.cy == 1:
                if (i - self.cy_para[0]) ** 2 + (
                    j - self.cy_para[1]
                ) ** 2 <= self.cy_para[2] ** 2:
                    self.mask[i, j] = 1.0

    @ti.kernel
    def compute_tau_sgs(self):  # Smagorinsky eddy-viscosity correction
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            f = self.f_old[i, j]  # whole local vector — no SNode alias issue
            feq = self.f_eq(i, j)
            Q_xx = 0.0
            Q_yy = 0.0
            Q_xy = 0.0
            for k in ti.static(range(9)):
                fneq = f[k] - feq[k]
                Q_xx += float(self.e[k, 0] * self.e[k, 0]) * fneq
                Q_yy += float(self.e[k, 1] * self.e[k, 1]) * fneq
                Q_xy += float(self.e[k, 0] * self.e[k, 1]) * fneq
            pi_neq = tm.sqrt(Q_xx**2 + Q_yy**2 + 2.0 * Q_xy**2)
            tau0 = self.tau
            self.tau_eff[i, j] = 0.5 * (
                tau0
                + tm.sqrt(tau0**2 + 18.0 * self.C_smag**2 * pi_neq / self.rho[i, j])
            )

    @ti.kernel
    def init_tau_eff(self):  # fill tau_eff with base tau before first SGS pass
        for i, j in self.tau_eff:
            self.tau_eff[i, j] = self.tau

    @ti.kernel
    def collide_and_stream(self):  # lbm core equation
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                feq = self.f_eq(ip, jp)
                inv_tau_local = 1.0 / self.tau_eff[ip, jp]
                self.f_new[i, j][k] = (1 - inv_tau_local) * self.f_old[ip, jp][k] + feq[
                    k
                ] * inv_tau_local

    @ti.kernel
    def update_macro_var(self):  # compute rho u v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0
            self.vel[i, j] = 0, 0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j] += (
                    tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]
                )

            self.vel[i, j] /= self.rho[i, j]

    @ti.kernel
    def apply_bc(self):  # impose boundary conditions
        # left and right
        for j in range(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(1, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in range(self.nx):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # cylindrical obstacle
        # Note: for cuda backend, putting 'if statement' inside loops can be much faster!
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.cy == 1 and self.mask[i, j] == 1:
                self.vel[i, j] = 0, 0  # velocity is zero at solid boundary
                inb = 0
                jnb = 0
                if i >= self.cy_para[0]:
                    inb = i + 1
                else:
                    inb = i - 1
                if j >= self.cy_para[1]:
                    jnb = j + 1
                else:
                    jnb = j - 1
                self.apply_bc_core(0, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if outer == 1:  # handle outer boundary
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr]

            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]

        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = (
            self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
        )

    def solve(self):
        gui = ti.GUI(self.name, (self.nx, 2 * self.ny))
        self.init()
        self.init_tau_eff()
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            for _ in range(100):
                if self.C_smag > 0.0:
                    self.compute_tau_sgs()
                self.collide_and_stream()
                self.update_macro_var()
                self.apply_bc()

            ##  code fragment displaying vorticity is contributed by woclass
            vel = self.vel.to_numpy()
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]
            vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
            ## color map
            colors = [
                (1, 1, 0),
                (0.953, 0.490, 0.016),
                (0, 0, 0),
                (0.176, 0.976, 0.529),
                (0, 1, 1),
            ]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "my_cmap", colors
            )
            vor_img = cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=-0.02, vmax=0.02), cmap=my_cmap
            ).to_rgba(vor)
            vel_img = cm.plasma(vel_mag / 0.15)
            img = np.concatenate((vor_img, vel_img), axis=1)
            gui.set_image(img)
            gui.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D2Q9 LBM fluid solver")
    parser.add_argument(
        "flow_case",
        nargs="?",
        type=int,
        default=0,
        help="0 = Kármán Vortex Street, 1 = Lid-driven Cavity (default: 0)",
    )
    parser.add_argument(
        "Re",
        nargs="?",
        type=float,
        default=400.0,
        help="Reynolds number for flow_case=0 (default: 400)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Grid refinement factor: multiplies nx, ny, D and cylinder "
        "geometry proportionally. scale=2 doubles the Re ceiling. (default: 1.0)",
    )
    parser.add_argument(
        "--C-smag",
        type=float,
        default=0.0,
        dest="C_smag",
        help="Smagorinsky constant for LES eddy-viscosity model "
        "(0 = disabled, typical value 0.1). (default: 0.0)",
    )
    args = parser.parse_args()

    if args.flow_case == 0:  # von Karman vortex street
        s = args.scale
        U = 0.1
        # Base geometry (scale=1): 801×201, cylinder D=40 at (160, 100) r=20
        D = 40.0 * s
        nx = int(801 * s)
        ny = int(201 * s)
        cx = 160.0 * s
        cy = 100.0 * s
        r = 20.0 * s
        niu = U * D / args.Re
        tau = 3.0 * niu + 0.5
        smag_str = f"  C_smag={args.C_smag}" if args.C_smag > 0.0 else ""
        print(
            f"Kármán Vortex Street | Re={args.Re:.1f}  scale={s}  "
            f"grid={nx}×{ny}  D={D:.0f}  niu={niu:.6f}  tau={tau:.6f}{smag_str}"
        )
        if tau <= 0.5:
            print(
                "ERROR: tau <= 0.5, simulation will be unstable. "
                "Lower Re or increase --scale."
            )
            sys.exit(1)
        lbm = lbm_solver(
            f"Kármán Vortex Street Re={args.Re:.0f} scale={s}",
            nx,
            ny,
            niu,
            [0, 0, 1, 0],
            [[U, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            1,
            [cx, cy, r],
            C_smag=args.C_smag,
        )
        lbm.solve()
    elif args.flow_case == 1:  # lid-driven cavity flow: Re = U*L/niu = 1000
        lbm = lbm_solver(
            "Lid-driven Cavity Flow",
            256,
            256,
            0.0255,
            [0, 0, 0, 0],
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]],
        )
        lbm.solve()
    else:
        print(
            "Invalid flow case! Please choose 0 (Kármán Vortex Street) or 1 (Lid-driven Cavity)."
        )

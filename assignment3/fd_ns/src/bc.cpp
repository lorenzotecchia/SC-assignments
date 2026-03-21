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

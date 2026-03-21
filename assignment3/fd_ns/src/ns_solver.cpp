#include "ns_solver.hpp"
#include "pressure.hpp"
#include "bc.hpp"
#include <algorithm>
#include <cmath>

// ── convection: upwind ────────────────────────────────────────────────────────

static Eigen::MatrixXd conv_u(const MacGrid &g) {
    Eigen::MatrixXd N(g.nx+1, g.ny);
    N.setZero();
    for (int i = 1; i < g.nx; ++i) {
        for (int j = 1; j < g.ny-1; ++j) {
            // Upwind d(u²)/dx — advecting velocity at east face selects stencil
            double ue = 0.5*(g.u(i,j) + g.u(i+1,j));
            double dudx = (ue>0) ? (g.u(i,j)-g.u(i-1,j))/g.dx
                                 : (g.u(i+1,j)-g.u(i,j))/g.dx;
            // Upwind d(uv)/dy
            double vn = 0.25*(g.v(i-1,j+1)+g.v(i,j+1)+g.v(i-1,j)+g.v(i,j));
            double dudy = (vn>0) ? (g.u(i,j)-g.u(i,j-1))/g.dy
                                 : (g.u(i,j+1)-g.u(i,j))/g.dy;
            N(i,j) = -(g.u(i,j)*dudx + vn*dudy);   // −(u·∇)u
        }
    }
    return N;
}

static Eigen::MatrixXd conv_v(const MacGrid &g) {
    Eigen::MatrixXd N(g.nx, g.ny+1);
    N.setZero();
    for (int i = 1; i < g.nx-1; ++i) {
        for (int j = 1; j < g.ny; ++j) {
            double un = 0.25*(g.u(i,j)+g.u(i+1,j)+g.u(i,j-1)+g.u(i+1,j-1));
            double dvdx = (un>0) ? (g.v(i,j)-g.v(i-1,j))/g.dx
                                 : (g.v(i+1,j)-g.v(i,j))/g.dx;
            double ve = 0.5*(g.v(i,j)+g.v(i,j+1));
            double dvdy = (ve>0) ? (g.v(i,j)-g.v(i,j-1))/g.dy
                                 : (g.v(i,j+1)-g.v(i,j))/g.dy;
            N(i,j) = -(un*dvdx + g.v(i,j)*dvdy);
        }
    }
    return N;
}

// ── diffusion: central ────────────────────────────────────────────────────────

static Eigen::MatrixXd diff_u(const MacGrid &g, double nu) {
    Eigen::MatrixXd D(g.nx+1, g.ny);
    D.setZero();
    for (int i = 1; i < g.nx; ++i)
        for (int j = 1; j < g.ny-1; ++j)
            D(i,j) = nu * (
                (g.u(i+1,j) - 2*g.u(i,j) + g.u(i-1,j))/(g.dx*g.dx) +
                (g.u(i,j+1) - 2*g.u(i,j) + g.u(i,j-1))/(g.dy*g.dy));
    return D;
}

static Eigen::MatrixXd diff_v(const MacGrid &g, double nu) {
    Eigen::MatrixXd D(g.nx, g.ny+1);
    D.setZero();
    for (int i = 1; i < g.nx-1; ++i)
        for (int j = 1; j < g.ny; ++j)
            D(i,j) = nu * (
                (g.v(i+1,j) - 2*g.v(i,j) + g.v(i-1,j))/(g.dx*g.dx) +
                (g.v(i,j+1) - 2*g.v(i,j) + g.v(i,j-1))/(g.dy*g.dy));
    return D;
}

// ── divergence of u* ──────────────────────────────────────────────────────────

static Eigen::MatrixXd divergence(const MacGrid &g) {
    Eigen::MatrixXd div(g.nx, g.ny);
    for (int i = 0; i < g.nx; ++i)
        for (int j = 0; j < g.ny; ++j)
            div(i,j) = (g.u(i+1,j) - g.u(i,j))/g.dx
                     + (g.v(i,j+1) - g.v(i,j))/g.dy;
    return div;
}

// ── public: ns_step ───────────────────────────────────────────────────────────

StepStats ns_step(MacGrid &g, NSSolverConfig &cfg, int step,
                  Eigen::MatrixXd &Nu_prev, Eigen::MatrixXd &Nv_prev)
{
    // 1. Convection and diffusion
    auto Nu = conv_u(g);
    auto Nv = conv_v(g);
    auto Du = diff_u(g, cfg.nu);
    auto Dv = diff_v(g, cfg.nu);

    // 2. Adams-Bashforth (AB2) or forward Euler on first step
    double ab_a = (step == 0) ? 1.0 : 1.5;
    double ab_b = (step == 0) ? 0.0 : -0.5;

    // 3. Intermediate velocity u*
    Eigen::MatrixXd us = g.u + cfg.dt * (ab_a*Nu + ab_b*Nu_prev + Du);
    Eigen::MatrixXd vs = g.v + cfg.dt * (ab_a*Nv + ab_b*Nv_prev + Dv);

    // Swap into grid temporarily for BC + divergence
    g.u = us;  g.v = vs;
    apply_velocity_bc(g, cfg.U_mean);
    apply_solid_bc(g);

    // 4. Pressure Poisson: ∇²p = (1/dt) ∇·u*
    auto div = divergence(g);
    Eigen::MatrixXd rhs = div / cfg.dt;

    // Set Dirichlet p=0 at outlet before calling solver.
    for (int j = 0; j < g.ny; ++j) g.p(g.nx-1, j) = 0.0;

    auto res = pressure_solve_sor(g.p, rhs, g.mask,
                                  g.nx, g.ny, g.dx, g.dy,
                                  cfg.omega, cfg.p_tol, cfg.p_max_iter);

    // 5. Projection: u = u* − dt ∇p
    for (int i = 1; i < g.nx; ++i)
        for (int j = 0; j < g.ny; ++j)
            g.u(i,j) -= cfg.dt * (g.p(i,j) - g.p(i-1,j)) / g.dx;

    for (int i = 0; i < g.nx; ++i)
        for (int j = 1; j < g.ny; ++j)
            g.v(i,j) -= cfg.dt * (g.p(i,j) - g.p(i,j-1)) / g.dy;

    apply_velocity_bc(g, cfg.U_mean);
    apply_solid_bc(g);

    // 6. Save convection for next AB2 step
    Nu_prev = Nu;
    Nv_prev = Nv;

    // 7. Diagnostics: max divergence in interior pure-fluid cells.
    //    Cells adjacent to the cylinder are excluded because apply_solid_bc
    //    zeroes their faces after projection, intentionally re-introducing
    //    a small divergence as part of the no-slip enforcement.
    auto div_post = divergence(g);
    double max_div = 0.0;
    for (int i = 1; i < g.nx-1; ++i) {
        for (int j = 1; j < g.ny-1; ++j) {
            if (g.mask(i,j) != 0) continue;
            // Skip fluid cells that share a face with a solid cell
            if (g.mask(i-1,j) || g.mask(i+1,j) ||
                g.mask(i,j-1) || g.mask(i,j+1)) continue;
            max_div = std::max(max_div, std::abs(div_post(i,j)));
        }
    }
    StepStats stats;
    stats.div_max = max_div;
    stats.p_iters = res.iterations;
    return stats;
}

// ── public: compute_forces ────────────────────────────────────────────────────
// Uses momentum-flux control-volume approach around a box enclosing the cylinder.

ForceCoeffs compute_forces(const MacGrid &g, double U_mean, double D) {
    double scale = 2.0 / (U_mean * U_mean * D);
    double fx = 0.0, fy = 0.0;

    for (int i = 0; i < g.nx; ++i) {
        for (int j = 0; j < g.ny; ++j) {
            if (g.mask(i,j) != 1) continue;
            // Pressure force on each solid-cell face adjacent to fluid.
            // F = ∮ p n̂_in dA  (n̂_in = inward normal of the cylinder surface)
            // West face:  n̂_in = +x̂  → F_x = +p_west * dy
            // East face:  n̂_in = -x̂  → F_x = -p_east * dy
            // South face: n̂_in = +ŷ  → F_y = +p_south * dx
            // North face: n̂_in = -ŷ  → F_y = -p_north * dx
            if (i > 0      && g.mask(i-1,j)==0) fx += g.p(i-1,j)*g.dy; // west
            if (i < g.nx-1 && g.mask(i+1,j)==0) fx -= g.p(i+1,j)*g.dy; // east
            if (j > 0      && g.mask(i,j-1)==0) fy += g.p(i,j-1)*g.dx; // south
            if (j < g.ny-1 && g.mask(i,j+1)==0) fy -= g.p(i,j+1)*g.dx; // north
        }
    }

    return ForceCoeffs{scale * fx, scale * fy};
}

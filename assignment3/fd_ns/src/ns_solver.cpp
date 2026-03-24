#include "ns_solver.hpp"
#include "bc.hpp"
#include "pressure.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

// ── convection: upwind
// ────────────────────────────────────────────────────────

static double step_conv_u(const MacGrid &g, int i, int j) {
  double ue = 0.5 * (g.u(i, j) + g.u(i + 1, j));
  double dudx = (ue > 0) ? (g.u(i, j) - g.u(i - 1, j)) / g.dx
                         : (g.u(i + 1, j) - g.u(i, j)) / g.dx;
  // Upwind d(uv)/dy
  double vn =
      0.25 * (g.v(i - 1, j + 1) + g.v(i, j + 1) + g.v(i - 1, j) + g.v(i, j));
  double dudy = (vn > 0) ? (g.u(i, j) - g.u(i, j - 1)) / g.dy
                         : (g.u(i, j + 1) - g.u(i, j)) / g.dy;
  return -(g.u(i, j) * dudx + vn * dudy); // −(u·∇)u
}

static double step_conv_v(const MacGrid &g, int i, int j) {
  double un =
      0.25 * (g.u(i, j) + g.u(i + 1, j) + g.u(i, j - 1) + g.u(i + 1, j - 1));
  double dvdx = (un > 0) ? (g.v(i, j) - g.v(i - 1, j)) / g.dx
                         : (g.v(i + 1, j) - g.v(i, j)) / g.dx;
  double ve = 0.5 * (g.v(i, j) + g.v(i, j + 1));
  double dvdy = (ve > 0) ? (g.v(i, j) - g.v(i, j - 1)) / g.dy
                         : (g.v(i, j + 1) - g.v(i, j)) / g.dy;
  return -(un * dvdx + g.v(i, j) * dvdy);
}

static Eigen::MatrixXd conv_u(const MacGrid &g) {
  Eigen::MatrixXd N(g.nx + 1, g.ny);
  N.setZero();
#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 1; i < g.nx; ++i) {
    for (int j = 1; j < g.ny - 1; ++j) {
      N(i, j) = step_conv_u(g, i, j);
    }
  }
  return N;
}

static Eigen::MatrixXd conv_v(const MacGrid &g) {
  Eigen::MatrixXd N(g.nx, g.ny + 1);
  N.setZero();
#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 1; i < g.nx - 1; ++i) {
    for (int j = 1; j < g.ny; ++j) {
      N(i, j) = step_conv_v(g, i, j);
    }
  }
  return N;
}
// ── convection: MUSCL
// ────────────────────────────────────────────────────────
static inline double van_leer(double r) {
  if (r <= 0.0)
    return 0.0; // no limiter (upwind) at extrema
  return (r + std::abs(r)) / (1.0 + std::abs(r));
}

static inline double minmod(double r) {
  if (r <= 0.0)
    return 0.0;
  return std::min(r, 1.0);
}

static inline double muscl_face(double phi_uu, double phi_u, double phi_d) {
  // phi_uu = two-upwind, phi_u = upwind, phi_d = downwind (relative to
  // advecting flow)
  double denom = (phi_u - phi_uu);
  double r = (std::abs(denom) > 1e-12) ? (phi_d - phi_u) / denom : 0.0;
  double val = phi_u + 0.5 * minmod(r) * (phi_d - phi_u);
  return val;
}

static Eigen::MatrixXd conv_muscl_u(const MacGrid &g) {
  Eigen::MatrixXd N(g.nx + 1, g.ny);
  N.setZero();
#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 1; i < g.nx; ++i) {
    for (int j = 1; j < g.ny - 1; ++j) {

      bool safe_stencil =
          (i >= 2 && i < g.nx - 2 && j >= 2 && j < g.ny - 2 &&
           !g.mask(i - 2, j) && !g.mask(i - 1, j) && !g.mask(i, j) &&
           !g.mask(i + 1, j) && !g.mask(i + 2, j) && !g.mask(i, j - 2) &&
           !g.mask(i, j - 1) && !g.mask(i, j + 1) && !g.mask(i, j + 2));

      if (safe_stencil) {

        // ---- East flux  F_e = ue * (u at east face) ----
        double ue = 0.5 * (g.u(i, j) + g.u(i + 1, j));
        double u_e = (ue >= 0)
                         ? muscl_face(g.u(i - 1, j), g.u(i, j), g.u(i + 1, j))
                         : muscl_face(g.u(i + 2, j), g.u(i + 1, j), g.u(i, j));
        double F_e = ue * u_e;

        // ---- West flux  F_w = uw * (u at west face) ----
        double uw = 0.5 * (g.u(i - 1, j) + g.u(i, j));
        double u_w = (uw >= 0)
                         ? muscl_face(g.u(i - 2, j), g.u(i - 1, j), g.u(i, j))
                         : muscl_face(g.u(i + 1, j), g.u(i, j), g.u(i - 1, j));
        double F_w = uw * u_w;

        // ---- North flux  G_n = vn * (u at north face) ----
        double vn = 0.25 * (g.v(i - 1, j + 1) + g.v(i, j + 1) + g.v(i - 1, j) +
                            g.v(i, j));
        double u_n = (vn >= 0)
                         ? muscl_face(g.u(i, j - 1), g.u(i, j), g.u(i, j + 1))
                         : muscl_face(g.u(i, j + 2), g.u(i, j + 1), g.u(i, j));
        double G_n = vn * u_n;

        // ---- South flux  G_s = vs * (u at south face) ----
        double vs = 0.25 * (g.v(i - 1, j) + g.v(i, j) + g.v(i - 1, j - 1) +
                            g.v(i, j - 1));
        double u_s = (vs >= 0)
                         ? muscl_face(g.u(i, j - 2), g.u(i, j - 1), g.u(i, j))
                         : muscl_face(g.u(i, j + 1), g.u(i, j), g.u(i, j - 1));
        double G_s = vs * u_s;

        N(i, j) = -((F_e - F_w) / g.dx + (G_n - G_s) / g.dy);
      } else {

        N(i, j) = step_conv_u(g, i, j);
      }
    }
  }
  return N;
}

static Eigen::MatrixXd conv_muscl_v(const MacGrid &g) {
  Eigen::MatrixXd N(g.nx, g.ny + 1);
  N.setZero();
#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 1; i < g.nx - 1; ++i) {
    for (int j = 1; j < g.ny; ++j) {

      bool safe_stencil =
          (i >= 2 && i <= g.nx - 3 && j >= 2 && j <= g.ny - 3 &&
           !g.mask(i - 2, j) && !g.mask(i - 1, j) && !g.mask(i, j) &&
           !g.mask(i + 1, j) && !g.mask(i + 2, j) && !g.mask(i, j - 2) &&
           !g.mask(i, j - 1) && !g.mask(i, j + 1) && !g.mask(i, j + 2));

      if (safe_stencil) {
        // ---- East flux  F_e = ve * (v at east face) ----
        double un = 0.25 * (g.u(i, j) + g.u(i + 1, j) + g.u(i, j - 1) +
                            g.u(i + 1, j - 1));
        double v_e = (un >= 0)
                         ? muscl_face(g.v(i - 1, j), g.v(i, j), g.v(i + 1, j))
                         : muscl_face(g.v(i + 2, j), g.v(i + 1, j), g.v(i, j));
        double F_e = un * v_e;

        // ---- West flux  F_w = uw * (u at west face) ----
        double uw_adv = 0.25 * (g.u(i - 1, j) + g.u(i, j) + g.u(i - 1, j - 1) +
                                g.u(i, j - 1));
        double v_w = (uw_adv >= 0.0)
                         ? muscl_face(g.v(i - 2, j), g.v(i - 1, j), g.v(i, j))
                         : muscl_face(g.v(i + 1, j), g.v(i, j), g.v(i - 1, j));
        double F_w = uw_adv * v_w;

        // ---- North flux  G_n = vn * (u at north face) ----
        double ve_n = 0.5 * (g.v(i, j) + g.v(i, j + 1));
        double v_n = (ve_n >= 0)
                         ? muscl_face(g.v(i, j - 1), g.v(i, j), g.v(i, j + 1))
                         : muscl_face(g.v(i, j + 2), g.v(i, j + 1), g.v(i, j));
        double G_n = ve_n * v_n;

        // ---- South flux  G_s = vs * (u at south face) ----
        double ve_s = 0.5 * (g.v(i, j - 1) + g.v(i, j));
        double v_s = (ve_s >= 0.0)
                         ? muscl_face(g.v(i, j - 2), g.v(i, j - 1), g.v(i, j))
                         : muscl_face(g.v(i, j + 1), g.v(i, j), g.v(i, j - 1));

        double G_s = ve_s * v_s;

        N(i, j) = -((F_e - F_w) / g.dx + (G_n - G_s) / g.dy);
      } else {
        N(i, j) = step_conv_v(g, i, j);
      }
    }
  }
  return N;
}

// ── convection: QUICK
// ────────────────────────────────────────────────────────

// ---------------------------------------------------------------------------
// QUICK face interpolation helper with TVD flux limiting.
// Returns the interpolated value at a cell face given the advecting velocity
// and three candidate stencil values:
//   phi_u2  — two cells upwind
//   phi_u1  — one cell upwind   (= cell just upwind of face)
//   phi_d1  — one cell downwind (= cell just downwind of face)
//   use_quick — false near boundaries → falls back to linear upwind
// ---------------------------------------------------------------------------
static inline double quick_face(double phi_up2, double phi_up, double phi_dn) {
  return (6.0 * phi_up + 3.0 * phi_dn - phi_up2) / 8.0;
}

static Eigen::MatrixXd conv_quick_u(const MacGrid &g) {
  Eigen::MatrixXd N(g.nx + 1, g.ny);
  N.setZero();
#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 1; i < g.nx; ++i) {
    for (int j = 1; j < g.ny - 1; ++j) {
      bool safe_stencil =
          (i >= 2 && i < g.nx - 2 && j >= 2 && j < g.ny - 2 &&
           !g.mask(i - 2, j) && !g.mask(i - 1, j) && !g.mask(i, j) &&
           !g.mask(i + 1, j) && !g.mask(i + 2, j) && !g.mask(i, j - 2) &&
           !g.mask(i, j - 1) && !g.mask(i, j + 1) && !g.mask(i, j + 2));
      if (safe_stencil) {

        // Upwind d(u²)/dx — advecting velocity at east face selects stencil
        double ue = 0.5 * (g.u(i, j) + g.u(i + 1, j));
        double u_e = (ue >= 0.0)
                         ? quick_face(g.u(i - 1, j), g.u(i, j), g.u(i + 1, j))
                         : quick_face(g.u(i + 2, j), g.u(i + 1, j), g.u(i, j));

        // ---- West flux  F_w = (u at west face)^2 ----
        double uw = 0.5 * (g.u(i - 1, j) + g.u(i, j));
        double u_w = (uw >= 0.0)
                         ? quick_face(g.u(i - 2, j), g.u(i - 1, j), g.u(i, j))
                         : quick_face(g.u(i + 1, j), g.u(i, j), g.u(i - 1, j));

        // ---- North flux  G_n = v_n * (u at north face) ----
        double vn = 0.25 * (g.v(i - 1, j + 1) + g.v(i, j + 1) + g.v(i - 1, j) +
                            g.v(i, j));
        double u_n = (vn >= 0.0)
                         ? quick_face(g.u(i, j - 1), g.u(i, j), g.u(i, j + 1))
                         : quick_face(g.u(i, j + 2), g.u(i, j + 1), g.u(i, j));

        // ---- South flux  G_s = v_s * (u at south face) ----
        double vs = 0.25 * (g.v(i - 1, j) + g.v(i, j) + g.v(i - 1, j - 1) +
                            g.v(i, j - 1));
        double u_s = (vs >= 0.0)
                         ? quick_face(g.u(i, j - 2), g.u(i, j - 1), g.u(i, j))
                         : quick_face(g.u(i, j + 1), g.u(i, j), g.u(i, j - 1));

        N(i, j) =
            -((ue * u_e - uw * u_w) / g.dx + (vn * u_n - vs * u_s) / g.dy);
      }

      else {
        N(i, j) = step_conv_u(g, i, j);
      }
    }
  }

  return N;
}

static Eigen::MatrixXd conv_quick_v(const MacGrid &g) {
  Eigen::MatrixXd N(g.nx, g.ny + 1);
  N.setZero();

  for (int i = 1; i < g.nx - 1; ++i) {
    for (int j = 1; j < g.ny; ++j) {
      bool safe_stencil =
          (i >= 2 && i < g.nx - 2 && j >= 2 && j < g.ny - 2 &&
           !g.mask(i - 2, j) && !g.mask(i - 1, j) && !g.mask(i, j) &&
           !g.mask(i + 1, j) && !g.mask(i + 2, j) && !g.mask(i, j - 2) &&
           !g.mask(i, j - 1) && !g.mask(i, j + 1) && !g.mask(i, j + 2));
      if (safe_stencil) {

        // ---- East flux  F_e = u_e * (v at east face) ----
        double un = 0.25 * (g.u(i, j) + g.u(i + 1, j) + g.u(i, j - 1) +
                            g.u(i + 1, j - 1));
        double v_e = (un >= 0.0)
                         ? quick_face(g.v(i - 1, j), g.v(i, j), g.v(i + 1, j))
                         : quick_face(g.v(i + 2, j), g.v(i + 1, j), g.v(i, j));

        // ---- West flux  F_w = u_w * (v at west face) ----
        double uw_adv = 0.25 * (g.u(i - 1, j) + g.u(i, j) + g.u(i - 1, j - 1) +
                                g.u(i, j - 1));
        double v_w = (uw_adv >= 0.0)
                         ? quick_face(g.v(i - 2, j), g.v(i - 1, j), g.v(i, j))
                         : quick_face(g.v(i + 1, j), g.v(i, j), g.v(i - 1, j));

        // ---- North flux  G_n = (v at north face)^2 ----
        double ve_n = 0.5 * (g.v(i, j) + g.v(i, j + 1));
        double v_n = (ve_n >= 0.0)
                         ? quick_face(g.v(i, j - 1), g.v(i, j), g.v(i, j + 1))
                         : quick_face(g.v(i, j + 2), g.v(i, j + 1), g.v(i, j));

        // ---- South flux  G_s = (v at south face)^2 ----
        double ve_s = 0.5 * (g.v(i, j - 1) + g.v(i, j));
        double v_s = (ve_s >= 0.0)
                         ? quick_face(g.v(i, j - 2), g.v(i, j - 1), g.v(i, j))
                         : quick_face(g.v(i, j + 1), g.v(i, j), g.v(i, j - 1));

        N(i, j) = -((un * v_e - uw_adv * v_w) / g.dx +
                    (ve_n * v_n - ve_s * v_s) / g.dy);
      } else {
        N(i, j) = step_conv_v(g, i, j);
      }
    }
  }
  return N;
}

// ── diffusion: central
// ────────────────────────────────────────────────────────

static Eigen::MatrixXd diff_u(const MacGrid &g, double nu) {
  Eigen::MatrixXd D(g.nx + 1, g.ny);
  D.setZero();
  for (int i = 1; i < g.nx; ++i)
    for (int j = 1; j < g.ny - 1; ++j)
      D(i, j) =
          nu *
          ((g.u(i + 1, j) - 2 * g.u(i, j) + g.u(i - 1, j)) / (g.dx * g.dx) +
           (g.u(i, j + 1) - 2 * g.u(i, j) + g.u(i, j - 1)) / (g.dy * g.dy));
  return D;
}

static Eigen::MatrixXd diff_v(const MacGrid &g, double nu) {
  Eigen::MatrixXd D(g.nx, g.ny + 1);
  D.setZero();
  for (int i = 1; i < g.nx - 1; ++i)
    for (int j = 1; j < g.ny; ++j)
      D(i, j) =
          nu *
          ((g.v(i + 1, j) - 2 * g.v(i, j) + g.v(i - 1, j)) / (g.dx * g.dx) +
           (g.v(i, j + 1) - 2 * g.v(i, j) + g.v(i, j - 1)) / (g.dy * g.dy));
  return D;
}

// ── divergence of u*
// ──────────────────────────────────────────────────────────

static Eigen::MatrixXd divergence(const MacGrid &g) {
  Eigen::MatrixXd div(g.nx, g.ny);
  for (int i = 0; i < g.nx; ++i)
    for (int j = 0; j < g.ny; ++j)
      div(i, j) = (g.u(i + 1, j) - g.u(i, j)) / g.dx +
                  (g.v(i, j + 1) - g.v(i, j)) / g.dy;
  return div;
}

// ── public: ns_step
// ───────────────────────────────────────────────────────────

StepStats ns_step(MacGrid &g, NSSolverConfig &cfg, int step,
                  Eigen::MatrixXd &Nu_prev, Eigen::MatrixXd &Nv_prev) {
  // 1. Convection and diffusion
  auto Nu = conv_muscl_u(g);
  auto Nv = conv_muscl_v(g);
  auto Du = diff_u(g, cfg.nu);
  auto Dv = diff_v(g, cfg.nu);

  // 2. Adams-Bashforth (AB2) or forward Euler on first step
  double ab_a = (step == 0) ? 1.0 : 1.5;
  double ab_b = (step == 0) ? 0.0 : -0.5;

  // 3. Intermediate velocity u*
  Eigen::MatrixXd us = g.u + cfg.dt * (ab_a * Nu + ab_b * Nu_prev + Du);
  Eigen::MatrixXd vs = g.v + cfg.dt * (ab_a * Nv + ab_b * Nv_prev + Dv);

  // Swap into grid temporarily for BC + divergence
  g.u = us;
  g.v = vs;
  apply_velocity_bc(g, cfg.U_mean);
  apply_solid_bc(g);

  // 4. Pressure Poisson: ∇²p = (1/dt) ∇·u*
  auto div = divergence(g);
  Eigen::MatrixXd rhs = div / cfg.dt;

  // Set Dirichlet p=0 at outlet before calling solver.
  for (int j = 0; j < g.ny; ++j)
    g.p(g.nx - 1, j) = 0.0;

  auto res = pressure_solve_sor(g.p, rhs, g.mask, g.nx, g.ny, g.dx, g.dy,
                                cfg.omega, cfg.p_tol, cfg.p_max_iter);

  // 5. Projection: u = u* − dt ∇p
  for (int i = 1; i < g.nx; ++i)
    for (int j = 0; j < g.ny; ++j)
      g.u(i, j) -= cfg.dt * (g.p(i, j) - g.p(i - 1, j)) / g.dx;

  for (int i = 0; i < g.nx; ++i)
    for (int j = 1; j < g.ny; ++j)
      g.v(i, j) -= cfg.dt * (g.p(i, j) - g.p(i, j - 1)) / g.dy;

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
  for (int i = 1; i < g.nx - 1; ++i) {
    for (int j = 1; j < g.ny - 1; ++j) {
      if (g.mask(i, j) != 0)
        continue;
      // Skip fluid cells that share a face with a solid cell
      if (g.mask(i - 1, j) || g.mask(i + 1, j) || g.mask(i, j - 1) ||
          g.mask(i, j + 1))
        continue;
      max_div = std::max(max_div, std::abs(div_post(i, j)));
    }
  }
  StepStats stats;
  stats.div_max = max_div;
  stats.p_iters = res.iterations;

  return stats;
}

// ── public: compute_forces
// ──────────────────────────────────────────────────── Uses momentum-flux
// control-volume approach around a box enclosing the cylinder.

ForceCoeffs compute_forces(const MacGrid &g, double U_mean, double D) {
  double scale = 2.0 / (U_mean * U_mean * D);
  double fx = 0.0, fy = 0.0;

  for (int i = 0; i < g.nx; ++i) {
    for (int j = 0; j < g.ny; ++j) {
      if (g.mask(i, j) != 1)
        continue;
      // Pressure force on each solid-cell face adjacent to fluid.
      // F = ∮ p n̂_in dA  (n̂_in = inward normal of the cylinder surface)
      // West face:  n̂_in = +x̂  → F_x = +p_west * dy
      // East face:  n̂_in = -x̂  → F_x = -p_east * dy
      // South face: n̂_in = +ŷ  → F_y = +p_south * dx
      // North face: n̂_in = -ŷ  → F_y = -p_north * dx
      if (i > 0 && g.mask(i - 1, j) == 0)
        fx += g.p(i - 1, j) * g.dy; // west
      if (i < g.nx - 1 && g.mask(i + 1, j) == 0)
        fx -= g.p(i + 1, j) * g.dy; // east
      if (j > 0 && g.mask(i, j - 1) == 0)
        fy += g.p(i, j - 1) * g.dx; // south
      if (j < g.ny - 1 && g.mask(i, j + 1) == 0)
        fy -= g.p(i, j + 1) * g.dx; // north
    }
  }

  return ForceCoeffs{scale * fx, scale * fy};
}

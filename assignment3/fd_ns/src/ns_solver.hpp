#pragma once
#include "mac_grid.hpp"

struct NSSolverConfig {
    double nu;          // kinematic viscosity = U_mean * D / Re
    double dt;          // time step
    double U_mean;      // mean inlet velocity
    double omega;       // SOR relaxation (default 1.8)
    double p_tol;       // pressure convergence tolerance (default 1e-4)
    int    p_max_iter;  // max SOR iterations (default 5000)
};

struct StepStats {
    double div_max;     // max |∇·u| after projection (divergence error)
    int    p_iters;     // SOR iterations used
};

struct ForceCoeffs {
    double cd, cl;      // drag and lift coefficients
};

// Advance grid by one time step.
// On first call (step==0) uses forward Euler for convection; thereafter AB2.
StepStats ns_step(MacGrid &g, NSSolverConfig &cfg, int step,
                  Eigen::MatrixXd &Nu_prev, Eigen::MatrixXd &Nv_prev);

// Compute drag and lift coefficients by momentum flux on a control box
// surrounding the cylinder.
// cd_scale = 2/(U_mean^2 * D);  box: ix0..ix1, jy0..jy1 in cell indices.
ForceCoeffs compute_forces(const MacGrid &g, double U_mean, double D);

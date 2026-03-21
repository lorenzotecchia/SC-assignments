#pragma once
#include <Eigen/Dense>

struct PoissonResult {
    int    iterations;
    double final_residual;
};

// Solve ∇²p = rhs on a nx×ny grid using SOR.
// mask(i,j)==1 → solid cell, skip.
// BCs: Dirichlet p=0 at outlet (i=nx-1), Neumann elsewhere.
// omega: SOR relaxation (1.0–1.95); tol: L∞ convergence criterion.
// p is modified in-place; warm-started from its current values.
PoissonResult pressure_solve_sor(
    Eigen::MatrixXd &p,
    const Eigen::MatrixXd &rhs,
    const Eigen::MatrixXi &mask,
    int nx, int ny, double dx, double dy,
    double omega, double tol, int max_iter);

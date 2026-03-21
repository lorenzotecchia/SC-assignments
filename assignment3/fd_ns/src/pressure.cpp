#include "pressure.hpp"
#include <cmath>
#include <algorithm>

PoissonResult pressure_solve_sor(
    Eigen::MatrixXd &p,
    const Eigen::MatrixXd &rhs,
    const Eigen::MatrixXi &mask,
    int nx, int ny, double dx, double dy,
    double omega, double tol, int max_iter)
{
    const double idx2 = 1.0 / (dx * dx);
    const double idy2 = 1.0 / (dy * dy);
    const double denom = 2.0 * (idx2 + idy2);

    // Dirichlet BCs are set by the caller before this function is invoked.
    // The outlet column (i=nx-1) is skipped in the sweep below and stays fixed.

    int iter = 0;
    double diff = 0.0;

    for (iter = 0; iter < max_iter; ++iter) {
        diff = 0.0;

        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (mask(i,j) == 1) continue;       // solid — skip
                if (i == nx-1)      continue;       // Dirichlet outlet

                // Neumann neighbours: reflect at boundaries.
                double pE = (i < nx-1) ? p(i+1, j) : p(i, j);   // outlet handled above
                double pW = (i > 0)    ? p(i-1, j) : p(i, j);   // inlet Neumann
                double pN = (j < ny-1) ? p(i, j+1) : p(i, j);   // top wall Neumann
                double pS = (j > 0)    ? p(i, j-1) : p(i, j);   // bottom wall Neumann

                double old = p(i, j);
                double pnew = (idx2*(pE + pW) + idy2*(pN + pS) - rhs(i,j)) / denom;
                p(i, j) = (1.0 - omega) * old + omega * pnew;

                double d = std::abs(p(i,j) - old);
                if (d > diff) diff = d;
            }
        }

        if (diff < tol) break;
    }

    return PoissonResult{iter, diff};
}

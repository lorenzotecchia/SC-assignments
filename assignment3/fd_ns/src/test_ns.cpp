#include "mac_grid.hpp"
#include "pressure.hpp"
#include "bc.hpp"
#include <cmath>
#include <cassert>
#include <cstdio>

static void test_poisson_manufactured() {
    // Manufactured solution that satisfies ALL solver BCs exactly:
    //   p(x,y) = cos(kx*x) * cos(ky*y)
    //   kx = pi/(2*Lx)  →  dp/dx|_{x=0} = 0 (Neumann inlet), p(Lx,y) = 0 (Dirichlet outlet)
    //   ky = pi/Ly      →  dp/dy|_{y=0} = dp/dy|_{y=Ly} = 0 (Neumann walls)
    //   Laplacian: -(kx²+ky²)*p
    int nx=40, ny=20;
    double Lx=2.0, Ly=1.0;
    double dx=Lx/nx, dy=Ly/ny;

    Eigen::MatrixXd p   = Eigen::MatrixXd::Zero(nx, ny);
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(nx, ny);
    Eigen::MatrixXi mask= Eigen::MatrixXi::Zero(nx, ny);
    Eigen::MatrixXd exact(nx, ny);

    double kx = M_PI/(2.0*Lx), ky = M_PI/Ly;
    for (int i=0; i<nx; ++i) {
        for (int j=0; j<ny; ++j) {
            double x = (i+0.5)*dx, y = (j+0.5)*dy;
            exact(i,j) = std::cos(kx*x)*std::cos(ky*y);
            rhs(i,j)   = -(kx*kx + ky*ky)*exact(i,j);
        }
    }

    // Pre-set Dirichlet at outlet column (caller's responsibility).
    for (int j=0; j<ny; ++j) p(nx-1,j) = exact(nx-1,j);

    pressure_solve_sor(p, rhs, mask, nx, ny, dx, dy, 1.8, 1e-8, 50000);

    double max_err = 0.0;
    for (int i=0; i<nx-1; ++i)   // skip outlet Dirichlet column
        for (int j=0; j<ny; ++j)
            max_err = std::max(max_err, std::abs(p(i,j) - exact(i,j)));

    printf("[test_poisson] max_err = %.2e  (expect < 1e-2)\n", max_err);
    assert(max_err < 1e-2);
    printf("[test_poisson] PASS\n");
}

int main() {
    test_poisson_manufactured();
    return 0;
}

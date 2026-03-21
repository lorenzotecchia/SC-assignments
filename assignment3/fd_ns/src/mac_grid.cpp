#include "mac_grid.hpp"

MacGrid::MacGrid(int nx, int ny, double Lx, double Ly,
                 double cyl_cx, double cyl_cy, double cyl_r)
    : nx(nx), ny(ny), dx(Lx/nx), dy(Ly/ny), Lx(Lx), Ly(Ly),
      u(Eigen::MatrixXd::Zero(nx+1, ny)),
      v(Eigen::MatrixXd::Zero(nx, ny+1)),
      p(Eigen::MatrixXd::Zero(nx, ny)),
      mask(Eigen::MatrixXi::Zero(nx, ny))
{
    // Mark solid cells whose centres fall inside the cylinder.
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double cx = (i + 0.5) * dx;
            double cy = (j + 0.5) * dy;
            double r2 = (cx - cyl_cx)*(cx - cyl_cx)
                      + (cy - cyl_cy)*(cy - cyl_cy);
            if (r2 <= cyl_r * cyl_r)
                mask(i, j) = 1;
        }
    }
}

void MacGrid::zero_velocity() {
    u.setZero();
    v.setZero();
}

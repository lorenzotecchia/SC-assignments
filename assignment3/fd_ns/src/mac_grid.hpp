#pragma once
#include <Eigen/Dense>

// MAC staggered grid for 2-D incompressible NS.
//
//  u[i][j]  at face (i·dx,       (j+½)·dy),  i ∈ [0,nx],  j ∈ [0,ny-1]
//  v[i][j]  at face ((i+½)·dx,   j·dy),       i ∈ [0,nx-1],j ∈ [0,ny]
//  p[i][j]  at cell ((i+½)·dx, (j+½)·dy),     i ∈ [0,nx-1],j ∈ [0,ny-1]
//  mask[i][j] = 1 if cell centre is inside cylinder (solid), 0 = fluid
struct MacGrid {
    int nx, ny;          // number of pressure cells
    double dx, dy;       // cell size
    double Lx, Ly;       // physical domain size

    Eigen::MatrixXd u;   // (nx+1) × ny
    Eigen::MatrixXd v;   // nx × (ny+1)
    Eigen::MatrixXd p;   // nx × ny
    Eigen::MatrixXi mask;// nx × ny

    // Construct grid and mark cylinder cells in mask.
    // cyl_cx, cyl_cy, cyl_r — cylinder centre and radius in physical units.
    MacGrid(int nx, int ny, double Lx, double Ly,
            double cyl_cx, double cyl_cy, double cyl_r);

    void zero_velocity();  // set u=v=0 (used for init)
};

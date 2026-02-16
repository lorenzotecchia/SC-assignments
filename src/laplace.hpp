#pragma once
#include <Eigen/Dense>
#include <functional>

struct JacobiResult {
  Eigen::MatrixXd solution;
  int iterations;
  double final_residual;
};

JacobiResult
jacobi_solve(int N, double tolerance, int max_iter,
             const std::function<void(Eigen::MatrixXd &)> &apply_boundary);

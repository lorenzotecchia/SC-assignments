#include "laplace.hpp"
#include <fstream>
#include <iostream>

int main() {
  int N = 100;
  double tolerance = 1e-5;
  int max_iter = 100000;

  auto apply_boundary = [](Eigen::MatrixXd &grid) {
    int n = grid.rows();
    // Bottom edge (row 0): c = 0
    grid.row(0).setZero();
    // Top edge (row n-1): c = 1
    grid.row(n - 1).setConstant(1.0);
    // No fixed left/right — periodic BCs are handled in the stencil
  };

  auto result = jacobi_solve(N, tolerance, max_iter, apply_boundary);

  std::cout << "Converged in " << result.iterations << " iterations\n";
  std::cout << "Final residual: " << result.final_residual << "\n";

  // Write solution to file for Python plotting
  std::ofstream out("output/laplace_solution.txt");
  out << result.solution << "\n";
  out.close();

  // Write convergence deltas for comparison plots
  std::ofstream dout("output/laplace_deltas.txt");
  for (double d : result.deltas) {
    dout << d << "\n";
  }
  dout.close();

  return 0;
}

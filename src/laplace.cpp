#include "laplace.hpp"
#include <cmath>
#include <utility>

JacobiResult
jacobi_solve(int N, double tolerance, int max_iter,
             const std::function<void(Eigen::MatrixXd &)> &apply_boundary) {
  Eigen::MatrixXd grid_old = Eigen::MatrixXd::Zero(N, N);
  Eigen::MatrixXd grid_new = Eigen::MatrixXd::Zero(N, N);
  apply_boundary(grid_old);
  apply_boundary(grid_new);

  int iter = 0;
  double diff = 0.0;
  std::vector<double> deltas;
  deltas.reserve(max_iter);

  for (iter = 0; iter < max_iter; iter++) {
    diff = 0.0;

    // #pragma omp parallel for collapse(2) reduction(max:diff) schedule(static)
    for (int i = 1; i < N - 1; i++) {
      for (int j = 0; j < N - 1; j++) {
        int jm = (j - 1 + (N - 1)) % (N - 1); // left neighbor (wraps)
        int jp = (j + 1) % (N - 1);           // right neighbor (wraps)

        grid_new(i, j) = 0.25 * (grid_old(i - 1, j) + grid_old(i + 1, j) +
                                 grid_old(i, jm) + grid_old(i, jp));

        double local_diff = std::abs(grid_new(i, j) - grid_old(i, j));
        if (local_diff > diff) {
          diff = local_diff;
        }
      }
    }

    // Enforce periodicity: last column = first column (sequential, cheap)
    for (int i = 1; i < N - 1; i++) {
      grid_new(i, N - 1) = grid_new(i, 0);
    }
    std::swap(grid_old, grid_new);
    deltas.push_back(diff);

    if (diff < tolerance) {
      break;
    }
  }
  return JacobiResult{grid_old, iter, diff, std::move(deltas)};
}

void fill_mask(Eigen::MatrixXi &mask, std::vector<Rect> rects, CellType type) {
  // Cast enum to int so Eigen can store it in MatrixXi
  int val = static_cast<int>(type);
  for (const Rect &r : rects) {
    for (int i = r.y0; i <= r.y1; ++i) {
      for (int j = r.x0; j <= r.x1; ++j) {
        mask(i, j) = val;
      }
    }
  }
}

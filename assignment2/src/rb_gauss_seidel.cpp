#include "library.hpp"

SorResult
rb_gauss_seidel_solve(int N, double tolerance, int max_iter,
                      const Eigen::MatrixXi &mask,
                      const std::function<void(Eigen::MatrixXd &)> &apply_boundary) {

  Eigen::MatrixXd grid = Eigen::MatrixXd::Zero(N, N);
  apply_boundary(grid);

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      if (mask(i, j) == static_cast<int>(CellType::OCCUPIED))
        grid(i, j) = 0.0;

  int iter = 0;
  double diff = 0.0;
  std::vector<double> deltas;
  deltas.reserve(max_iter);

  for (iter = 0; iter < max_iter; ++iter) {
    diff = 0.0;

    // ── Red pass: cells where (i+j) is even ──
    #pragma omp parallel for collapse(2) reduction(max:diff) schedule(static)
    for (int i = 1; i < N - 1; ++i) {
      for (int j = 0; j < N - 1; ++j) {
        if ((i + j) % 2 != 0)
          continue;
        if (mask(i, j) == static_cast<int>(CellType::OCCUPIED))
          continue;

        int jm = (j - 1 + (N - 1)) % (N - 1);
        int jp = (j + 1) % (N - 1);

        double old_val = grid(i, j);
        double avg = 0.25 * (grid(i + 1, j) + grid(i - 1, j) +
                             grid(i, jp) + grid(i, jm));
        grid(i, j) = avg;

        double local_diff = std::abs(grid(i, j) - old_val);
        if (local_diff > diff)
          diff = local_diff;
      }
    }

    // ── Black pass: cells where (i+j) is odd ──
    #pragma omp parallel for collapse(2) reduction(max:diff) schedule(static)
    for (int i = 1; i < N - 1; ++i) {
      for (int j = 0; j < N - 1; ++j) {
        if ((i + j) % 2 != 1)
          continue;
        if (mask(i, j) == static_cast<int>(CellType::OCCUPIED))
          continue;

        int jm = (j - 1 + (N - 1)) % (N - 1);
        int jp = (j + 1) % (N - 1);

        double old_val = grid(i, j);
        double avg = 0.25 * (grid(i + 1, j) + grid(i - 1, j) +
                             grid(i, jp) + grid(i, jm));
        grid(i, j) = avg;

        double local_diff = std::abs(grid(i, j) - old_val);
        if (local_diff > diff)
          diff = local_diff;
      }
    }

    // Enforce periodicity: last column mirrors first column.
    for (int i = 1; i < N - 1; ++i)
      grid(i, N - 1) = grid(i, 0);

    deltas.push_back(diff);
    if (diff < tolerance)
      break;
  }

  return SorResult{grid, iter, diff, std::move(deltas)};
}

#include "library.hpp"

SorResult
sor_solve(int N, double omega, double tolerance, int max_iter,
          const Eigen::MatrixXi &mask,
          const std::function<void(Eigen::MatrixXd &)> &apply_boundary,
          const Eigen::MatrixXd *initial_guess) {

  Eigen::MatrixXd grid = initial_guess ? *initial_guess
                                       : Eigen::MatrixXd::Zero(N, N);
  apply_boundary(grid);

  // Enforce occupied (aggregate) cells to 0 in the initial state.
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

    for (int i = 1; i < N - 1; ++i) {
      for (int j = 0; j < N - 1; ++j) {

        int jm = (j - 1 + (N - 1)) % (N - 1);
        int jp = (j + 1) % (N - 1);

        if (mask(i, j) == static_cast<int>(CellType::OCCUPIED))
          continue;

        double c_north = grid(i + 1, j);
        double c_south = grid(i - 1, j);
        double c_east = grid(i, jp);
        double c_west = grid(i, jm);

        double old_val = grid(i, j);
        double avg = 0.25 * (c_north + c_south + c_east + c_west);
        grid(i, j) = omega * avg + (1.0 - omega) * old_val;

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

// ── DLA helpers ──────────────────────────────────────────────────────────────

std::vector<std::pair<int, int>> find_candidates(int N,
                                                 const Eigen::MatrixXi &mask) {
  std::vector<std::pair<int, int>> candidates;
  // Offsets: up, down, left, right.
  static const int di[] = {-1, 1, 0, 0};
  static const int dj[] = {0, 0, -1, 1};

  for (int i = 1; i < N - 1; ++i) {
    for (int j = 0; j < N; ++j) {
      if (mask(i, j) != static_cast<int>(CellType::EMPTY))
        continue;
      for (int d = 0; d < 4; ++d) {
        int ni = i + di[d];
        // Periodic in x (columns).
        int nj = (j + dj[d] + N) % N;
        if (ni >= 0 && ni < N &&
            mask(ni, nj) == static_cast<int>(CellType::OCCUPIED)) {
          candidates.emplace_back(i, j);
          break; // only add once
        }
      }
    }
  }
  return candidates;
}

std::vector<double>
compute_pg(const std::vector<std::pair<int, int>> &candidates,
           const Eigen::MatrixXd &concentration, double eta) {
  std::vector<double> pg(candidates.size());
  double sum = 0.0;
  for (size_t k = 0; k < candidates.size(); ++k) {
    auto [i, j] = candidates[k];
    pg[k] = std::pow(concentration(i, j), eta);
    sum += pg[k];
  }
  if (sum > 0.0) {
    for (auto &p : pg)
      p /= sum;
  }
  return pg;
}

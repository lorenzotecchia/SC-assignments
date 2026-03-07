/**
 * main_scaling.cpp — DLA grid-size scaling benchmark.
 *
 * Runs DLA with warm-start SOR for N ∈ {50, 100, 200}, records per-step
 * solve time and iteration count, and writes output/scaling.csv.
 */
#include "library.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

using Clock = std::chrono::high_resolution_clock;
using us    = std::chrono::microseconds;

static double elapsed_us(Clock::time_point s, Clock::time_point e) {
  return std::chrono::duration_cast<us>(e - s).count();
}

int main() {
  const std::vector<int> grid_sizes = {50, 100, 200};
  const double tol       = 1e-5;
  const int    max_iter   = 100000;
  const double omega      = 1.95;
  const double eta        = 1.0;
  const int    growth_steps = 200;
  const unsigned seed     = 42;

  auto apply_bc = [](Eigen::MatrixXd &g) {
    g.row(0).setZero();
    g.row(g.rows() - 1).setConstant(1.0);
  };

  std::ofstream csv("output/scaling.csv");
  csv << "N,step,solve_ms,sor_iters\n";

  std::cout << "═══ Grid-Size Scaling Benchmark ═══\n"
            << "steps=" << growth_steps << "  seed=" << seed
            << "  eta=" << eta << "\n\n";

  for (int N : grid_sizes) {
    std::cout << "── N=" << N << " ──\n";

    Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(N, N);
    mask(0, N / 2) = static_cast<int>(CellType::OCCUPIED);

    // Analytical gradient initial guess.
    Eigen::MatrixXd init(N, N);
    for (int i = 0; i < N; ++i)
      init.row(i).setConstant(static_cast<double>(i) / (N - 1));

    auto sres = sor_solve(N, omega, tol, max_iter, mask, apply_bc, &init);

    std::mt19937 rng(seed);
    double total_ms = 0;

    for (int step = 0; step < growth_steps; ++step) {
      auto cands = find_candidates(N, mask);
      if (cands.empty()) {
        std::cout << "  no candidates at step " << step << "\n";
        break;
      }

      auto pg = compute_pg(cands, sres.solution, eta);
      std::discrete_distribution<size_t> dist(pg.begin(), pg.end());
      auto [ci, cj] = cands[dist(rng)];
      mask(ci, cj) = static_cast<int>(CellType::OCCUPIED);

      auto t0 = Clock::now();
      sres = sor_solve(N, omega, tol, max_iter, mask, apply_bc,
                       &sres.solution);
      auto t1 = Clock::now();
      double ms = elapsed_us(t0, t1) / 1e3;
      total_ms += ms;

      csv << N << "," << step << "," << ms << "," << sres.iterations << "\n";
    }

    std::cout << "  total solve: " << total_ms << " ms\n";
  }

  std::cout << "\nCSV written to output/scaling.csv\n";
  return 0;
}

#include "library.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

using Clock = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

static double elapsed_us(Clock::time_point s, Clock::time_point e) {
  return std::chrono::duration_cast<us>(e - s).count();
}

// Run a full DLA growth loop and return (total_solver_iters, total_solve_ms).
template <typename SolveFn>
static std::pair<long, double>
run_dla(int N, double eta, int growth_steps,
        SolveFn solve_fn) {

  auto apply_boundary = [](Eigen::MatrixXd &grid) {
    int n = grid.rows();
    grid.row(0).setZero();
    grid.row(n - 1).setConstant(1.0);
  };

  Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(N, N);
  mask(0, N / 2) = static_cast<int>(CellType::OCCUPIED);

  // Analytical gradient initial guess for first solve.
  Eigen::MatrixXd init_guess(N, N);
  for (int i = 0; i < N; ++i)
    init_guess.row(i).setConstant(static_cast<double>(i) / (N - 1));

  auto res = solve_fn(N, mask, apply_boundary, &init_guess);
  long total_iters = res.iterations;
  double total_solve_us = 0.0;

  std::mt19937 rng(42);

  for (int step = 0; step < growth_steps; ++step) {
    auto candidates = find_candidates(N, mask);
    if (candidates.empty())
      break;

    auto pg = compute_pg(candidates, res.solution, eta);
    std::discrete_distribution<size_t> dist(pg.begin(), pg.end());
    size_t chosen = dist(rng);
    auto [ci, cj] = candidates[chosen];
    mask(ci, cj) = static_cast<int>(CellType::OCCUPIED);

    auto t0 = Clock::now();
    res = solve_fn(N, mask, apply_boundary, &res.solution);
    auto t1 = Clock::now();

    total_iters += res.iterations;
    total_solve_us += elapsed_us(t0, t1);
  }

  return {total_iters, total_solve_us / 1e3};
}

int main() {
  const int N = 100;
  const double tolerance = 1e-5;
  const int max_iter = 100000;
  const int growth_steps = 200;

  const double etas[] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0};
  const double omegas[] = {1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 1.95};

  // ── SOR sweep: eta × omega ──
  {
    std::ofstream out("output/sweep_eta_omega.csv");
    out << "eta,omega,total_iters,total_time_ms\n";

    for (double eta : etas) {
      for (double omega : omegas) {
        std::cout << "SOR  eta=" << eta << " omega=" << omega << " ... "
                  << std::flush;

        auto solve_fn = [&](int n, const Eigen::MatrixXi &mask,
                            const std::function<void(Eigen::MatrixXd &)> &bc,
                            const Eigen::MatrixXd *guess) {
          return sor_solve(n, omega, tolerance, max_iter, mask, bc, guess);
        };

        auto [iters, time_ms] = run_dla(N, eta, growth_steps, solve_fn);
        out << eta << "," << omega << "," << iters << "," << time_ms << "\n";
        std::cout << iters << " iters, " << time_ms << " ms\n";
      }
    }
  }

  // ── Red-Black Gauss-Seidel sweep: eta only ──
  {
    std::ofstream out("output/sweep_rbgs.csv");
    out << "eta,total_iters,total_time_ms\n";

    for (double eta : etas) {
      std::cout << "RBGS eta=" << eta << " ... " << std::flush;

      auto solve_fn = [&](int n, const Eigen::MatrixXi &mask,
                          const std::function<void(Eigen::MatrixXd &)> &bc,
                          const Eigen::MatrixXd *guess) {
        return rb_gauss_seidel_solve(n, tolerance, max_iter, mask, bc, guess);
      };

      auto [iters, time_ms] = run_dla(N, eta, growth_steps, solve_fn);
      out << eta << "," << iters << "," << time_ms << "\n";
      std::cout << iters << " iters, " << time_ms << " ms\n";
    }
  }

  std::cout << "\ndone. outputs: output/sweep_eta_omega.csv, "
               "output/sweep_rbgs.csv\n";
  return 0;
}

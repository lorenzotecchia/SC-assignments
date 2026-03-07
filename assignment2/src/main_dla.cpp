#include "library.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

using Clock = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

static double elapsed_us(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration_cast<us>(end - start).count();
}

static void write_matrix(const std::string &path, const Eigen::MatrixXd &m) {
  std::ofstream out(path);
  out << m << "\n";
}

static void write_mask(const std::string &path, const Eigen::MatrixXi &m) {
  std::ofstream out(path);
  out << m << "\n";
}

// Append a concentration snapshot to the snapshots file.
static void append_snapshot(std::ofstream &snap_file, int step,
                            const Eigen::MatrixXd &m) {
  int N = m.rows();
  snap_file << "# step " << step << "\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (j > 0)
        snap_file << " ";
      snap_file << m(i, j);
    }
    snap_file << "\n";
  }
}

static void write_profile(const std::string &path,
                          const std::vector<StepProfile> &profiles) {
  std::ofstream out(path);
  out << "step candidates_us pg_us select_us solve_us snapshot_us sor_iters\n";
  for (const auto &p : profiles) {
    out << p.step << " " << p.candidates_us << " " << p.pg_us << " "
        << p.select_us << " " << p.solve_us << " " << p.snapshot_us << " "
        << p.sor_iters << "\n";
  }
}

int main() {
  const int N = 100;
  const double tolerance = 1e-5;
  const int max_iter = 100000;
  const double omega = 1.95;
  const double eta = 1.0;
  const int growth_steps = 1000;
  const int snap_interval = 5;
  const std::vector<unsigned> seeds = {100, 200, 300, 400, 500};
  const int n_runs = static_cast<int>(seeds.size());

  auto apply_boundary = [](Eigen::MatrixXd &grid) {
    int n = grid.rows();
    grid.row(0).setZero();
    grid.row(n - 1).setConstant(1.0);
  };

  std::cout << "═══ DLA SOR (multi-seed) ═══\n"
            << "N=" << N << "  steps=" << growth_steps << "  eta=" << eta
            << "  omega=" << omega << "  runs=" << n_runs << "\n\n";

  // Multi-run CSV
  std::ofstream multi_csv("output/dla_multi.csv");
  multi_csv << "seed,step,solve_ms,sor_iters\n";

  for (int r = 0; r < n_runs; ++r) {
    unsigned seed = seeds[r];
    bool first_run = (r == 0);

    std::cout << "── run " << (r + 1) << "/" << n_runs
              << " (seed=" << seed << ") ──\n";

    Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(N, N);
    mask(0, N / 2) = static_cast<int>(CellType::OCCUPIED);

    auto sres = sor_solve(N, omega, tolerance, max_iter, mask, apply_boundary);

    // Only first run writes snapshots, growth log, and profile.
    std::ofstream growth_log, snap_file;
    std::vector<StepProfile> profiles;
    if (first_run) {
      growth_log.open("output/dla_growth_log.txt");
      growth_log << "# step i j\n";
      snap_file.open("output/dla_snapshots.txt");
      append_snapshot(snap_file, 0, sres.solution);
      profiles.reserve(growth_steps);
    }

    std::mt19937 rng(seed);
    auto t_loop_start = Clock::now();

    for (int step = 0; step < growth_steps; ++step) {
      StepProfile prof{};
      prof.step = step;

      auto t0 = Clock::now();
      auto candidates = find_candidates(N, mask);
      auto t1 = Clock::now();
      prof.candidates_us = elapsed_us(t0, t1);

      if (candidates.empty()) {
        std::cout << "  no candidates at step " << step << ", stopping.\n";
        break;
      }

      t0 = Clock::now();
      auto pg = compute_pg(candidates, sres.solution, eta);
      t1 = Clock::now();
      prof.pg_us = elapsed_us(t0, t1);

      t0 = Clock::now();
      std::discrete_distribution<size_t> dist(pg.begin(), pg.end());
      size_t chosen = dist(rng);
      auto [ci, cj] = candidates[chosen];
      mask(ci, cj) = static_cast<int>(CellType::OCCUPIED);
      if (first_run)
        growth_log << step << " " << ci << " " << cj << "\n";
      t1 = Clock::now();
      prof.select_us = elapsed_us(t0, t1);

      t0 = Clock::now();
      sres = sor_solve(N, omega, tolerance, max_iter, mask, apply_boundary);
      t1 = Clock::now();
      prof.solve_us = elapsed_us(t0, t1);
      prof.sor_iters = sres.iterations;

      // Per-step row for multi-run CSV
      multi_csv << seed << "," << step << "," << prof.solve_us / 1e3 << ","
                << prof.sor_iters << "\n";

      if (first_run) {
        t0 = Clock::now();
        if ((step + 1) % snap_interval == 0)
          append_snapshot(snap_file, step + 1, sres.solution);
        t1 = Clock::now();
        prof.snapshot_us = elapsed_us(t0, t1);
        profiles.push_back(prof);
      }

      if ((step + 1) % 250 == 0)
        std::cout << "  step " << step + 1 << ": SOR " << sres.iterations
                  << " iters, solve=" << prof.solve_us / 1e3 << " ms\n";
    }

    auto t_loop_end = Clock::now();
    double total_ms = elapsed_us(t_loop_start, t_loop_end) / 1e3;
    std::cout << "  total: " << total_ms << " ms\n";

    if (first_run) {
      append_snapshot(snap_file, growth_steps, sres.solution);
      write_matrix("output/dla_concentration.txt", sres.solution);
      write_mask("output/dla_cluster.txt", mask);
      write_profile("output/dla_profile.txt", profiles);
    }
  }

  std::cout << "\ndone. outputs written to output/\n";
  return 0;
}

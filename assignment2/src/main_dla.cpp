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
  const double eta = 1.0;        // morphology parameter
  const int growth_steps = 1000; // number of cells to grow
  const int snap_interval = 5;    // save snapshot every N steps

  // Boundary conditions: bottom row c=0 (cluster seed), top row c=1.
  auto apply_boundary = [](Eigen::MatrixXd &grid) {
    int n = grid.rows();
    grid.row(0).setZero();
    grid.row(n - 1).setConstant(1.0);
  };

  // ── Initialise mask with seed at bottom-centre ──
  Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(N, N);
  mask(0, N / 2) = static_cast<int>(CellType::OCCUPIED);

  // ── Initial Laplace solve (profiled) ──
  auto t_init_start = Clock::now();
  auto sres = sor_solve(N, omega, tolerance, max_iter, mask, apply_boundary);
  auto t_init_end = Clock::now();
  std::cout << "initial solve: " << sres.iterations
            << " iters, residual=" << sres.final_residual
            << ", time=" << elapsed_us(t_init_start, t_init_end) / 1e3
            << " ms\n";

  // ── Open output files ──
  std::ofstream growth_log("output/dla_growth_log.txt");
  growth_log << "# step i j\n";

  std::ofstream snap_file("output/dla_snapshots.txt");
  append_snapshot(snap_file, 0, sres.solution);

  // ── DLA growth loop ──
  std::mt19937 rng(100);
  std::vector<StepProfile> profiles;
  profiles.reserve(growth_steps);
  auto t_loop_start = Clock::now();

  for (int step = 0; step < growth_steps; ++step) {
    StepProfile prof{};
    prof.step = step;

    // 1. Find candidates
    auto t0 = Clock::now();
    auto candidates = find_candidates(N, mask);
    auto t1 = Clock::now();
    prof.candidates_us = elapsed_us(t0, t1);

    if (candidates.empty()) {
      std::cout << "no candidates at step " << step << ", stopping.\n";
      break;
    }

    // 2. Compute p_g
    t0 = Clock::now();
    auto pg = compute_pg(candidates, sres.solution, eta);
    t1 = Clock::now();
    prof.pg_us = elapsed_us(t0, t1);

    // 3. Stochastic selection
    t0 = Clock::now();
    std::discrete_distribution<size_t> dist(pg.begin(), pg.end());
    size_t chosen = dist(rng);
    auto [ci, cj] = candidates[chosen];
    mask(ci, cj) = static_cast<int>(CellType::OCCUPIED);
    growth_log << step << " " << ci << " " << cj << "\n";
    t1 = Clock::now();
    prof.select_us = elapsed_us(t0, t1);

    // 4. SOR solve (warm-start)
    t0 = Clock::now();
    sres = sor_solve(N, omega, tolerance, max_iter, mask, apply_boundary);
    t1 = Clock::now();
    prof.solve_us = elapsed_us(t0, t1);
    prof.sor_iters = sres.iterations;

    // 5. Snapshot I/O
    t0 = Clock::now();
    if ((step + 1) % snap_interval == 0)
      append_snapshot(snap_file, step + 1, sres.solution);
    t1 = Clock::now();
    prof.snapshot_us = elapsed_us(t0, t1);

    profiles.push_back(prof);

    if ((step + 1) % 50 == 0)
      std::cout << "step " << step + 1 << ": added (" << ci << "," << cj
                << "), SOR " << sres.iterations
                << " iters, solve=" << " iters, solve=" << prof.solve_us / 1e3
                << " ms\n";
  }

  auto t_loop_end = Clock::now();

  // Final snapshot.
  append_snapshot(snap_file, growth_steps, sres.solution);

  write_matrix("output/dla_concentration.txt", sres.solution);
  write_mask("output/dla_cluster.txt", mask);
  write_profile("output/dla_profile.txt", profiles);

  // ── Summary ──
  double total_ms = elapsed_us(t_loop_start, t_loop_end) / 1e3;
  double total_solve = 0, total_cand = 0, total_pg = 0, total_sel = 0,
         total_snap = 0;
  for (const auto &p : profiles) {
    total_solve += p.solve_us;
    total_cand += p.candidates_us;
    total_pg += p.pg_us;
    total_sel += p.select_us;
    total_snap += p.snapshot_us;
  }
  double total_all =
      total_solve + total_cand + total_pg + total_sel + total_snap;

  std::cout << "\n═══ PROFILE SUMMARY (" << profiles.size() << " steps, "
            << total_ms << " ms total) ═══\n";
  std::cout << "  SOR solve:       " << total_solve / 1e3 << " ms ("
            << 100.0 * total_solve / total_all << " %)\n";
  std::cout << "  find_candidates: " << total_cand / 1e3 << " ms ("
            << 100.0 * total_cand / total_all << " %)\n";
  std::cout << "  compute_pg:      " << total_pg / 1e3 << " ms ("
            << 100.0 * total_pg / total_all << " %)\n";
  std::cout << "  selection:       " << total_sel / 1e3 << " ms ("
            << 100.0 * total_sel / total_all << " %)\n";
  std::cout << "  snapshot I/O:    " << total_snap / 1e3 << " ms ("
            << 100.0 * total_snap / total_all << " %)\n";

  std::cout << "done. outputs written to output/\n";
  return 0;
}

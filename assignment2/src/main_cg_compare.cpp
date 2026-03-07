#include "library.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>

using Clock = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

static double elapsed_us(Clock::time_point s, Clock::time_point e) {
  return std::chrono::duration_cast<us>(e - s).count();
}

struct StepRecord {
  int step;
  double solve_ms;
  int iters;
};

struct RunStats {
  double total_solve_ms;
  int total_solver_iters;
  int growth_steps_done;
  Eigen::MatrixXd final_conc;
  Eigen::MatrixXi final_mask;
  std::vector<StepRecord> records;
};

static void append_snapshot(std::ofstream &f, int step,
                            const Eigen::MatrixXd &m) {
  int N = m.rows();
  f << "# step " << step << "\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (j > 0) f << " ";
      f << m(i, j);
    }
    f << "\n";
  }
}

// ── DLA growth with SOR ──────────────────────────────────────────────────────
static RunStats run_dla_sor(int N, int growth_steps, double omega, double tol,
                            int max_iter, unsigned seed, double eta) {
  auto apply_bc = [](Eigen::MatrixXd &g) {
    g.row(0).setZero();
    g.row(g.rows() - 1).setConstant(1.0);
  };

  Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(N, N);
  mask(0, N / 2) = static_cast<int>(CellType::OCCUPIED);
  auto sres = sor_solve(N, omega, tol, max_iter, mask, apply_bc);

  std::mt19937 rng(seed);
  double total_solve = 0;
  int total_iters = 0;
  int steps = 0;
  std::vector<StepRecord> records;
  records.reserve(growth_steps);

  for (int step = 0; step < growth_steps; ++step) {
    auto cands = find_candidates(N, mask);
    if (cands.empty())
      break;

    auto pg = compute_pg(cands, sres.solution, eta);
    std::discrete_distribution<size_t> dist(pg.begin(), pg.end());
    auto [ci, cj] = cands[dist(rng)];
    mask(ci, cj) = static_cast<int>(CellType::OCCUPIED);

    auto t0 = Clock::now();
    sres = sor_solve(N, omega, tol, max_iter, mask, apply_bc);
    auto t1 = Clock::now();
    double ms = elapsed_us(t0, t1) / 1e3;
    total_solve += ms;
    total_iters += sres.iterations;
    steps = step + 1;
    records.push_back({step, ms, sres.iterations});
  }

  return {total_solve, total_iters, steps, sres.solution, mask,
          std::move(records)};
}

// ── DLA growth with CG ──────────────────────────────────────────────────────
static RunStats run_dla_cg(int N, int growth_steps, double tol, int max_iter,
                           unsigned seed, double eta, int snap_interval) {
  auto apply_bc = [](Eigen::MatrixXd &g) {
    g.row(0).setZero();
    g.row(g.rows() - 1).setConstant(1.0);
  };

  Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(N, N);
  mask(0, N / 2) = static_cast<int>(CellType::OCCUPIED);
  auto sres = cg_solve(N, tol, max_iter, mask, apply_bc);

  std::ofstream growth_log("output/cg_growth_log.txt");
  growth_log << "# step i j\n";
  std::ofstream snap_file("output/cg_snapshots.txt");
  append_snapshot(snap_file, 0, sres.solution);

  std::mt19937 rng(seed);
  double total_solve = 0;
  int total_iters = 0;
  int steps = 0;
  std::vector<StepRecord> records;
  records.reserve(growth_steps);

  for (int step = 0; step < growth_steps; ++step) {
    auto cands = find_candidates(N, mask);
    if (cands.empty())
      break;

    auto pg = compute_pg(cands, sres.solution, eta);
    std::discrete_distribution<size_t> dist(pg.begin(), pg.end());
    auto [ci, cj] = cands[dist(rng)];
    mask(ci, cj) = static_cast<int>(CellType::OCCUPIED);
    growth_log << step << " " << ci << " " << cj << "\n";

    auto t0 = Clock::now();
    sres = cg_solve(N, tol, max_iter, mask, apply_bc);
    auto t1 = Clock::now();
    double ms = elapsed_us(t0, t1) / 1e3;
    total_solve += ms;
    total_iters += sres.iterations;
    steps = step + 1;
    records.push_back({step, ms, sres.iterations});

    if ((step + 1) % snap_interval == 0)
      append_snapshot(snap_file, step + 1, sres.solution);
  }
  append_snapshot(snap_file, growth_steps, sres.solution);

  return {total_solve, total_iters, steps, sres.solution, mask,
          std::move(records)};
}

// CG run without snapshot/growth-log I/O (for multi-seed runs).
static RunStats run_dla_cg_no_snap(int N, int growth_steps, double tol,
                                   int max_iter, unsigned seed, double eta) {
  auto apply_bc = [](Eigen::MatrixXd &g) {
    g.row(0).setZero();
    g.row(g.rows() - 1).setConstant(1.0);
  };

  Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(N, N);
  mask(0, N / 2) = static_cast<int>(CellType::OCCUPIED);
  auto sres = cg_solve(N, tol, max_iter, mask, apply_bc);

  std::mt19937 rng(seed);
  double total_solve = 0;
  int total_iters = 0;
  int steps = 0;
  std::vector<StepRecord> records;
  records.reserve(growth_steps);

  for (int step = 0; step < growth_steps; ++step) {
    auto cands = find_candidates(N, mask);
    if (cands.empty())
      break;

    auto pg = compute_pg(cands, sres.solution, eta);
    std::discrete_distribution<size_t> dist(pg.begin(), pg.end());
    auto [ci, cj] = cands[dist(rng)];
    mask(ci, cj) = static_cast<int>(CellType::OCCUPIED);

    auto t0 = Clock::now();
    sres = cg_solve(N, tol, max_iter, mask, apply_bc);
    auto t1 = Clock::now();
    double ms = elapsed_us(t0, t1) / 1e3;
    total_solve += ms;
    total_iters += sres.iterations;
    steps = step + 1;
    records.push_back({step, ms, sres.iterations});
  }

  return {total_solve, total_iters, steps, sres.solution, mask,
          std::move(records)};
}

static void write_csv(const std::string &path,
                      const std::vector<StepRecord> &sor_rec,
                      const std::vector<StepRecord> &cg_rec) {
  std::ofstream out(path);
  out << "step,sor_solve_ms,sor_iters,cg_solve_ms,cg_iters\n";
  size_t n = std::min(sor_rec.size(), cg_rec.size());
  for (size_t i = 0; i < n; ++i) {
    out << sor_rec[i].step << "," << sor_rec[i].solve_ms << ","
        << sor_rec[i].iters << "," << cg_rec[i].solve_ms << ","
        << cg_rec[i].iters << "\n";
  }
}

static void write_conc_diff(const std::string &path,
                            const Eigen::MatrixXd &diff) {
  std::ofstream out(path);
  out << diff << "\n";
}

// ── main ─────────────────────────────────────────────────────────────────────
int main() {
  const int N = 100;
  const double tol = 1e-5;
  const int max_iter = 100000;
  const double omega = 1.95;
  const double eta = 1.0;
  const int growth_steps = 200;
  const int snap_interval = 5;
  const std::vector<unsigned> seeds = {42, 123, 256, 314, 999};
  const int n_runs = static_cast<int>(seeds.size());

  std::cout << "═══ DLA Solver Comparison: SOR vs Eigen-CG ═══\n"
            << "N=" << N << "  steps=" << growth_steps << "  eta=" << eta
            << "  tol=" << tol << "  runs=" << n_runs << "\n\n";

  // ── Multi-run CSV: seed, step, solver, solve_ms, iters ──
  std::ofstream multi_csv("output/cg_compare_multi.csv");
  multi_csv << "seed,step,solver,solve_ms,iters\n";

  // Accumulators for the summary table.
  double sor_wall_total = 0, cg_wall_total = 0;
  double sor_solve_total = 0, cg_solve_total = 0;
  int sor_iters_total = 0, cg_iters_total = 0;

  // We keep the last seed's results for single-run outputs (animation, diff).
  RunStats last_sor, last_cg;

  for (int r = 0; r < n_runs; ++r) {
    unsigned seed = seeds[r];
    std::cout << "── run " << (r + 1) << "/" << n_runs
              << " (seed=" << seed << ") ──\n";

    auto t0 = Clock::now();
    auto sor = run_dla_sor(N, growth_steps, omega, tol, max_iter, seed, eta);
    auto t1 = Clock::now();
    double sor_wall = elapsed_us(t0, t1) / 1e3;

    t0 = Clock::now();
    RunStats cg_run;
    // Only write snapshots/growth-log for the first seed.
    if (r == 0)
      cg_run = run_dla_cg(N, growth_steps, tol, max_iter, seed, eta,
                           snap_interval);
    else
      cg_run = run_dla_cg_no_snap(N, growth_steps, tol, max_iter, seed, eta);
    t1 = Clock::now();
    double cg_wall = elapsed_us(t0, t1) / 1e3;

    sor_wall_total += sor_wall;
    cg_wall_total += cg_wall;
    sor_solve_total += sor.total_solve_ms;
    cg_solve_total += cg_run.total_solve_ms;
    sor_iters_total += sor.total_solver_iters;
    cg_iters_total += cg_run.total_solver_iters;

    for (const auto &rec : sor.records)
      multi_csv << seed << "," << rec.step << ",SOR," << rec.solve_ms << ","
                << rec.iters << "\n";
    for (const auto &rec : cg_run.records)
      multi_csv << seed << "," << rec.step << ",CG," << rec.solve_ms << ","
                << rec.iters << "\n";

    last_sor = std::move(sor);
    last_cg = std::move(cg_run);

    std::cout << "  SOR: " << std::fixed << std::setprecision(1) << sor_wall
              << " ms   CG: " << cg_wall << " ms\n";
  }

  // ── Single-seed outputs (backward compat) ──
  write_csv("output/cg_compare.csv", last_sor.records, last_cg.records);
  write_conc_diff("output/cg_conc_diff.txt",
                  (last_sor.final_conc - last_cg.final_conc).cwiseAbs());

  // ── Summary ──
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "\n═══ Aggregate over " << n_runs << " runs ═══\n";
  std::cout << std::setw(24) << "" << std::setw(14) << "SOR" << std::setw(14)
            << "CG" << "\n";
  std::cout << std::setw(24) << "Mean wall time (ms):" << std::setw(14)
            << sor_wall_total / n_runs << std::setw(14)
            << cg_wall_total / n_runs << "\n";
  std::cout << std::setw(24) << "Mean solve time (ms):" << std::setw(14)
            << sor_solve_total / n_runs << std::setw(14)
            << cg_solve_total / n_runs << "\n";
  std::cout << std::setw(24) << "Mean solver iters:" << std::setw(14)
            << sor_iters_total / n_runs << std::setw(14)
            << cg_iters_total / n_runs << "\n";

  double max_conc_diff =
      (last_sor.final_conc - last_cg.final_conc).cwiseAbs().maxCoeff();
  int mask_diff = (last_sor.final_mask - last_cg.final_mask).cwiseAbs().sum();
  std::cout << std::scientific << std::setprecision(4);
  std::cout << "\nLast-run max |c_SOR - c_CG|: " << max_conc_diff << "\n";
  std::cout << "Last-run cluster diffs:      " << mask_diff << "\n";

  std::cout << "\nCSV written to output/cg_compare_multi.csv\n";
  return 0;
}

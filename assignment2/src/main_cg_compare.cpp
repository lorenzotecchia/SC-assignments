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
  const unsigned seed = 42;
  const int snap_interval = 5;

  std::cout << "═══ DLA Solver Comparison: SOR vs Eigen-CG ═══\n"
            << "N=" << N << "  steps=" << growth_steps << "  eta=" << eta
            << "  tol=" << tol << "\n\n";

  std::cout << "Running SOR (omega=" << omega << ") ...\n";
  auto t0 = Clock::now();
  auto sor = run_dla_sor(N, growth_steps, omega, tol, max_iter, seed, eta);
  auto t1 = Clock::now();
  double sor_wall = elapsed_us(t0, t1) / 1e3;

  std::cout << "Running CG  ...\n";
  t0 = Clock::now();
  auto cg = run_dla_cg(N, growth_steps, tol, max_iter, seed, eta, snap_interval);
  t1 = Clock::now();
  double cg_wall = elapsed_us(t0, t1) / 1e3;

  double max_conc_diff =
      (sor.final_conc - cg.final_conc).cwiseAbs().maxCoeff();
  int mask_diff = (sor.final_mask - cg.final_mask).cwiseAbs().sum();

  // Write per-step CSV and concentration difference matrix.
  write_csv("output/cg_compare.csv", sor.records, cg.records);
  Eigen::MatrixXd conc_diff =
      (sor.final_conc - cg.final_conc).cwiseAbs();
  write_conc_diff("output/cg_conc_diff.txt", conc_diff);

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "\n═══ Results ═══\n";
  std::cout << std::setw(22) << "" << std::setw(14) << "SOR" << std::setw(14)
            << "CG" << "\n";
  std::cout << std::setw(22) << "Wall time (ms):" << std::setw(14) << sor_wall
            << std::setw(14) << cg_wall << "\n";
  std::cout << std::setw(22) << "Solve time (ms):" << std::setw(14)
            << sor.total_solve_ms << std::setw(14) << cg.total_solve_ms
            << "\n";
  std::cout << std::setw(22) << "Total solver iters:" << std::setw(14)
            << sor.total_solver_iters << std::setw(14) << cg.total_solver_iters
            << "\n";
  std::cout << std::setw(22) << "Growth steps:" << std::setw(14)
            << sor.growth_steps_done << std::setw(14) << cg.growth_steps_done
            << "\n";

  std::cout << std::scientific << std::setprecision(4);
  std::cout << "\nMax |c_SOR - c_CG|:  " << max_conc_diff << "\n";
  std::cout << "Cluster cell diffs:  " << mask_diff << "\n";

  std::cout << "\nCSV written to output/cg_compare.csv\n";
  return 0;
}

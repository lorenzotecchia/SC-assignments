#include "laplace.hpp"
#include <fstream>
#include <iostream>

// Helper: write a MatrixXd to a text file (space-separated, one row per line).
static void write_matrix(const std::string &path, const Eigen::MatrixXd &m) {
  std::ofstream out(path);
  out << m << "\n";
}

// Helper: write a vector of doubles to a text file (one value per line).
static void write_deltas(const std::string &path,
                         const std::vector<double> &deltas) {
  std::ofstream out(path);
  for (double d : deltas) out << d << "\n";
}

int main() {
  const int    N         = 50;
  const double tolerance = 1e-5;
  const int    max_iter  = 100000;
  const double omega     = 1.95; // good default for N=50

  // Boundary conditions: bottom row c=0, top row c=1, periodic in x.
  auto apply_boundary = [](Eigen::MatrixXd &grid) {
    int n = grid.rows();
    grid.row(0).setZero();
    grid.row(n - 1).setConstant(1.0);
  };

  // ── 1. Baseline Jacobi (existing, unchanged) ────────────────────────────
  auto jres = jacobi_solve(N, tolerance, max_iter, apply_boundary);
  std::cout << "laplace jacobi: " << jres.iterations << " iters, residual=" << jres.final_residual << "\n";
  write_matrix("output/laplace_solution.txt", jres.solution);
  write_deltas("output/laplace_deltas.txt", jres.deltas);

  // ── 2. Baseline SOR (no objects) ────────────────────────────────────────
  Eigen::MatrixXi mask_empty = Eigen::MatrixXi::Zero(N, N);
  auto sres = sor_solve(N, omega, tolerance, max_iter, mask_empty, apply_boundary);
  std::cout << "laplace sor (baseline): " << sres.iterations << " iters, residual=" << sres.final_residual << "\n";
  write_matrix("output/laplace_sor_baseline.txt", sres.solution);
  write_deltas("output/laplace_sor_baseline_deltas.txt", sres.deltas);

  // ── 3. K: Sinks ─────────────────────────────────────────────────────────
  // Two rectangular sinks: a large one and a smaller offset one.
  Eigen::MatrixXi mask_sink = Eigen::MatrixXi::Zero(N, N);
  fill_mask(mask_sink,
            {{10, 10, 20, 20},   // Rect{x0=10, y0=10, x1=20, y1=20}
             {30, 25, 45, 35}},  // second rectangle
            CellType::SINK);

  auto kres = sor_solve(N, omega, tolerance, max_iter, mask_sink, apply_boundary);
  std::cout << "laplace sor (sinks): " << kres.iterations << " iters, residual=" << kres.final_residual << "\n";
  write_matrix("output/laplace_sink.txt", kres.solution);
  write_deltas("output/laplace_sink_deltas.txt", kres.deltas);

  // Write mask so Python can draw object outlines.
  {
    std::ofstream out("output/laplace_sink_mask.txt");
    out << mask_sink << "\n";
  }

  // Omega sweep for K: measure iteration count vs omega with sinks present.
  {
    std::ofstream out("output/laplace_omega_sweep.txt");
    out << "omega iterations_no_obj iterations_sink\n";
    for (double w = 1.0; w < 2.0; w += 0.05) {
      auto r_empty = sor_solve(N, w, tolerance, max_iter, mask_empty, apply_boundary);
      auto r_sink  = sor_solve(N, w, tolerance, max_iter, mask_sink,  apply_boundary);
      out << w << " " << r_empty.iterations << " " << r_sink.iterations << "\n";
    }
    std::cout << "laplace omega sweep: done\n";
  }

  // ── 4. L: Insulator ─────────────────────────────────────────────────────
  Eigen::MatrixXi mask_ins = Eigen::MatrixXi::Zero(N, N);
  fill_mask(mask_ins,
            {{20, 15, 30, 35}},  // one rectangle as insulator
            CellType::INSULATOR);

  auto lres = sor_solve(N, omega, tolerance, max_iter, mask_ins, apply_boundary);
  std::cout << "laplace sor (insulator): " << lres.iterations << " iters, residual=" << lres.final_residual << "\n";
  write_matrix("output/laplace_insulator.txt", lres.solution);
  write_deltas("output/laplace_insulator_deltas.txt", lres.deltas);
  {
    std::ofstream out("output/laplace_insulator_mask.txt");
    out << mask_ins << "\n";
  }

  return 0;
}

#pragma once
#include <Eigen/Dense>
#include <functional>
#include <vector>

// ── Existing struct (unchanged) ─────────────────────────────────────────────
struct JacobiResult {
  Eigen::MatrixXd solution;
  int iterations;
  double final_residual;
  std::vector<double> deltas;
};

JacobiResult
jacobi_solve(int N, double tolerance, int max_iter,
             const std::function<void(Eigen::MatrixXd &)> &apply_boundary);

// ── New additions for K & L ──────────────────────────────────────────────────

// Cell type: NORMAL is solved by SOR; SINK is forced to 0; INSULATOR blocks
// flux.
enum class CellType { NORMAL = 0, SINK = 1, INSULATOR = 2 };

// Axis-aligned rectangle in grid-index coordinates (inclusive on all sides).
struct Rect {
  int x0, y0, x1, y1;
};

// Fill all cells inside each rectangle with the given CellType.
void fill_mask(Eigen::MatrixXi &mask, std::vector<Rect> rects, CellType type);

// Result from the SOR solver (mirrors JacobiResult).
struct SorResult {
  Eigen::MatrixXd solution;
  int iterations;
  double final_residual;
  std::vector<double> deltas;
};

// SOR solver with object mask support.
// omega: relaxation parameter (1 < omega < 2 for over-relaxation).
// mask:  N×N integer matrix; 0=NORMAL, 1=SINK, 2=INSULATOR.
SorResult
sor_solve(int N, double omega, double tolerance, int max_iter,
          const Eigen::MatrixXi &mask,
          const std::function<void(Eigen::MatrixXd &)> &apply_boundary);

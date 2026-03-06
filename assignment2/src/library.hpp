#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

// Cell types for DLA: EMPTY = diffusing, OCCUPIED = part of the aggregate.
enum class CellType : int { EMPTY = 0, OCCUPIED = 1 };

// Result from the SOR solver (mirrors JacobiResult).
struct SorResult {
  Eigen::MatrixXd solution;
  int iterations;
  double final_residual;
  std::vector<double> deltas;
};

// Per-step timing breakdown (microseconds).
struct StepProfile {
  int step;
  double candidates_us; // find_candidates
  double pg_us;         // compute_pg
  double select_us;     // stochastic selection
  double solve_us;      // SOR solve
  double snapshot_us;   // I/O (snapshot writing)
  int sor_iters;        // SOR iterations this step
};

// SOR solver for DLA.
// omega: relaxation parameter (1 < omega < 2 for over-relaxation).
// mask:  N×N integer matrix; 0=EMPTY, 1=OCCUPIED.
SorResult
sor_solve(int N, double omega, double tolerance, int max_iter,
          const Eigen::MatrixXi &mask,
          const std::function<void(Eigen::MatrixXd &)> &apply_boundary);

// Find growth candidates: empty cells adjacent to occupied cells.
std::vector<std::pair<int, int>>
find_candidates(int N, const Eigen::MatrixXi &mask);

// Compute p_g for each candidate: p_g(i,j) = c_{i,j}^η / Σ c_{k,l}^η
std::vector<double> compute_pg(const std::vector<std::pair<int, int>> &candidates,
                               const Eigen::MatrixXd &concentration,
                               double eta);

// Red-Black Gauss-Seidel solver (ω=1, checkerboard ordering, OpenMP parallel).
SorResult
rb_gauss_seidel_solve(int N, double tolerance, int max_iter,
                      const Eigen::MatrixXi &mask,
                      const std::function<void(Eigen::MatrixXd &)> &apply_boundary);

// Conjugate Gradient solver (Eigen SparseCG, builds Laplacian each call).
SorResult
cg_solve(int N, double tolerance, int max_iter,
         const Eigen::MatrixXi &mask,
         const std::function<void(Eigen::MatrixXd &)> &apply_boundary);

#include "library.hpp"
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

SorResult
cg_solve(int N, double tolerance, int max_iter,
         const Eigen::MatrixXi &mask,
         const std::function<void(Eigen::MatrixXd &)> &apply_boundary) {

  // Map free interior cells to linear-system indices.
  // Interior rows: 1..N-2.  Columns: 0..N-2 (col N-1 mirrors col 0).
  Eigen::MatrixXi idx = Eigen::MatrixXi::Constant(N, N, -1);
  int n_unknowns = 0;
  for (int i = 1; i < N - 1; ++i)
    for (int j = 0; j < N - 1; ++j)
      if (mask(i, j) != static_cast<int>(CellType::OCCUPIED))
        idx(i, j) = n_unknowns++;

  if (n_unknowns == 0) {
    Eigen::MatrixXd grid = Eigen::MatrixXd::Zero(N, N);
    apply_boundary(grid);
    return SorResult{grid, 0, 0.0, {}};
  }

  // Boundary values (top row = 1, bottom row = 0).
  Eigen::MatrixXd bc = Eigen::MatrixXd::Zero(N, N);
  apply_boundary(bc);

  // Build the SPD system  4·u(i,j) - Σ neighbors = rhs
  // (negated Laplacian so the matrix is positive-definite for CG).
  using Triplet = Eigen::Triplet<double>;
  std::vector<Triplet> triplets;
  triplets.reserve(5 * n_unknowns);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_unknowns);

  for (int i = 1; i < N - 1; ++i) {
    for (int j = 0; j < N - 1; ++j) {
      if (idx(i, j) < 0)
        continue;
      int row = idx(i, j);
      triplets.emplace_back(row, row, 4.0);

      int jm = (j - 1 + (N - 1)) % (N - 1);
      int jp = (j + 1) % (N - 1);
      int nbrs[][2] = {{i - 1, j}, {i + 1, j}, {i, jm}, {i, jp}};

      for (auto &nb : nbrs) {
        int ni = nb[0], nj = nb[1];
        if (ni == 0 || ni == N - 1) {
          // Dirichlet boundary — move known value to RHS.
          rhs(row) += bc(ni, nj);
        } else if (mask(ni, nj) == static_cast<int>(CellType::OCCUPIED)) {
          // Occupied cell has c = 0; no RHS contribution.
        } else {
          int col = idx(ni, nj);
          triplets.emplace_back(row, col, -1.0);
        }
      }
    }
  }

  Eigen::SparseMatrix<double> A(n_unknowns, n_unknowns);
  A.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                           Eigen::Lower | Eigen::Upper>
      cg;
  cg.setMaxIterations(max_iter);
  cg.setTolerance(tolerance);
  cg.compute(A);

  Eigen::VectorXd x = cg.solve(rhs);

  // Unpack solution back onto the 2-D grid.
  Eigen::MatrixXd grid = Eigen::MatrixXd::Zero(N, N);
  apply_boundary(grid);
  for (int i = 1; i < N - 1; ++i)
    for (int j = 0; j < N - 1; ++j)
      if (idx(i, j) >= 0)
        grid(i, j) = x(idx(i, j));

  // Periodicity: last column mirrors first.
  for (int i = 1; i < N - 1; ++i)
    grid(i, N - 1) = grid(i, 0);

  return SorResult{grid, static_cast<int>(cg.iterations()), cg.error(), {}};
}

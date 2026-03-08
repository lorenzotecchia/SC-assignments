#ifndef GRAY_SCOTT_HPP
#define GRAY_SCOTT_HPP

#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

//
// Performs one diffusion step of the simulation: updates interior cells
// (excluding ghost and fixed boundaries) into new state.
//
// @param u_old             State of u at previous step (including ghost
//                          columns).
// @param u_new             State in which the new state of u will be stored.
// @param v_old             State of v at previous step (including ghost
//                          columns).
// @param v_new             State in which the new state of v will be stored.
// @param u_coefficient     Coefficient of u in the finite difference scheme.
// @param v_coefficient     Coefficient of v in the finite difference scheme.
// @param N                 Side length of the square grid (including ghost
//                          cells).
//
void diffusion_step(const std::vector<double> &u_old,
                    std::vector<double> &u_new,
                    const std::vector<double> &v_old,
                    std::vector<double> &v_new, double u_coefficient,
                    double v_coefficient, int N);

//
// Performs one reaction step of the simulation: updates interior cells
// (excluding ghost and fixed boundaries) into new state.
//
// @param u_old             Current state of u (including ghost
//                          columns).
// @param v_old             Current state of v (including ghost
//                          columns).
// @param f_rate            Feed rate at wich U is added into the
//                          system.
// @param k_rate            Added to f_rate forms the decay rate of V.
// @param dt                Time interval between time steps.
// @param N                 Side length of the square grid (including ghost
//                          cells).
//
void reaction_step(std::vector<double> &u, std::vector<double> &v,
                   double f_rate, double k_rate, double dt, int N);

//
// Simulation of a 2D reaction-diffusion Gray-Scott process on the domain
// [0,1]x[0,1]. The chemicals reacting are U and V. Their concentration is
// denoted by u and v, respectively.
// The concentration varies according to:

// Boundary conditions:
//   periodic in y:
//      c(x, 0) = c(x, 1)
//   Periodic in x
//      c(0, y) = c(1, y)
//
// where c(x,y) is the concentration field of u or v.
//
// The domain is discretized on a uniform grid.
// At each time step, the system is described by two matrices whose element
// u_ij and v_ij approximates u(i*dx, j*dy) and v(i*dx, j*dy), where dx and
// dy are the spatial grid spacings.
//
// The simulation runs at full temporal resolution (t_num steps) for
// accuracy, but only every save_every-th frame is stored for output.
//
// @param u_history                 Vector of saved system states of u.
// @param v_history                 Vector of saved system states of v.
// @param u_init                    Initial U concentration field (including
//                                  ghost boundaries).
// @param v_init                    Initial V concentration field (including
//                                  ghost boundaries).
// @param t_num                     Number of time steps (including initial
//                                  condition).
// @param save_every                Interval between saved frames.
// @param u_coefficient             Diffusion coefficient factor of U that
//                                  appears in the finite-difference
//                                  discretization (D*dt/dx^2).
// @param v_coefficient             Diffusion coefficient factor of V that
//                                  appears in the finite-difference
//                                  discretization (D*dt/dx^2).
// @param f_rate                    Feed rate at wich U is added into the
//                                  system.
// @param k_rate                    Added to f_rate forms the decay rate of
// V.
// @param dt                        Time interval between two steps.
// @param update_ghost_boundaries   Function that update ghost boundaries,
// to
//                                  enforce periodic or Neumann boudnary
//                                  condition.
//

// Write a single saved frame (interior cells only) to an already-open stream,
// followed by a blank line separator.
void save_frame_txt(std::ofstream &output, const std::vector<double> &data,
                    int N);

template <typename Function>
void simulate_gray_scott(std::ofstream &u_out, std::ofstream &v_out,
                         std::vector<double> &u_init,
                         std::vector<double> &v_init, int N, int t_num,
                         int save_every, double u_coefficient,
                         double v_coefficient, double f_rate, double k_rate,
                         double dt, Function update_ghost_boundaries) {

  update_ghost_boundaries(u_init, N);
  update_ghost_boundaries(v_init, N);

  // save initial condition
  save_frame_txt(u_out, u_init, N);
  save_frame_txt(v_out, v_init, N);
  std::vector<double> u_old = u_init;
  std::vector<double> v_old = v_init;
  std::vector<double> u_new(N * N);
  std::vector<double> v_new(N * N);

  for (int t = 1; t < t_num; t++) {
    // reaction step computed for half time step, in-place on u_old, v_old
    reaction_step(u_old, v_old, f_rate, k_rate, dt / 2, N);
    update_ghost_boundaries(u_old, N);
    update_ghost_boundaries(v_old, N);

    // diffusion step computed for the whole time step, reads u_old and writes
    // u_new
    diffusion_step(u_old, u_new, v_old, v_new, u_coefficient, v_coefficient, N);
    update_ghost_boundaries(u_new, N);
    update_ghost_boundaries(v_new, N);
    std::swap(u_old, u_new);
    std::swap(v_old, v_new);

    // reaction step computed for half time step, in-place on u_old, v_old
    reaction_step(u_old, v_old, f_rate, k_rate, dt / 2, N);

    // save data
    if (t % save_every == 0) {
      save_frame_txt(u_out, u_old, N);
      save_frame_txt(v_out, v_old, N);
    }
  }
}

//
// Performs the update rule for a single interior cell in the u or v
// concentration field, relative to the reaction step.
//
// @param matrix            Matrix of previous time step concentration field.
// @param i                 Row index of the cell.
// @param j                 Column index of the cell.
// @param coefficient       Coefficient of the chemical in the finite difference
//                          scheme (D*dt/dx^2).
// @return                  New value of the concentration at the cell (i, j).
//
double diffusion_update(const std::vector<double> &grid, int i, int j,
                        double coefficient, int N);

//
// Performs the update rule for a single interior cell in the u and v
// concentration field, relative to the reaction step.
//
// @param u_ij_old          State of u at previous step (including ghost
//                          columns).
// @param u_ij_new          State in which the new state of u will be
// stored.
// @param v_ij_old          State of v at previous step (including ghost
//                          columns).
// @param v_ij_new          State in which the new state of v will be
// stored.
// @param f_rate            Feed rate at wich U is added into the
//                          system.
// @param k_rate            Added to f_rate forms the decay rate of V.
// @param dt                Time interval between time steps.
// @param max_iterations    Maximum number of iterations for the Newton
// implicit
//                          method.
// @param tolerance         Maximum residual tolerated.
//
void reaction_update(double &u_ij, double &v_ij, double f_rate, double k_rate,
                     double dt, const int &max_iterations,
                     const double &tolerance);

// Compute residual for the iterative Newton implicit method.
void compute_residual(double &residual_u, double &residual_v,
                      const double &u_ij_old, const double &u_ij_new,
                      const double &v_ij_old, const double &v_ij_new,
                      const double f_rate, const double k_rate, double dt);

// compute jacobian for the iterative Newton implicit method.
void compute_jacobian(double u_ij_new, double v_ij_new, double dt,
                      double f_rate, double k_rate, double &J11, double &J12,
                      double &J21, double &J22);

// Update ghost columns and rows to enforce periodic boundary conditions in x
// and y. The matrix is expected to have ghost columns at indices 0 and
// n_columns-1, which are copies of columns at indeces n_columns - 2 and 1,
// respectively. Also, the matrix is expected to have ghost rows at indices 0
// and n_rows-1, which are copies of rows at indeces n_rows - 2 and 1,
// respectively.
//
// @param matrix            Matrix in which ghost boundaries will be updated.
//
void update_periodic_boundaries(std::vector<double> &grid, int N);

// Update ghost columns to enforce zero-flux Neumann boundary conditions in x
// and y. The matrix is expected to have ghost columns at indices 0 and
// n_columns-1, which are copies of columns at indeces 1 and n_columns-2,
// respectively. Also, the matrix is expected to have ghost rows at indices 0
// and n_rows-1, which are copies of rows at indeces 1 and n_rows-2,
// respectively.
//
// @param matrix            Matrix in which ghost boundaries will be updated.
//

void update_insulated_boundaries(std::vector<double> &grid, int N);

// Save simulation data to a text file. Ghost columns (first and last
// columns) are omitted when writing.
//
// @param data_collector    Vector of system states at each saved step.
// @param output_filename   Path to the output file.
//
void initialize(std::vector<double> &u_init, std::vector<double> &v_init,
                double width, double u_value, double v_value,
                double perturbation, std::mt19937 &rng, int N);

void save_data_txt(const std::vector<std::vector<double>> &data_collector,
                   const std::string &output_filename, int N);

// Save simulation metadata to text files.
//
// @param x_num                 Number of spatial grid points (real columns).
// @param t_save                Number of saved time steps.
// @param x_delta               Spatial step size.
// @param t_delta_save          Time interval between saved frames.
// @param diffusion_constant    Diffusion coefficient.
// @param output_filename_xt    Path to the file for grid/time metadata.
// @param output_filename_params Path to the file for simulation parameters.
//
void save_metadata_txt(int x_num, int t_save, double x_delta,
                       double t_delta_save, double u_diff_constant,
                       double v_diff_constant, double f_rate, double k_rate,
                       double perturbation,
                       const std::string &output_filename_xt,
                       const std::string &output_filename_params);

#endif // GRAY_SCOTT_HPP
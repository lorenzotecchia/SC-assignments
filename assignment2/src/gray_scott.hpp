#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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
// At each time step, the system is described by two matrices whose element u_ij
// and v_ij approximates u(i*dx, j*dy) and v(i*dx, j*dy), where dx and dy are
// the spatial grid spacings.
//
// The simulation runs at full temporal resolution (t_num steps) for
// accuracy, but only every save_every-th frame is stored for output.
//
// @param u_history         Vector of saved system states of u.
// @param v_history         Vector of saved system states of v.
// @param u_init            Initial U concentration field (including ghost
//                          boundaries).
// @param v_init            Initial V concentration field (including ghost
//                          boundaries).
// @param t_num             Number of time steps (including initial condition).
// @param save_every        Interval between saved frames.
// @param u_coefficient     Diffusion coefficient factor of U that appears in
// the
//                          finite-difference discretization (D*dt/dx^2).
// @param v_coefficient     Diffusion coefficient factor of V that appears in
// the
//                          finite-difference discretization (D*dt/dx^2).
// @param f_rate            Feed rate at wich U is added into the system.
// @param k_rate            Added to f_rate forms the decay rate of V.
//
void simulate_diffusion(
    std::vector<std::vector<std::vector<double>>> &u_history,
    std::vector<std::vector<std::vector<double>>> &v_history,
    const std::vector<std::vector<double>> &u_init,
    const std::vector<std::vector<double>> &v_init, int t_num, int save_every,
    double u_coefficient, double v_coefficient, double f_rate, double k_rate);

//
// Performs one step of the simulation: updates interior cells (excluding
// ghost and fixed boundaries) into new state, then updates ghost boundaries.
//
// @param u_old             State of u at previous step (including ghost
//                          columns).
// @param u_new             State in which the new state of u will be stored.
// @param v_old             State of v at previous step (including ghost
//                          columns).
// @param v_new             State in which the new state of v will be stored.
// @param u_coefficient     Coefficient of u in the finite difference scheme.
// @param v_coefficient     Coefficient of v in the finite difference scheme.
// @param f_rate            Feed rate at wich U is added into the system.
// @param k_rate            Added to f_rate forms the decay rate of V.
// @param n_rows            Number of rows of the 2D grid (including boundary
//                          rows).
// @param n_columns         Number of columns of the 2D grid (including ghost
//                          columns).
//
void step_diffusion(const std::vector<std::vector<double>> &u_old,
                    std::vector<std::vector<double>> &u_new,
                    const std::vector<std::vector<double>> &v_old,
                    std::vector<std::vector<double>> &v_new,
                    double u_coefficient, double v_coefficient, double f_rate,
                    double k_rate, int n_rows, int n_columns);

//
// Performs the update rule for a single interior cell in the u concentration
// field.
//
// @param u                 Matrix containing the grid (including ghost
//                          columns) of concentration of u.
// @param v                 Matrix containing the grid (including ghost
//                          columns) of concentration of v.
// @param i                 Row index of the cell.
// @param j                 Column index of the cell.
// @param u_coefficient     Coefficient of u in the finite difference scheme
//                          (D*dt/dx^2).
// @param f_rate            Feed rate at wich U is added into the system.
// @return                  New value of u at the cell (i, j).
//
double u_update(const std::vector<std::vector<double>> &u,
                const std::vector<std::vector<double>> &v, int i, int j,
                double u_coefficient, double f_rate);

//
// Performs the update rule for a single interior cell in the v concentration
// field.
//
// @param u                 Matrix containing the grid (including ghost
//                          columns) of concentration of u.
// @param v                 Matrix containing the grid (including ghost
//                          columns) of concentration of v.
// @param i                 Row index of the cell.
// @param j                 Column index of the cell.
// @param v_coefficient     Coefficient of v in the finite difference scheme
//                          (D*dt/dx^2).
// @param f_rate            Feed rate at wich U is added into the system.
// @param k_rate            Added to f_rate forms the decay rate of V.
// @return                  New value of v at the cell (i, j).
//
double v_update(const std::vector<std::vector<double>> &u,
                const std::vector<std::vector<double>> &v, int i, int j,
                double v_coefficient, double f_rate, double k_rate);

// Update ghost columns to enforce periodic boundary conditions in x and y.
// The matrix is expected to have ghost columns at indices 0 and n_columns-1,
// which are copies of columns at indeces n_columns - 2 and 1, respectively.
// also, the matrix is expected to have ghost rows at indices 0 and n_rows-1,
// which are copies of rows at indeces n_rows - 2 and 1, respectively.
// @param matrix            Matrix in which ghost boundaries will be updated.
//
void update_ghost_boundaries(std::vector<std::vector<double>> &matrix);

// Save simulation data to a text file. Ghost columns (first and last columns)
// are omitted when writing.
//
// @param data_collector    Vector of system states at each saved step.
// @param output_filename   Path to the output file.
//

void initialize(std::vector<std::vector<double>> &u_init,
                std::vector<std::vector<double>> &v_init, double width,
                double u_value, double v_value, double perturbation,
                std::mt19937 &rng);

void save_data_txt(
    const std::vector<std::vector<std::vector<double>>> &data_collector,
    const std::string &output_filename);

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
                       const std::string &output_filename_xt,
                       const std::string &output_filename_params);
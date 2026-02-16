#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

//
// Simulation of a 2D diffusion process on the domain [0,1]x[0,1].
// Boundary conditions:
//   Dirichlet in y:
//      c(x, y=1)=1
//      c(x, y=0)=0
//   Periodic in x
//      c(0, y) = c(1, y)
//
// where c(x,y) is the concentration field.
//
// The domain is discretized on a uniform grid.
// At each time step, the system is described by a matrix whose element c_ij
// approximates c(i*dx, j*dy), where dx and dy are the spatial grid spacings.
//
// The simulation runs at full temporal resolution (t_num steps) for
// accuracy, but only every save_every-th frame is stored for output.
//
// @param initial_state     Initial concentration field (including ghost
// boundaries).
// @param t_num             Number of time steps (including initial condition).
// @param save_every        Interval between saved frames.
// @param coefficient       Diffusion coefficient factor that appears in the
//                          finite-difference discretization (D*dt/dx^2).
//
// @return                  Vector of saved system states.
//
std::vector<std::vector<std::vector<double>>>
simulate_diffusion(const std::vector<std::vector<double>> &initial_state,
                   int t_num, int save_every, double coefficient);

//
// Performs one step of the simulation: updates interior cells (excluding
// ghost and fixed boundaries) into new_state, then updates ghost boundaries and
// check if fixed Dirichlet top/bottom boundaries have not been changed
// (assert).
//
// @param old_state         State of the previous step (including ghost
//                          columns).
// @param new_state         State in which the new state will be stored.
// @param coefficient       Coefficient of the finite difference scheme.
// @param n_rows            Number of rows of the 2D grid (including boundary
//                          rows).
// @param n_columns         Number of columns of the 2D grid (including ghost
//                          columns).
//
void step_diffusion(const std::vector<std::vector<double>> &old_state,
                    std::vector<std::vector<double>> &new_state,
                    double coefficient, int n_rows, int n_columns);

//
// Performs the update rule for a single interior cell.
//
// @param matrix            Matrix containing the grid (including ghost
//                          columns).
// @param i                 Row index of the cell.
// @param j                 Column index of the cell.
// @param coefficient       Coefficient of the finite difference scheme
//                          (D*dt/dx^2).
// @return                  New value of the cell (i, j).
//
double cell_update(const std::vector<std::vector<double>> &matrix, int i, int j,
                   double coefficient);

// Update ghost columns to enforce periodic boundary conditions in x.
// The matrix is expected to have ghost columns at indices 0 and n_columns-1,
// which are copies of columns at indeces n_columns - 2 and 1, respectively.
//
// @param matrix            Matrix in which ghost boundaries will be updated.
//
void update_ghost_boundaries(std::vector<std::vector<double>> &matrix);

//
// Force upper and lower boundaries to have concentration equal to one and zero,
// respectivley
//
// @param matrix            matrix in which fixed boundaries are forced
//
void create_fixed_boundaries(std::vector<std::vector<double>> &matrix);

// Verify top and bottom rows are fixed to 1.0 and 0.0 respectively (assert).
//
// @param matrix            Matrix in which fixed boundaries are checked.
//
void check_fixed_boundaries(const std::vector<std::vector<double>> &matrix);

// Save simulation data to a text file. Ghost columns (first and last columns)
// are omitted when writing.
//
// @param data_collector    Vector of system states at each saved step.
// @param output_filename   Path to the output file.
//
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
                       double t_delta_save, double diffusion_constant,
                       const std::string &output_filename_xt,
                       const std::string &output_filename_params);
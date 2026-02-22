#include <functional>
#include <span>
#include <string>
#include <vector>

void run_case(const std::string &prefix, int x_num, double x1, double x2,
              int t_num, double t1, double t2, double c,
              const std::vector<double> &x_vec,
              const std::vector<double> &u1_init, int save_every);

//
// Advances the 1D wave equation solution by one time step using the
// standard leapfrog (central difference) scheme.
//
// Given the solution at two consecutive time levels U(t-dt) and U(t),
// computes U(t+dt) via:
//
//   U(t+dt, x) = alpha^2     * U(t, x+dx)
//              + 2(1-alpha^2) * U(t, x)
//              + alpha^2      * U(t, x-dx)
//              -                U(t-dt, x)
//
// where alpha = c * dt / dx is the CFL number. This is a second-order
// accurate explicit scheme in both space and time.
//
// Boundary values at x=x1 and x=x2 are prescribed directly by the
// Dirichlet boundary condition functions u_x1(t) and u_x2(t).
//
// @param t      The new time level: current time + dt.
// @param alpha  CFL stability coefficient from fd1d_wave_alpha.
// @param u_x1   Left boundary condition as a function of time.
// @param u_x2   Right boundary condition as a function of time.
// @param u1     Solution at the previous time level U(t-dt).
// @param u2     Solution at the current time level U(t).
// @return       The solution U at the new time level U(t+dt).
//

std::vector<double> fd1d_wave_step(double t, double alpha,
                                   const std::function<double(double)> &u_x1,
                                   const std::function<double(double)> &u_x2,
                                   std::span<const double> u1,
                                   std::span<const double> u2);

//
// Performs the first time step of the 1D wave equation using a special
// bootstrapping formula.
//
// The standard leapfrog scheme (see fd1d_wave_step) requires the solution
// at two previous time levels: U(t) and U(t-dt). At the very first step
// we only have U(t=0) from the initial condition. To obtain the missing
// U(t-dt), we use the initial velocity dU/dt(t=0,x) via a central
// difference approximation:
//
//   U(t-dt, x) ~ U(t+dt, x) - 2*dt * dU/dt(t, x)
//
// Substituting this into the leapfrog stencil and solving for U(t+dt, x)
// yields the first-step formula:
//
//   U(t+dt, x) = 0.5 * alpha^2 * U(t, x+dx)
//              + (1 - alpha^2)  * U(t, x)
//              + 0.5 * alpha^2 * U(t, x-dx)
//              + dt * dU/dt(t, x)
//
// Boundary values are set directly from the Dirichlet conditions u_x1(t)
// and u_x2(t).
//
// @param x_vec    Spatial node coordinates (size determines the grid).
// @param t        Time after the first step: t = t1 + t_delta.
// @param t_delta  Time step size.
// @param alpha    CFL stability coefficient from fd1d_wave_alpha.
// @param u_x1     Left boundary condition as a function of time.
// @param u_x2     Right boundary condition as a function of time.
// @param ut_t1    Initial velocity dU/dt(t=0, x) as a function of x_vec.
// @param u1       Initial displacement U(t=0, x).
// @return         The solution U at the first time step.
//

std::vector<double> fd1d_wave_start(
    std::span<const double> x_vec, double t, double t_delta, double alpha,
    const std::function<double(double)> &u_x1,
    const std::function<double(double)> &u_x2,
    const std::function<std::vector<double>(std::span<const double>)> &ut_t1,
    std::span<const double> u1);

std::vector<double> run_case_leapfrog(const std::string &prefix, int x_num,
                                      double x1, double x2, int t_num,
                                      double t1, double t2, double c,
                                      const std::vector<double> &x_vec,
                                      const std::vector<double> &u1_init,
                                      const std::vector<double> &v1_init,
                                      int save_every);

void leapfrog_wave_step(double t, double t_delta, double alpha,
                        const std::function<double(double)> &u_x1,
                        const std::function<double(double)> &u_x2,
                        std::span<const double> u1, std::span<double> u2,
                        std::span<const double> v1, std::span<double> v2);

std::vector<double> leapfrog_wave_start(
    std::span<const double> x_vec, double t_delta, double alpha,
    const std::function<double(double)> &u_x1,
    const std::function<double(double)> &u_x2,
    const std::function<std::vector<double>(std::span<const double>)> &ut_t1,
    const std::span<const double> &u1_init,
    const std::span<const double> &v1_init);

//
// Computes the CFL (Courant-Friedrichs-Lewy) stability coefficient
// for the 1D wave equation explicit timestepping scheme.
//
// Given a spatial grid [x1, x2] with x_num equally spaced nodes and
// a time domain [t1, t2] with t_num equally spaced steps, computes:
//
//   alpha = c * dt / dx
//
// where dt = (t2 - t1) / (t_num - 1) and dx = (x2 - x1) / (x_num - 1).
//
// The scheme is stable only when |alpha| <= 1. If violated, the central
// coefficient (1 - alpha^2) becomes negative and the solution diverges.
// Prints a warning to stderr if the condition is not met.
//
// @param x_num  Number of spatial nodes (including endpoints).
// @param x1     Left boundary of the spatial domain.
// @param x2     Right boundary of the spatial domain.
// @param t_num  Number of time steps (including initial condition).
// @param t1     Start time.
// @param t2     End time.
// @param c      Wave propagation speed (c = sqrt(T/mu) for a string).
// @return       The stability coefficient alpha.
//
double fd1d_wave_alpha(int x_num, double x1, double x2, int t_num, double t1,
                       double t2, double c);

//
// Evaluates a piecewise linear interpolant at a set of query points.
//
// Given data points (xd[i], yd[i]) sorted by ascending xd, constructs
// a piecewise linear spline and evaluates it at each point in xv.
// Values outside the data range are clamped to the nearest endpoint
// (i.e. yd[0] for xv < xd[0], yd[nd-1] for xv > xd[nd-1]).
//
// Useful for defining initial conditions on a grid that may not
// coincide with the data points (e.g. a triangular pluck shape).
//
// @param xd  X-coordinates of the data points (sorted ascending).
// @param yd  Y-coordinates of the data points.
// @param xv  X-coordinates of the query (evaluation) points.
// @return    Interpolated Y-values at each point in xv.
//

std::vector<double> piecewise_linear(std::span<const double> xd,
                                     std::span<const double> yd,
                                     std::span<const double> xv);

//
// Writes a column-major matrix (m rows x n columns) to a text file.
//
// Each row of the output file corresponds to one column of the matrix.
// The data is stored in column-major order: element (i, j) is at
// table[i + j * m]. Values are written with 16 digits of precision
// in a field of width 24.
//
// Throws std::runtime_error if the file cannot be opened.
//
// @param output_filename  Path to the output file.
// @param m                Number of rows.
// @param n                Number of columns.
// @param table            Matrix data in column-major layout (size m*n).
//

void r8mat_write(const std::string &output_filename, int m, int n,
                 std::span<const double> table);

//
// Creates a vector of n linearly spaced values from a_first to a_last.
//
// Equivalent to numpy.linspace(a_first, a_last, n). Both endpoints are
// included. If n == 1, returns the midpoint (a_first + a_last) / 2.
//
// @param n        Number of values to generate.
// @param a_first  First value in the sequence.
// @param a_last   Last value in the sequence.
// @return         Vector of n evenly spaced values.
//
std::vector<double> r8vec_linspace_new(int n, double a_first, double a_last);

//
// Original algorithms by John Burkardt (24 January 2012).
// Distributed under the MIT license.
//
// Reference:
//   George Lindfield, John Penny,
//   Numerical Methods Using MATLAB,
//   Second Edition, Prentice Hall, 1999,
//   ISBN: 0-13-012641-1, LC: QA297.P45.
//
// Modernized to C++20 (std::vector, std::span, std::function).
#include <functional>
#include <span>
#include <string>
#include <vector>

double fd1d_wave_alpha(int x_num, double x1, double x2, int t_num, double t1,
                       double t2, double c);

std::vector<double> fd1d_wave_start(
    std::span<const double> x_vec, double t, double t_delta, double alpha,
    const std::function<double(double)> &u_x1,
    const std::function<double(double)> &u_x2,
    const std::function<std::vector<double>(std::span<const double>)> &ut_t1,
    std::span<const double> u1);

std::vector<double> fd1d_wave_step(double t, double alpha,
                                   const std::function<double(double)> &u_x1,
                                   const std::function<double(double)> &u_x2,
                                   std::span<const double> u1,
                                   std::span<const double> u2);

std::vector<double> piecewise_linear(std::span<const double> xd,
                                     std::span<const double> yd,
                                     std::span<const double> xv);

void r8mat_write(const std::string &output_filename, int m, int n,
                 std::span<const double> table);

std::vector<double> r8vec_linspace_new(int n, double a_first, double a_last);

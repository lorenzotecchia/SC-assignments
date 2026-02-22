#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>

#include "fd1d_wave.hpp"

void run_case(const std::string &prefix, int x_num, double x1, double x2,
              int t_num, double t1, double t2, double c,
              const std::vector<double> &x_vec,
              const std::vector<double> &u1_init, int save_every) {

  auto alpha = fd1d_wave_alpha(x_num, x1, x2, t_num, t1, t2, c);
  auto t_delta = (t2 - t1) / static_cast<double>(t_num - 1);

  auto u_x1 = [](double) { return 0.0; };
  auto u_x2 = [](double) { return 0.0; };
  auto ut_t1 = [](std::span<const double> x) {
    return std::vector<double>(x.size(), 0.0);
  };

  int t_save = (t_num + save_every - 1) / save_every;
  std::vector<double> u_mat(x_num * t_save);

  // Store initial condition (frame 0)
  for (int j = 0; j < x_num; j++) {
    u_mat[j + 0 * x_num] = u1_init[j];
  }

  // First time step
  auto t = t1 + t_delta;
  auto u2 =
      fd1d_wave_start(x_vec, t, t_delta, alpha, u_x1, u_x2, ut_t1, u1_init);

  if (1 % save_every == 0) {
    int col = 1 / save_every;
    for (int j = 0; j < x_num; j++) {
      u_mat[j + col * x_num] = u2[j];
    }
  }

  // Remaining time steps
  auto u_old = u1_init;
  auto u_cur = u2;

  for (int i = 2; i < t_num; i++) {
    t = t1 + i * t_delta;
    auto u_new = fd1d_wave_step(t, alpha, u_x1, u_x2, u_old, u_cur);

    if (i % save_every == 0) {
      int col = i / save_every;
      for (int j = 0; j < x_num; j++) {
        u_mat[j + col * x_num] = u_new[j];
      }
    }

    u_old = u_cur;
    u_cur = u_new;
  }

  r8mat_write("output/" + prefix + "_wave_data.txt", x_num, t_save, u_mat);
}

std::vector<double> fd1d_wave_step(double t, double alpha,
                                   const std::function<double(double)> &u_x1,
                                   const std::function<double(double)> &u_x2,
                                   std::span<const double> u1,
                                   std::span<const double> u2) {
  auto x_num = static_cast<int>(u1.size());
  std::vector<double> u3(x_num);

  u3[0] = u_x1(t);

  for (int j = 1; j < x_num - 1; j++) {
    u3[j] = alpha * alpha * u2[j + 1] + 2.0 * (1.0 - alpha * alpha) * u2[j] +
            alpha * alpha * u2[j - 1] - u1[j];
  }

  u3[x_num - 1] = u_x2(t);

  return u3;
}

std::vector<double> fd1d_wave_start(
    std::span<const double> x_vec, double t, double t_delta, double alpha,
    const std::function<double(double)> &u_x1,
    const std::function<double(double)> &u_x2,
    const std::function<std::vector<double>(std::span<const double>)> &ut_t1,
    std::span<const double> u1) {
  auto x_num = static_cast<int>(x_vec.size());
  auto ut = ut_t1(x_vec);

  std::vector<double> u2(x_num);

  u2[0] = u_x1(t);

  for (int j = 1; j < x_num - 1; j++) {
    u2[j] = alpha * alpha * u1[j + 1] / 2.0 + (1.0 - alpha * alpha) * u1[j] +
            alpha * alpha * u1[j - 1] / 2.0 + t_delta * ut[j];
  }

  u2[x_num - 1] = u_x2(t);

  return u2;
}

std::vector<double> run_case_leapfrog(const std::string &prefix, int x_num,
                                      double x1, double x2, int t_num,
                                      double t1, double t2, double c,
                                      const std::vector<double> &x_vec,
                                      const std::vector<double> &u1_init,
                                      const std::vector<double> &v1_init,
                                      int save_every) {

  auto alpha = fd1d_wave_alpha(x_num, x1, x2, t_num, t1, t2, c);
  auto t_delta = (t2 - t1) / static_cast<double>(t_num - 1);

  auto u_x1 = [](double) { return 0.0; };
  auto u_x2 = [](double) { return 0.0; };
  auto ut_t1 = [](std::span<const double> x) {
    return std::vector<double>(x.size(), 0.0);
  };

  int t_save = (t_num + save_every - 1) / save_every;
  std::vector<double> u_mat(x_num * t_save);

  // Store initial condition (frame 0)
  for (int j = 0; j < x_num; j++) {
    u_mat[j + 0 * x_num] = u1_init[j];
  }

  // First time step
  auto t = t1 + t_delta;
  auto v2_half = leapfrog_wave_start(x_vec, t_delta, alpha, u_x1, u_x2, ut_t1,
                                     u1_init, v1_init);
  auto u2 =
      fd1d_wave_start(x_vec, t, t_delta, alpha, u_x1, u_x2, ut_t1, u1_init);

  if (1 % save_every == 0) {
    int col = 1 / save_every;
    for (int j = 0; j < x_num; j++) {
      u_mat[j + col * x_num] = u2[j];
    }
  }

  // Remaining time steps
  auto u_old = u1_init;
  auto u_cur = u2;
  auto v_old = v1_init;
  auto v_cur = v2_half;
  for (int i = 2; i < t_num; i++) {
    t = t1 + i * t_delta;
    leapfrog_wave_step(t, t_delta, alpha, u_x1, u_x2, u_old, u_cur, v_old,
                       v_cur);

    if (i % save_every == 0) {
      int col = i / save_every;
      for (int j = 0; j < x_num; j++) {
        u_mat[j + col * x_num] = u_cur[j];
      }
    }

    // swap old-current data
    u_old = u_cur;
    v_old = v_cur;
  }

  r8mat_write("output/" + prefix + "_wave_data.txt", x_num, t_save, u_mat);
  return u_mat;
}

void leapfrog_wave_step(double t, double t_delta, double alpha,
                        const std::function<double(double)> &u_x1,
                        const std::function<double(double)> &u_x2,
                        std::span<const double> u1, std::span<double> u2,
                        std::span<const double> v1, std::span<double> v2) {

  auto x_num = static_cast<int>(u1.size());

  // boundary conditions
  u2[0] = u_x1(t);
  u2[x_num - 1] = u_x2(t);

  // interior cells update
  for (int j = 1; j < x_num - 1; j++) {
    v2[j] =
        v1[j] + alpha * alpha * (u1[j + 1] - 2.0 * u1[j] + u1[j - 1]) * t_delta;
    u2[j] = u1[j] + v2[j] * t_delta;
  }
}

std::vector<double> leapfrog_wave_start(
    std::span<const double> x_vec, double t_delta, double alpha,
    const std::function<double(double)> &u_x1,
    const std::function<double(double)> &u_x2,
    const std::function<std::vector<double>(std::span<const double>)> &ut_t1,
    const std::span<const double> &u1_init,
    const std::span<const double> &v1_init) {
  auto x_num = static_cast<int>(x_vec.size());
  std::vector<double> v2_half(x_num);

  // copy boundary values
  v2_half[0] = v1_init[0];
  v2_half[x_num - 1] = v1_init[x_num - 1];

  for (int j = 1; j < x_num - 1; j++) {
    v2_half[j] = v1_init[j] +
                 t_delta * alpha * alpha *
                     (u1_init[j + 1] * 0.5 - u1_init[j] + u1_init[j - 1] * 0.5);
  }
  return v2_half;
}

double fd1d_wave_alpha(int x_num, double x1, double x2, int t_num, double t1,
                       double t2, double c) {

  auto t_delta = (t2 - t1) / static_cast<double>(t_num - 1);
  auto x_delta = (x2 - x1) / static_cast<double>(x_num - 1);
  auto alpha = c * t_delta / x_delta;

  std::cout << "\n";
  std::cout << "  Stability condition ALPHA = C * DT / DX = " << alpha << "\n";

  if (1.0 < fabs(alpha)) {
    std::cerr << "\n";
    std::cerr << "FD1D_WAVE_ALPHA - Warning!\n";
    std::cerr << "  The stability condition |ALPHA| <= 1 fails.\n";
    std::cerr << "  Computed results are liable to be inaccurate.\n";
  }

  return alpha;
}

std::vector<double> piecewise_linear(std::span<const double> xd,
                                     std::span<const double> yd,
                                     std::span<const double> xv) {
  auto nd = static_cast<int>(xd.size());
  auto nv = static_cast<int>(xv.size());
  std::vector<double> yv(nv);

  for (int iv = 0; iv < nv; iv++) {
    if (xv[iv] < xd[0]) {
      yv[iv] = yd[0];
    } else if (xd[nd - 1] < xv[iv]) {
      yv[iv] = yd[nd - 1];
    } else {
      for (int id = 1; id < nd; id++) {
        if (xv[iv] < xd[id]) {
          yv[iv] = ((xd[id] - xv[iv]) * yd[id - 1] +
                    (xv[iv] - xd[id - 1]) * yd[id]) /
                   (xd[id] - xd[id - 1]);
          break;
        }
      }
    }
  }

  return yv;
}

void r8mat_write(const std::string &output_filename, int m, int n,
                 std::span<const double> table) {
  std::ofstream output;

  output.open(output_filename);

  if (!output) {
    throw std::runtime_error("R8MAT_WRITE: Could not open file: " +
                             output_filename);
  }

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      output << "  " << std::setw(24) << std::setprecision(16)
             << table[i + j * m];
    }
    output << "\n";
  }

  output.close();
}

std::vector<double> r8vec_linspace_new(int n, double a_first, double a_last) {
  std::vector<double> a(n);

  if (n == 1) {
    a[0] = (a_first + a_last) / 2.0;
  } else {
    for (int i = 0; i < n; i++) {
      a[i] = (static_cast<double>(n - 1 - i) * a_first +
              static_cast<double>(i) * a_last) /
             static_cast<double>(n - 1);
    }
  }

  return a;
}

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

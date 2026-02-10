#include "fd1d_wave.hpp"
#include <cmath>
#include <numbers>
#include <string>

//
// Runs the 1D wave equation simulation for a given initial condition
// and writes subsampled results to output/<prefix>_wave_data.txt.
//
// The simulation runs at full temporal resolution (t_num steps) for
// accuracy, but only every save_every-th frame is stored for output.
//
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

int main() {
  int x_num = 201;
  double x1 = 0.0;
  double x2 = 1.0;
  int t_num = 4001;
  double t1 = 0.0;
  double t2 = 4.0;
  double c = 1.0;
  int save_every = 4;

  auto x_vec = r8vec_linspace_new(x_num, x1, x2);

  // Write shared x grid once
  r8mat_write("output/x_data.txt", x_num, 1, x_vec);

  // Case i: Psi(x, 0) = sin(2*pi*x)
  {
    std::vector<double> u1(x_num);
    for (int j = 0; j < x_num; j++) {
      u1[j] = std::sin(2.0 * std::numbers::pi * x_vec[j]);
    }
    run_case("case1", x_num, x1, x2, t_num, t1, t2, c, x_vec, u1,
             save_every);
  }

  // Case ii: Psi(x, 0) = sin(5*pi*x)
  {
    std::vector<double> u1(x_num);
    for (int j = 0; j < x_num; j++) {
      u1[j] = std::sin(5.0 * std::numbers::pi * x_vec[j]);
    }
    run_case("case2", x_num, x1, x2, t_num, t1, t2, c, x_vec, u1,
             save_every);
  }

  // Case iii: Psi(x, 0) = sin(5*pi*x) if 1/5 < x < 2/5, else 0
  {
    std::vector<double> u1(x_num, 0.0);
    for (int j = 0; j < x_num; j++) {
      if (x_vec[j] > 1.0 / 5.0 && x_vec[j] < 2.0 / 5.0) {
        u1[j] = std::sin(5.0 * std::numbers::pi * x_vec[j]);
      }
    }
    run_case("case3", x_num, x1, x2, t_num, t1, t2, c, x_vec, u1,
             save_every);
  }

  return 0;
}

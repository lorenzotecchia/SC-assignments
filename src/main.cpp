#include "fd1d_wave.hpp"
#include <cmath>
#include <numbers>
#include <string>

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
    run_case("case1", x_num, x1, x2, t_num, t1, t2, c, x_vec, u1, save_every);
  }

  // Case ii: Psi(x, 0) = sin(5*pi*x)
  {
    std::vector<double> u1(x_num);
    for (int j = 0; j < x_num; j++) {
      u1[j] = std::sin(5.0 * std::numbers::pi * x_vec[j]);
    }
    run_case("case2", x_num, x1, x2, t_num, t1, t2, c, x_vec, u1, save_every);
  }

  // Case iii: Psi(x, 0) = sin(5*pi*x) if 1/5 < x < 2/5, else 0
  {
    std::vector<double> u1(x_num, 0.0);
    for (int j = 0; j < x_num; j++) {
      if (x_vec[j] > 1.0 / 5.0 && x_vec[j] < 2.0 / 5.0) {
        u1[j] = std::sin(5.0 * std::numbers::pi * x_vec[j]);
      }
    }
    run_case("case3", x_num, x1, x2, t_num, t1, t2, c, x_vec, u1, save_every);
  }

  return 0;
}

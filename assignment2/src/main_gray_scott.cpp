#include "gray_scott.hpp"
#include <iostream>
#include <random>

int main() {
  int x_num = 256;
  int y_num = x_num;
  int save_every = 100;
  double u_diff_constant = 0.16;
  double v_diff_constant = 0.08;
  double f_rate = 0.035;
  double k_rate = 0.060;
  double x_delta = 1;
  double t_delta = 1;
  int t_num = 10000;

  int t_save = (t_num + save_every - 1) / save_every;
  double t_delta_save = save_every * t_delta;

  double u_coefficient = t_delta * u_diff_constant / (x_delta * x_delta);
  double v_coefficient = t_delta * v_diff_constant / (x_delta * x_delta);

  std::cout << "diffusion: coefficient of u, v = " << u_coefficient << ", "
            << v_coefficient << '\n';
  std::mt19937 rng(42);
  std::vector<std::vector<double>> u_init(x_num + 2,
                                          std::vector<double>(y_num + 2, 0.0));
  std::vector<std::vector<double>> v_init(x_num + 2,
                                          std::vector<double>(y_num + 2, 0.0));
  initialize(u_init, v_init, 0.1, 0.5, 0.25, 0.0, rng);

  // data collectors:
  std::vector<std::vector<std::vector<double>>> u_history(
      t_save,
      std::vector<std::vector<double>>(x_num, std::vector<double>(y_num)));

  std::vector<std::vector<std::vector<double>>> v_history(
      t_save,
      std::vector<std::vector<double>>(x_num, std::vector<double>(y_num)));

  simulate_diffusion(u_history, v_history, u_init, v_init, t_num, save_every,
                     u_coefficient, v_coefficient, f_rate, k_rate);

  save_metadata_txt(x_num, t_save, x_delta, t_delta_save, u_diff_constant,
                    v_diff_constant, f_rate, k_rate,
                    "output/gray_scott_xt_data.txt",
                    "output/gray_scott_params.txt");
  save_data_txt(u_history, "output/gray_scott_u_data.txt");
  save_data_txt(v_history, "output/gray_scott_v_data.txt");
}
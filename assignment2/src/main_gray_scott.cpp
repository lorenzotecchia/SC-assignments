#include "gray_scott.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

int main(int argc, char *argv[]) {
  int x_num = 512;
  int N = x_num + 2; // grid side length including ghost cells
  int t_num = 20000;
  int save_every = 200;
  double u_diff_constant = 0.16;
  double v_diff_constant = 0.08;
  double f_rate = 0.035; // feed rate
  double k_rate = 0.057; // kill rate
  double x_delta = 1;
  double t_delta = 1;
  double u_init_vaule = 0.5;
  double v_init_vaule = 0.25;
  double width = 0.1; // width of square in which is contained V at time zero.
  double perturbation =
      0.1; // half range of uniform pertubration of initial state.

  // optional CLI overrides: ./gray_scott_cpp [f_rate] [k_rate]
  if (argc >= 3) {
    f_rate = std::atof(argv[1]);
    k_rate = std::atof(argv[2]);
  }

  // build output filename suffix from parameters
  std::string suffix;
  if (argc >= 3) {
    suffix = std::string("_f") + argv[1] + "_k" + argv[2];
  } else {
    std::ostringstream oss;
    oss << "_f" << std::fixed << std::setprecision(3) << f_rate
        << "_k" << std::fixed << std::setprecision(3) << k_rate;
    suffix = oss.str();
  }

  int t_save = (t_num + save_every - 1) / save_every;
  double t_delta_save = save_every * t_delta;

  double u_coefficient = t_delta * u_diff_constant / (x_delta * x_delta);
  double v_coefficient = t_delta * v_diff_constant / (x_delta * x_delta);

  std::mt19937 rng(42);
  std::vector<double> u_init(N * N);
  std::vector<double> v_init(N * N);
  initialize(u_init, v_init, width, u_init_vaule, v_init_vaule, perturbation,
             rng, N);

  std::ofstream u_out("output/gray_scott_u_data" + suffix + ".txt");
  std::ofstream v_out("output/gray_scott_v_data" + suffix + ".txt");
  if (!u_out || !v_out) {
    throw std::runtime_error("Could not open output files");
  }

  simulate_gray_scott(u_out, v_out, u_init, v_init, N, t_num, save_every,
                      u_coefficient, v_coefficient, f_rate, k_rate, t_delta,
                      update_insulated_boundaries);

  u_out.close();
  v_out.close();

  save_metadata_txt(x_num, t_save, x_delta, t_delta_save, u_diff_constant,
                    v_diff_constant, f_rate, k_rate, perturbation,
                    "output/gray_scott_xt_data" + suffix + ".txt",
                    "output/gray_scott_params" + suffix + ".txt");
}
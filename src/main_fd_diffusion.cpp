#include "fd_diffusion.hpp"
#include <iostream>

int main() {
  int x_num = 401;
  int y_num = x_num;
  int t_num = 15001;
  int save_every = 100;
  double diffusion_constant = 1.0;
  double x_delta = 1.0 / (x_num - 1);
  double t_delta = x_delta * x_delta / diffusion_constant / 15.0;

  int t_save = (t_num + save_every - 1) / save_every;
  double t_delta_save = save_every * t_delta;

  double coefficient = t_delta * diffusion_constant / (x_delta * x_delta);

  std::cout << "diffusion: coefficient = " << coefficient << '\n';

  double zero = 0;
  std::vector<std::vector<double>> initial_state(
      y_num, std::vector<double>(x_num + 2, zero));
  create_fixed_boundaries(initial_state);
  check_fixed_boundaries(initial_state);

  auto data = simulate_diffusion(initial_state, t_num, save_every, coefficient);

  save_metadata_txt(x_num, t_save, x_delta, t_delta_save, diffusion_constant,
                    "output/fd_diffusion_xt_data.txt",
                    "output/fd_diffusion_params.txt");
  save_data_txt(data, "output/fd_diffusion_data.txt");
}
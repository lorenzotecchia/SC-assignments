#include "gray_scott.hpp"
#include <cassert>
#include <random>

void simulate_diffusion(
    std::vector<std::vector<std::vector<double>>> &u_history,
    std::vector<std::vector<std::vector<double>>> &v_history,
    const std::vector<std::vector<double>> &u_init,
    const std::vector<std::vector<double>> &v_init, int t_num, int save_every,
    double u_coefficient, double v_coefficient, double f_rate, double k_rate) {
  int n_columns = u_init[0].size();
  int n_rows = u_init.size();

  u_history[0] = u_init;
  v_history[0] = v_init;
  std::vector<std::vector<double>> u_old = u_init;
  std::vector<std::vector<double>> v_old = v_init;
  std::vector<std::vector<double>> u_new(n_rows,
                                         std::vector<double>(n_columns));
  std::vector<std::vector<double>> v_new(n_rows,
                                         std::vector<double>(n_columns));

  int t_save_index = 1;
  for (int t = 1; t < t_num; t++) {
    step_diffusion(u_old, u_new, v_old, v_new, u_coefficient, v_coefficient,
                   f_rate, k_rate, n_rows, n_columns);

    std::swap(u_old, u_new);
    std::swap(v_old, v_new);

    if (t % save_every == 0) {
      assert(t_save_index < u_history.size());
      assert(t_save_index < v_history.size());
      u_history[t_save_index] = u_old;
      v_history[t_save_index] = v_old;
      t_save_index++;
    }
  }
}

void step_diffusion(const std::vector<std::vector<double>> &u_old,
                    std::vector<std::vector<double>> &u_new,
                    const std::vector<std::vector<double>> &v_old,
                    std::vector<std::vector<double>> &v_new,
                    double u_coefficient, double v_coefficient, double f_rate,
                    double k_rate, int n_rows, int n_columns) {

#pragma omp parallel for collapse(2)
  for (int i = 1; i < n_rows - 1; i++) {
    for (int j = 1; j < n_columns - 1; j++) {
      u_new[i][j] = u_update(u_old, v_old, i, j, u_coefficient, f_rate);
      v_new[i][j] = v_update(u_old, v_old, i, j, v_coefficient, f_rate, k_rate);
    }
  }

  update_ghost_boundaries(u_new);
  update_ghost_boundaries(v_new);
}

double u_update(const std::vector<std::vector<double>> &u,
                const std::vector<std::vector<double>> &v, int i, int j,
                double u_coefficient, double f_rate) {
  double diffusion_update =
      u[i][j] + u_coefficient * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] +
                                 u[i][j + 1] - 4 * u[i][j]);
  double reaction_update =
      -u[i][j] * v[i][j] * v[i][j] + f_rate - f_rate * u[i][j];
  return diffusion_update + reaction_update;
}

double v_update(const std::vector<std::vector<double>> &u,
                const std::vector<std::vector<double>> &v, int i, int j,
                double v_coefficient, double f_rate, double k_rate) {
  double diffusion_update =
      v[i][j] + v_coefficient * (v[i - 1][j] + v[i + 1][j] + v[i][j - 1] +
                                 v[i][j + 1] - 4 * v[i][j]);
  double reaction_update =
      +u[i][j] * v[i][j] * v[i][j] - (f_rate + k_rate) * v[i][j];
  return diffusion_update + reaction_update;
}

void update_ghost_boundaries(std::vector<std::vector<double>> &matrix) {
  int n_rows = matrix.size();
  int n_columns = matrix[0].size();

  for (int i = 1; i < n_rows - 1; i++) {
    matrix[i][0] = matrix[i][n_columns - 2]; // left ghost
    matrix[i][n_columns - 1] = matrix[i][1]; // right ghost
  }

  for (int j = 1; j < n_columns - 1; j++) {
    matrix[0][j] = matrix[n_rows - 2][j]; // upper ghost
    matrix[n_rows - 1][j] = matrix[1][j]; // bottom ghost
  }
}

void initialize(std::vector<std::vector<double>> &u_init,
                std::vector<std::vector<double>> &v_init, double width,
                double u_value, double v_value, double perturbation,
                std::mt19937 &rng) {
  int n_columns = u_init[0].size();
  int n_rows = u_init.size();
  int half_square_width = n_columns * width / 2;
  int centre = n_columns / 2;

  perturbation = std::abs(perturbation);
  std::uniform_real_distribution<double> noise(-perturbation, perturbation);

  for (int j = 1; j < n_columns - 1; j++) {
    for (int i = 1; i < n_rows - 1; i++) {
      if (i >= centre - half_square_width && i < centre + half_square_width &&
          j >= centre - half_square_width && j < centre + half_square_width) {
        u_init[i][j] = u_value + noise(rng);
        v_init[i][j] = v_value + noise(rng);
      } else {
        u_init[i][j] = u_value + noise(rng);
        v_init[i][j] = std::abs(noise(rng));
      }
    }
  }
}

void save_data_txt(
    const std::vector<std::vector<std::vector<double>>> &data_collector,
    const std::string &output_filename) {
  int n_rows = data_collector[0].size();
  int n_columns = data_collector[0][0].size();

  std::ofstream output;
  output.open(output_filename);

  if (!output) {
    throw std::runtime_error("R8MAT_WRITE: Could not open file: " +
                             output_filename);
  }

  for (int t = 0; t < data_collector.size(); t++) {
    for (int i = 0; i < n_rows; i++) {
      for (int j = 1; j < n_columns - 1; j++) {
        output << "  " << std::setw(24) << std::setprecision(16)
               << data_collector[t][i][j];
      }
      output << "\n";
    }
    output << "\n";
  }
  output.close();
}

void save_metadata_txt(int x_num, int t_save, double x_delta,
                       double t_delta_save, double u_diff_constant,
                       double v_diff_constant, double f_rate, double k_rate,
                       const std::string &output_filename_xt,
                       const std::string &output_filename_params) {

  std::ofstream output;
  output.open(output_filename_params);

  if (!output) {
    throw std::runtime_error("R8MAT_WRITE: Could not open file: " +
                             output_filename_xt);
  }

  output << "{\n";
  output << "  \"x_num\": " << x_num << ",\n";
  output << "  \"t_save\": " << t_save << ",\n";
  output << "  \"x_delta\": " << x_delta << ",\n";
  output << "  \"t_delta_save\": " << t_delta_save << "\n";
  output << "  \"u_diff_constant\": " << u_diff_constant << ",\n";
  output << "  \"v_diff_constant\": " << v_diff_constant << ",\n";
  output << "  \"f_rate\": " << k_rate << ",\n";
  output << "  \"k_rate\": " << k_rate << ",\n";

  output << "}\n";

  output.close();

  output.open(output_filename_xt);

  if (!output) {
    throw std::runtime_error("R8MAT_WRITE: Could not open file: " +
                             output_filename_xt);
  }

  for (int t_idx = 0; t_idx < t_save; ++t_idx) {
    double t = t_idx * t_delta_save;
    output << "  " << std::setw(24) << std::setprecision(16) << t;
  }

  output << "\n";

  for (int x_idx = 0; x_idx < x_num; ++x_idx) {
    double x = x_idx * x_delta;
    output << "  " << std::setw(24) << std::setprecision(16) << x;
  }

  output << "\n";
  output.close();
}

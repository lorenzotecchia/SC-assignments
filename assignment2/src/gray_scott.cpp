#include "gray_scott.hpp"
#include <cassert>
#include <cmath>
#include <random>

void diffusion_step(const std::vector<double> &u_old,
                    std::vector<double> &u_new,
                    const std::vector<double> &v_old,
                    std::vector<double> &v_new, double u_coefficient,
                    double v_coefficient, int N) {

#pragma omp parallel for collapse(2)
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      u_new[i * N + j] = diffusion_update(u_old, i, j, u_coefficient, N);
      v_new[i * N + j] = diffusion_update(v_old, i, j, v_coefficient, N);
    }
  }
}

void reaction_step(std::vector<double> &u, std::vector<double> &v,
                   double f_rate, double k_rate, double dt, int N) {
#pragma omp parallel for collapse(2)
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      reaction_update(u[i * N + j], v[i * N + j], f_rate, k_rate, dt, 100,
                      1e-6);
    }
  }
}

double diffusion_update(const std::vector<double> &grid, int i, int j,
                        double coefficient, int N) {
  double sum = grid[i * N + j] +
               coefficient * (grid[(i - 1) * N + j] + grid[(i + 1) * N + j] +
                              grid[i * N + j - 1] + grid[i * N + j + 1] -
                              4 * grid[i * N + j]);
  return sum;
}

void reaction_update(double &u_ij, double &v_ij, double f_rate, double k_rate,
                     double dt, const int &max_iterations,
                     const double &tolerance) {
  double u_ij_old = u_ij;
  double v_ij_old = v_ij;
  for (int iter = 0; iter < max_iterations; ++iter) {
    double residual_u, residual_v;
    compute_residual(residual_u, residual_v, u_ij_old, u_ij, v_ij_old, v_ij,
                     f_rate, k_rate, dt);
    if (std::sqrt(residual_u * residual_u + residual_v * residual_v) <
        tolerance) {
      break;
    }

    double J11, J12, J21, J22;
    compute_jacobian(u_ij, v_ij, dt, f_rate, k_rate, J11, J12, J21, J22);

    double det = J11 * J22 - J12 * J21;
    double delta_u = (residual_u * J22 - residual_v * J12) / det;
    double delta_v = (-residual_u * J21 + residual_v * J11) / det;

    u_ij -= delta_u;
    v_ij -= delta_v;
  }
}

void compute_residual(double &residual_u, double &residual_v,
                      const double &u_ij_old, const double &u_ij_new,
                      const double &v_ij_old, const double &v_ij_new,
                      double f_rate, double k_rate, double dt) {
  residual_u = u_ij_new - u_ij_old +
               dt * (u_ij_new * v_ij_new * v_ij_new + f_rate * (u_ij_new - 1));
  residual_v =
      v_ij_new - v_ij_old -
      dt * (u_ij_new * v_ij_new * v_ij_new - (f_rate + k_rate) * v_ij_new);
}

void compute_jacobian(double u_ij_new, double v_ij_new, double dt,
                      double f_rate, double k_rate, double &J11, double &J12,
                      double &J21, double &J22) {
  J11 = 1 + dt * (v_ij_new * v_ij_new + f_rate);
  J12 = 2 * dt * u_ij_new * v_ij_new;
  J21 = -dt * v_ij_new * v_ij_new;
  J22 = 1 - dt * (2 * u_ij_new * v_ij_new - (f_rate + k_rate));
}

void update_periodic_boundaries(std::vector<double> &grid, int N) {
  for (int i = 1; i < N - 1; i++) {
    grid[i * N] = grid[i * N + N - 2];     // left ghost
    grid[i * N + N - 1] = grid[i * N + 1]; // right ghost
  }
  for (int j = 1; j < N - 1; j++) {
    grid[j] = grid[(N - 2) * N + j];     // upper ghost
    grid[(N - 1) * N + j] = grid[N + j]; // bottom ghost
  }
}

void update_insulated_boundaries(std::vector<double> &grid, int N) {
  for (int i = 0; i < N; i++) {
    grid[i * N] = grid[i * N + 1];             // left insulation
    grid[i * N + N - 1] = grid[i * N + N - 2]; // right insulation
  }
  for (int j = 1; j < N - 1; j++) {
    grid[j] = grid[N + j];                         // upper insulation
    grid[(N - 1) * N + j] = grid[(N - 2) * N + j]; // bottom insulation
  }
}

void initialize(std::vector<double> &u_init, std::vector<double> &v_init,
                double width, double u_value, double v_value,
                double perturbation, std::mt19937 &rng, int N) {
  int half_square_width = N * width / 2;
  int centre = N / 2;

  perturbation = std::abs(perturbation);
  std::uniform_real_distribution<double> noise(-perturbation, perturbation);

  for (int j = 1; j < N - 1; j++) {
    for (int i = 1; i < N - 1; i++) {
      if (i >= centre - half_square_width && i < centre + half_square_width &&
          j >= centre - half_square_width && j < centre + half_square_width) {
        u_init[i * N + j] = u_value + noise(rng);
        v_init[i * N + j] = v_value + noise(rng);
      } else {
        u_init[i * N + j] = u_value + noise(rng);
        v_init[i * N + j] = std::abs(noise(rng));
      }
    }
  }
}

void save_frame_txt(std::ofstream &output, const std::vector<double> &data,
                    int N) {
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      output << "  " << std::setw(24) << std::setprecision(16)
             << data[i * N + j];
    }
    output << "\n";
  }
  output << "\n";
}

void save_data_txt(const std::vector<std::vector<double>> &data_collector,
                   const std::string &output_filename, int N) {
  std::ofstream output;
  output.open(output_filename);

  if (!output) {
    throw std::runtime_error("Could not open file: " + output_filename);
  }

  for (size_t t = 0; t < data_collector.size(); t++) {
    save_frame_txt(output, data_collector[t], N);
  }
  output.close();
}

void save_metadata_txt(int x_num, int t_save, double x_delta,
                       double t_delta_save, double u_diff_constant,
                       double v_diff_constant, double f_rate, double k_rate,
                       double perturbation,
                       const std::string &output_filename_xt,
                       const std::string &output_filename_params) {
  std::ofstream output;
  output.open(output_filename_params);

  if (!output) {
    throw std::runtime_error("Could not open file: " + output_filename_params);
  }

  output << "{\n";
  output << "  \"x_num\": " << x_num << ",\n";
  output << "  \"t_save\": " << t_save << ",\n";
  output << "  \"x_delta\": " << x_delta << ",\n";
  output << "  \"t_delta_save\": " << t_delta_save << ",\n";
  output << "  \"u_diff_constant\": " << u_diff_constant << ",\n";
  output << "  \"v_diff_constant\": " << v_diff_constant << ",\n";
  output << "  \"f_rate\": " << f_rate << ",\n";
  output << "  \"k_rate\": " << k_rate << ",\n";
  output << "  \"perturbation\": " << perturbation << "\n";

  output << "}\n";

  output.close();

  output.open(output_filename_xt);

  if (!output) {
    throw std::runtime_error("Could not open file: " + output_filename_xt);
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

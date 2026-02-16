#include "fd_diffusion.hpp"
#include <cassert>

// TODO: maybe initial state could be anything, and ghost/fixed boundaries
// should be added inside simulate_diffusion.
std::vector<std::vector<std::vector<double>>>
simulate_diffusion(const std::vector<std::vector<double>> &initial_state,
                   int t_num, int save_every, double coefficient) {
  int n_columns = initial_state[0].size();
  int n_rows = initial_state.size();
  int t_save = (t_num + save_every - 1) / save_every;

  std::vector<std::vector<std::vector<double>>> data_collector(
      t_save,
      std::vector<std::vector<double>>(n_rows, std::vector<double>(n_columns)));

  data_collector[0] = initial_state;
  std::vector<std::vector<double>> old_state = initial_state;
  std::vector<std::vector<double>> new_state(n_rows,
                                             std::vector<double>(n_columns));
  create_fixed_boundaries(new_state);

  int t_save_index = 1;
  for (int t = 1; t < t_num; t++) {
    step_diffusion(old_state, new_state, coefficient, n_rows, n_columns);
    std::swap(old_state, new_state);

    if (t % save_every == 0) {
      assert(t_save_index < data_collector.size());
      data_collector[t_save_index++] = new_state;
    }
  }

  return data_collector;
}

void step_diffusion(const std::vector<std::vector<double>> &old_state,
                    std::vector<std::vector<double>> &new_state,
                    double coefficient, int n_rows, int n_columns) {

#pragma omp parallel for collapse(2)
  for (int i = 1; i < n_rows - 1; i++) {
    for (int j = 1; j < n_columns - 1; j++) {
      new_state[i][j] = cell_update(old_state, i, j, coefficient);
    }
  }

  update_ghost_boundaries(new_state);
  check_fixed_boundaries(new_state);
}

double cell_update(const std::vector<std::vector<double>> &matrix, int i, int j,
                   double coefficient) {
  double sum =
      matrix[i][j] +
      coefficient * (matrix[i - 1][j] + matrix[i + 1][j] + matrix[i][j - 1] +
                     matrix[i][j + 1] - 4 * matrix[i][j]);
  return sum;
}

void update_ghost_boundaries(std::vector<std::vector<double>> &matrix) {
  int n_rows = matrix.size();
  int n_columns = matrix[0].size();

  for (int i = 1; i < n_rows - 1; i++) {
    matrix[i][0] = matrix[i][n_columns - 2]; // left ghost
    matrix[i][n_columns - 1] = matrix[i][1]; // right ghost
  }
}

void create_fixed_boundaries(std::vector<std::vector<double>> &matrix) {
  int n_rows = matrix.size();
  int n_columns = matrix[0].size();

  double one = 1.0;
  double zero = 0.0;
  std::vector<double> vec_ones(n_columns, one);
  std::vector<double> vec_zeros(n_columns, zero);

  matrix[0] = vec_zeros;
  matrix[n_rows - 1] = vec_ones;
}

void check_fixed_boundaries(const std::vector<std::vector<double>> &matrix) {
  int n_rows = matrix.size();
  int n_columns = matrix[0].size();

  double one = 1.0;
  double zero = 0.0;
  std::vector<double> vec_ones(n_columns, one);
  std::vector<double> vec_zeros(n_columns, zero);

  assert(matrix[0] == vec_zeros);
  assert(matrix[n_rows - 1] == vec_ones);
}

// TODO maybe I can use less code and implement just one general function to
// save data
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
                       double t_delta_save, double diffusion_constant,
                       const std::string &output_filename_xt,
                       const std::string &output_filename_params) {

  std::ofstream output;
  output.open(output_filename_params);

  if (!output) {
    throw std::runtime_error("R8MAT_WRITE: Could not open file: " +
                             output_filename_xt);
  }

  output << "{\n";
  output << "  \"diffusion_constant\": " << diffusion_constant << ",\n";
  output << "  \"x_num\": " << x_num << ",\n";
  output << "  \"t_save\": " << t_save << ",\n";
  output << "  \"x_delta\": " << x_delta << ",\n";
  output << "  \"t_delta_save\": " << t_delta_save << "\n";
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

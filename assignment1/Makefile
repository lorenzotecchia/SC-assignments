CXX        := g++
CXX_OMP    := g++-15
CXXFLAGS   := -std=c++20 -Wall -Wextra -O2
EIGEN_INC  := -I/opt/homebrew/include/eigen3
PYTHON     := python3

SRC_DIR    := src
OUT_DIR    := output
OUT_PLOT_DIR := output/plots

.PHONY: all wave laplace diffusion build-wave build-laplace build-diffusion \
        run-wave run-laplace run-diffusion plot-wave plot-laplace plot-diffusion clean

# Default: build and run all
all: plot-wave plot-laplace plot-diffusion

# ── Wave equation ─────────────────────────────────────────────────────

WAVE_BIN   := wave_sim
WAVE_SRCS  := $(SRC_DIR)/main.cpp $(SRC_DIR)/fd1d_wave.cpp
WAVE_DATA  := $(OUT_DIR)/case1_leapfrog_wave_data.txt \
              $(OUT_DIR)/case2_leapfrog_wave_data.txt \
              $(OUT_DIR)/case3_leapfrog_wave_data.txt

wave: plot-wave

build-wave: $(WAVE_BIN)

$(WAVE_BIN): $(WAVE_SRCS) $(SRC_DIR)/fd1d_wave.hpp
	@echo "build: wave_sim"
	@$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -o $@ $(WAVE_SRCS)

$(WAVE_DATA): $(WAVE_BIN) | $(OUT_DIR)
	@echo "run:   wave_sim"
	@./$(WAVE_BIN)

run-wave: $(WAVE_DATA)

plot-wave: $(WAVE_DATA)
	@echo "plot:  wave"
	@$(PYTHON) scripts/plot_cpp_wave.py

# ── Laplace equation (Jacobi with OpenMP) ─────────────────────────────

LAPLACE_BIN  := laplace_sim
LAPLACE_SRCS := $(SRC_DIR)/laplace_main.cpp $(SRC_DIR)/laplace.cpp

laplace: plot-laplace

build-laplace: $(LAPLACE_BIN)

$(LAPLACE_BIN): $(LAPLACE_SRCS) $(SRC_DIR)/laplace.hpp
	@echo "build: laplace_sim"
	@$(CXX_OMP) $(CXXFLAGS) -fopenmp $(EIGEN_INC) -I$(SRC_DIR) -o $@ $(LAPLACE_SRCS)

LAPLACE_OUTPUTS := $(OUT_DIR)/laplace_solution.txt \
                   $(OUT_DIR)/laplace_sink.txt \
                   $(OUT_DIR)/laplace_insulator.txt \
                   $(OUT_DIR)/laplace_omega_sweep.txt

$(LAPLACE_OUTPUTS): $(LAPLACE_BIN) | $(OUT_DIR)
	@echo "run:   laplace_sim"
	@./$(LAPLACE_BIN)

run-laplace: $(LAPLACE_OUTPUTS)

plot-laplace: $(OUT_DIR)/laplace_solution.txt
	@echo "plot:  laplace"
	@$(PYTHON) scripts/plot_laplace.py

# ── Diffusion equation (OpenMP) ───────────────────────────────────────

DIFFUSION_BIN  := diffusion_sim
DIFFUSION_SRCS := $(SRC_DIR)/main_fd_diffusion.cpp $(SRC_DIR)/fd_diffusion.cpp

diffusion: plot-diffusion

build-diffusion: $(DIFFUSION_BIN)

$(DIFFUSION_BIN): $(DIFFUSION_SRCS) $(SRC_DIR)/fd_diffusion.hpp
	@echo "build: diffusion_sim"
	@$(CXX_OMP) $(CXXFLAGS) -fopenmp -I$(SRC_DIR) -o $@ $(DIFFUSION_SRCS)

$(OUT_DIR)/fd_diffusion_data.txt: $(DIFFUSION_BIN) | $(OUT_DIR)
	@echo "run:   diffusion_sim"
	@./$(DIFFUSION_BIN)

run-diffusion: $(OUT_DIR)/fd_diffusion_data.txt

plot-diffusion: $(OUT_DIR)/fd_diffusion_data.txt
	@echo "plot:  diffusion"
	@$(PYTHON) scripts/fd_diffusion.py

# ── Shared ────────────────────────────────────────────────────────────

$(OUT_DIR):
	@mkdir -p $(OUT_DIR)
	@mkdir -p $(OUT_PLOT_DIR)

clean:
	@rm -f $(WAVE_BIN) $(LAPLACE_BIN) $(DIFFUSION_BIN)
	@rm -f $(OUT_DIR)/*.txt
	@echo "clean: done"

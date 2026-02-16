CXX        := g++
CXX_OMP    := g++-15
CXXFLAGS   := -std=c++20 -Wall -Wextra -O2
EIGEN_INC  := -I/opt/homebrew/include/eigen3
PYTHON     := python3

SRC_DIR    := src
OUT_DIR    := output

.PHONY: all wave laplace build-wave build-laplace run-wave run-laplace \
        plot-wave plot-laplace clean

# Default: build and run both
all: plot-wave plot-laplace

# ── Wave equation ─────────────────────────────────────────────────────

WAVE_BIN   := wave_sim
WAVE_SRCS  := $(SRC_DIR)/main.cpp $(SRC_DIR)/fd1d_wave.cpp
WAVE_DATA  := $(OUT_DIR)/case1_wave_data.txt \
              $(OUT_DIR)/case2_wave_data.txt \
              $(OUT_DIR)/case3_wave_data.txt

wave: plot-wave

build-wave: $(WAVE_BIN)

$(WAVE_BIN): $(WAVE_SRCS) $(SRC_DIR)/fd1d_wave.hpp
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -o $@ $(WAVE_SRCS)

$(WAVE_DATA): $(WAVE_BIN) | $(OUT_DIR)
	./$(WAVE_BIN)

run-wave: $(WAVE_DATA)

plot-wave: $(WAVE_DATA)
	$(PYTHON) scripts/plot_cpp_wave.py

# ── Laplace equation (Jacobi with OpenMP) ─────────────────────────────

LAPLACE_BIN  := laplace_sim
LAPLACE_SRCS := $(SRC_DIR)/laplace_main.cpp $(SRC_DIR)/laplace.cpp

laplace: plot-laplace

build-laplace: $(LAPLACE_BIN)

$(LAPLACE_BIN): $(LAPLACE_SRCS) $(SRC_DIR)/laplace.hpp
	$(CXX_OMP) $(CXXFLAGS) -fopenmp $(EIGEN_INC) -I$(SRC_DIR) -o $@ $(LAPLACE_SRCS)

$(OUT_DIR)/laplace_solution.txt: $(LAPLACE_BIN) | $(OUT_DIR)
	./$(LAPLACE_BIN)

run-laplace: $(OUT_DIR)/laplace_solution.txt

plot-laplace: $(OUT_DIR)/laplace_solution.txt
	$(PYTHON) scripts/plot_laplace.py

# ── Shared ────────────────────────────────────────────────────────────

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

clean:
	rm -f $(WAVE_BIN) $(LAPLACE_BIN)
	rm -f $(OUT_DIR)/*.txt

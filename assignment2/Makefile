CXX        := g++-15
CXXFLAGS   := -std=c++20 -Wall -Wextra -O2 -fopenmp
EIGEN_INC  := -I/opt/homebrew/include/eigen3
BLAS_INC   := -I/opt/homebrew/opt/openblas/include
BLAS_LIB   := -L/opt/homebrew/opt/openblas/lib -lopenblas
BLAS_FLAGS := -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE
PYTHON     := python3

SRC_DIR    := src
OUT_DIR    := output

.PHONY: all build run plot clean

# ── DLA (Diffusion-Limited Aggregation) ───────────────────────────────

DLA_BIN   := dla_sim
DLA_SRCS  := $(SRC_DIR)/main_dla.cpp $(SRC_DIR)/sor.cpp

all: plot

build: $(DLA_BIN)

$(DLA_BIN): $(DLA_SRCS) $(SRC_DIR)/library.hpp
	@echo "build: dla_sim"
	@$(CXX) $(CXXFLAGS) $(EIGEN_INC) $(BLAS_INC) $(BLAS_FLAGS) -I$(SRC_DIR) -o $@ $(DLA_SRCS) $(BLAS_LIB)

DLA_OUTPUTS := $(OUT_DIR)/dla_concentration.txt $(OUT_DIR)/dla_cluster.txt

$(DLA_OUTPUTS): $(DLA_BIN) | $(OUT_DIR)
	@echo "run:   dla_sim"
	@./$(DLA_BIN)

run: $(DLA_OUTPUTS)

plot: $(DLA_OUTPUTS)
	@echo "plot:  dla"
	@$(PYTHON) scripts/plot_dla.py

# ── Sweep benchmark (eta × omega) ────────────────────────────────────

SWEEP_BIN  := sweep_bin
SWEEP_SRCS := $(SRC_DIR)/sweep_main.cpp $(SRC_DIR)/sor.cpp $(SRC_DIR)/rb_gauss_seidel.cpp

$(SWEEP_BIN): $(SWEEP_SRCS) $(SRC_DIR)/library.hpp
	@echo "build: sweep_bin"
	@$(CXX) $(CXXFLAGS) $(EIGEN_INC) $(BLAS_INC) $(BLAS_FLAGS) -I$(SRC_DIR) -o $@ $(SWEEP_SRCS) $(BLAS_LIB)

SWEEP_OUTPUTS := $(OUT_DIR)/sweep_eta_omega.csv $(OUT_DIR)/sweep_rbgs.csv

$(SWEEP_OUTPUTS): $(SWEEP_BIN) | $(OUT_DIR)
	@echo "run:   sweep_bin"
	@./$(SWEEP_BIN)

sweep: $(SWEEP_OUTPUTS)
	@echo "tables: generating LaTeX"
	@$(PYTHON) scripts/generate_tables.py

$(OUT_DIR):
	@mkdir -p $(OUT_DIR) $(OUT_DIR)/plots

clean:
	@rm -f $(DLA_BIN) sweep_bin
	@rm -f $(OUT_DIR)/*.txt $(OUT_DIR)/*.csv
	@echo "clean: done"

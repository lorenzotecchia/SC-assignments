CXX        := g++-15
CXXFLAGS   := -std=c++20 -Wall -Wextra -O2 -fopenmp
EIGEN_INC  := -I/opt/homebrew/include/eigen3
BLAS_INC   := -I/opt/homebrew/opt/openblas/include
BLAS_LIB   := -L/opt/homebrew/opt/openblas/lib -lopenblas
BLAS_FLAGS := -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE
PYTHON     := python3

SRC_DIR    := src
OUT_DIR    := output
PLOT_DIR   := $(OUT_DIR)/plots

ALL_FLAGS  := $(CXXFLAGS) $(EIGEN_INC) $(BLAS_INC) $(BLAS_FLAGS) -I$(SRC_DIR)

.PHONY: all build run plot clean cg eta sweep scaling

# ── DLA (Diffusion-Limited Aggregation) ───────────────────────────────

DLA_BIN   := dla_sim
DLA_SRCS  := $(SRC_DIR)/main_dla.cpp $(SRC_DIR)/sor.cpp

all: clean build run plot cg

build: $(DLA_BIN)

$(DLA_BIN): $(DLA_SRCS) $(SRC_DIR)/library.hpp
	@echo "build: $(DLA_BIN)"
	@$(CXX) $(ALL_FLAGS) -o $@ $(DLA_SRCS) $(BLAS_LIB)

DLA_OUTPUTS := $(OUT_DIR)/dla_concentration.txt $(OUT_DIR)/dla_cluster.txt

$(DLA_OUTPUTS): $(DLA_BIN) | $(OUT_DIR)
	@echo "run:   $(DLA_BIN)"
	@./$(DLA_BIN)

run: $(DLA_OUTPUTS)

plot: $(DLA_OUTPUTS)
	@echo "plot:  dla"
	@$(PYTHON) scripts/plot_dla.py

# ── Sweep benchmark (eta × omega) ────────────────────────────────────

SWEEP_BIN  := sweep_bin
SWEEP_SRCS := $(SRC_DIR)/sweep_main.cpp $(SRC_DIR)/sor.cpp $(SRC_DIR)/rb_gauss_seidel.cpp

$(SWEEP_BIN): $(SWEEP_SRCS) $(SRC_DIR)/library.hpp
	@echo "build: $(SWEEP_BIN)"
	@$(CXX) $(ALL_FLAGS) -o $@ $(SWEEP_SRCS) $(BLAS_LIB)

SWEEP_OUTPUTS := $(OUT_DIR)/sweep_eta_omega.csv $(OUT_DIR)/sweep_rbgs.csv

$(SWEEP_OUTPUTS): $(SWEEP_BIN) | $(OUT_DIR)
	@echo "run:   $(SWEEP_BIN)"
	@./$(SWEEP_BIN)

sweep: $(SWEEP_OUTPUTS)
	@echo "tables: generating LaTeX"
	@$(PYTHON) scripts/generate_tables.py
	@echo "plot:  sweep"
	@$(PYTHON) scripts/plot_sweep.py

# ── CG vs SOR comparison ──────────────────────────────────────────────

CG_BIN    := cg_compare
CG_SRCS   := $(SRC_DIR)/main_cg_compare.cpp $(SRC_DIR)/sor.cpp $(SRC_DIR)/cg.cpp

$(CG_BIN): $(CG_SRCS) $(SRC_DIR)/library.hpp
	@echo "build: $(CG_BIN)"
	@$(CXX) $(ALL_FLAGS) -o $@ $(CG_SRCS) $(BLAS_LIB)

cg: $(CG_BIN) | $(OUT_DIR)
	@./$(CG_BIN)
	@echo "plot:  cg_compare"
	@$(PYTHON) scripts/plot_cg_compare.py

# ── η investigation (uses dla_sim) ───────────────────────────────────

eta: $(DLA_BIN) | $(OUT_DIR)
	@./$(DLA_BIN)
	@echo "plot:  eta clusters"
	@$(PYTHON) scripts/plot_eta.py

# ── Grid-size scaling benchmark ──────────────────────────────────────

SCALE_BIN  := scaling_bin
SCALE_SRCS := $(SRC_DIR)/main_scaling.cpp $(SRC_DIR)/sor.cpp

$(SCALE_BIN): $(SCALE_SRCS) $(SRC_DIR)/library.hpp
	@echo "build: $(SCALE_BIN)"
	@$(CXX) $(ALL_FLAGS) -o $@ $(SCALE_SRCS) $(BLAS_LIB)

scaling: $(SCALE_BIN) | $(OUT_DIR)
	@./$(SCALE_BIN)
	@echo "plot:  scaling"
	@$(PYTHON) scripts/plot_scaling.py

# ── Directories & cleanup ────────────────────────────────────────────

$(OUT_DIR):
	@mkdir -p $(OUT_DIR) $(PLOT_DIR)

clean:
	@rm -f $(DLA_BIN) $(SWEEP_BIN) $(CG_BIN) $(SCALE_BIN)
	@rm -f $(OUT_DIR)/*.txt $(OUT_DIR)/*.csv
	# @rm -f $(PLOT_DIR)/*.png $(PLOT_DIR)/*.gif
	@echo "clean: done"

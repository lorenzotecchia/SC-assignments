CXX      := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -O2
PYTHON   := python3

SRC_DIR  := src
OUT_DIR  := output
BIN      := wave_sim

SRCS     := $(SRC_DIR)/main.cpp $(SRC_DIR)/fd1d_wave.cpp
DATA     := $(OUT_DIR)/case1_wave_data.txt \
            $(OUT_DIR)/case2_wave_data.txt \
            $(OUT_DIR)/case3_wave_data.txt

.PHONY: all build run plot clean

# Default: compile, simulate, and plot
all: plot

build: $(BIN)

$(BIN): $(SRCS) $(SRC_DIR)/fd1d_wave.hpp
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -o $@ $(SRCS)

$(DATA): $(BIN) | $(OUT_DIR)
	./$(BIN)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

run: $(DATA)

plot: $(DATA)
	$(PYTHON) scripts/plot_cpp_wave.py

clean:
	rm -f $(BIN)
	rm -f $(OUT_DIR)/*.txt

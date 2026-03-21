CXX       := $(shell command -v g++-15 >/dev/null 2>&1 && echo g++-15 || echo g++)
CXXFLAGS  := -std=c++20 -Wall -Wextra -O2 -fopenmp
EIGEN_INC := -I/opt/homebrew/include/eigen3
BLAS_INC  := -I/opt/homebrew/opt/openblas/include
BLAS_LIB  := -L/opt/homebrew/opt/openblas/lib -lopenblas
BLAS_FLAGS:= -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE
SRC       := src

ALL_FLAGS := $(CXXFLAGS) $(EIGEN_INC) $(BLAS_INC) $(BLAS_FLAGS) -I$(SRC)

BIN       := fd_karman
SRCS      := $(SRC)/main.cpp $(SRC)/mac_grid.cpp $(SRC)/pressure.cpp \
             $(SRC)/ns_solver.cpp $(SRC)/bc.cpp $(SRC)/io.cpp

.PHONY: all clean test

all: $(BIN)

$(BIN): $(SRCS) $(SRC)/mac_grid.hpp $(SRC)/pressure.hpp \
        $(SRC)/ns_solver.hpp $(SRC)/bc.hpp $(SRC)/io.hpp
	@echo "build: $(BIN)"
	@$(CXX) $(ALL_FLAGS) -o $@ $(SRCS) $(BLAS_LIB)

TEST_BIN  := test_ns
TEST_SRCS := $(SRC)/test_ns.cpp $(SRC)/mac_grid.cpp $(SRC)/pressure.cpp \
             $(SRC)/bc.cpp $(SRC)/io.cpp

$(TEST_BIN): $(TEST_SRCS) $(SRC)/mac_grid.hpp $(SRC)/pressure.hpp \
             $(SRC)/bc.hpp $(SRC)/io.hpp
	@echo "build: $(TEST_BIN)"
	@$(CXX) $(ALL_FLAGS) -o $@ $(TEST_SRCS) $(BLAS_LIB)

test: $(TEST_BIN)
	@./$(TEST_BIN)

clean:
	@rm -f $(BIN) $(TEST_BIN)
	@rm -f output/*.csv output/*.txt
	@echo "clean: done"

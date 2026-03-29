#include "io.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>

void write_force_csv(const std::string &path, double t, double cd, double cl) {
    bool new_file = !std::ifstream(path).good();
    FILE *f = std::fopen(path.c_str(), "a");
    if (!f) return;
    if (new_file) std::fprintf(f, "time,cd,cl\n");
    std::fprintf(f, "%.6f,%.8f,%.8f\n", t, cd, cl);
    std::fclose(f);
}

void write_vorticity(const std::string &path, const MacGrid &g) {
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f) return;
    // Vorticity at cell centres: ω = ∂v/∂x − ∂u/∂y
    for (int i = 1; i < g.nx-1; ++i) {
        for (int j = 1; j < g.ny-1; ++j) {
            double dvdx = (g.v(i+1,j) - g.v(i-1,j)) / (2*g.dx);
            double dudy = (g.u(i,j+1) - g.u(i,j-1)) / (2*g.dy);
            double x = (i + 0.5)*g.dx, y = (j + 0.5)*g.dy;
            std::fprintf(f, "%.5f %.5f %.6f\n", x, y, dvdx - dudy);
        }
    }
    std::fclose(f);
}

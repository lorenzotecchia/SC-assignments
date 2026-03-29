#include "mac_grid.hpp"
#include "ns_solver.hpp"
#include "bc.hpp"
#include "io.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <filesystem>

static void usage(const char *prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --Re     <float>   Reynolds number      (default: 100)\n"
        "  --nx     <int>     Grid cells in x      (default: 440)\n"
        "  --ny     <int>     Grid cells in y      (default: 82)\n"
        "  --dt     <float>   Time step             (default: 0.001)\n"
        "  --tend   <float>   End time              (default: 20.0)\n"
        "  --omega  <float>   SOR relaxation        (default: 1.8)\n"
        "  --outdir <str>     Output directory      (default: output)\n"
        "  --snap   <int>     Vorticity snap every N steps (default: 500)\n",
        prog);
    std::exit(1);
}

int main(int argc, char **argv) {
    // ── defaults ──
    double Re=100, dt=0.001, tend=20.0, omega=1.8;
    int    nx=440, ny=82, snap=500;
    std::string outdir = "output";

    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i],"--Re"))     Re     = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i],"--nx"))     nx     = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--ny"))     ny     = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"--dt"))     dt     = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i],"--tend"))   tend   = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i],"--omega"))  omega  = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i],"--outdir")) outdir = argv[++i];
        else if (!std::strcmp(argv[i],"--snap"))   snap   = std::atoi(argv[++i]);
        else usage(argv[0]);
    }

    const double Lx=2.2, Ly=0.41, D=0.1, U_mean=1.0;
    const double nu = U_mean * D / Re;

    std::printf("════════════════════════════════════════════\n");
    std::printf("  FD Karman Vortex Street Solver\n");
    std::printf("════════════════════════════════════════════\n");
    std::printf("  Re     = %.1f\n", Re);
    std::printf("  nu     = %.6f\n", nu);
    std::printf("  grid   = %d x %d\n", nx, ny);
    std::printf("  dt     = %.5f\n", dt);
    std::printf("  tend   = %.1f\n", tend);
    std::printf("  CFL    = %.3f\n", dt * U_mean * 1.5 / (Lx/nx));
    std::printf("════════════════════════════════════════════\n");

    MacGrid g(nx, ny, Lx, Ly, 0.2, 0.2, 0.05);
    apply_velocity_bc(g, U_mean);

    NSSolverConfig cfg{nu, dt, U_mean, omega, 1e-4, 5000};
    Eigen::MatrixXd Nu_prev = Eigen::MatrixXd::Zero(nx+1, ny);
    Eigen::MatrixXd Nv_prev = Eigen::MatrixXd::Zero(nx, ny+1);

    std::filesystem::create_directories(outdir);
    std::string csv_path = outdir + "/drag_lift.csv";
    // Clear old CSV
    std::remove(csv_path.c_str());

    int steps = static_cast<int>(tend / dt);
    for (int s = 0; s < steps; ++s) {
        auto stats  = ns_step(g, cfg, s, Nu_prev, Nv_prev);
        auto forces = compute_forces(g, U_mean, D);
        double t    = (s+1) * dt;

        write_force_csv(csv_path, t, forces.cd, forces.cl);

        if (s % snap == 0) {
            char fname[256];
            std::snprintf(fname, sizeof(fname), "%s/vorticity_%06d.txt",
                          outdir.c_str(), s);
            write_vorticity(fname, g);
            std::printf("  t=%8.3f  Cd=%+8.4f  Cl=%+8.4f  "
                        "div=%6.2e  p_iters=%d\n",
                        t, forces.cd, forces.cl,
                        stats.div_max, stats.p_iters);
        }
    }

    std::printf("\nDone. Output in %s/\n", outdir.c_str());
    return 0;
}

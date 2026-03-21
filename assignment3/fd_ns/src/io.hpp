#pragma once
#include "mac_grid.hpp"
#include <string>
#include <vector>

// Append one row to drag_lift.csv (creates file on first call).
void write_force_csv(const std::string &path, double t, double cd, double cl);

// Write vorticity field as plain-text grid (x y omega) for matplotlib.
void write_vorticity(const std::string &path, const MacGrid &g);

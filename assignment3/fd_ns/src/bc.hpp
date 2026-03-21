#pragma once
#include "mac_grid.hpp"

// Apply all velocity BCs to the MAC grid (call after each projected step).
// U_mean: mean inflow velocity; parabolic profile: u_in = 1.5*U*4y(H-y)/H²
void apply_velocity_bc(MacGrid &g, double U_mean);

// Enforce no-slip inside and on all solid (mask==1) cells.
void apply_solid_bc(MacGrid &g);

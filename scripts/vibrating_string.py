import sys
from pathlib import Path
from scipy.stats import qmc
import numpy as np
import math

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.wave_equation import update_wave, simulate_wave, animate

L = 1
c = 1
time_steps = 1e4
N = 500
delta_x = L / N
delta_t = 0.001
initial_x = np.sin(2 * np.pi * delta_x * np.array(range(N)))
time_steps = 1000

print(f"c * delta_t / delta_x = {c*delta_t/delta_x}")
wave_states = simulate_wave(initial_x, delta_x, delta_t, c, time_steps)

animate(wave_states, delta_x)

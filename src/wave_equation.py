import numpy as np
from numpy import ndarray


def update_wave(x_current: ndarray, x_old: ndarray, const2: float) -> ndarray:
    """
    give me a sec and I will explain what happend.
    """
    x_new = x_old.copy()
    x_new[1, -2] = (
        2 * x_current[1, -2]
        - x_old[1, -2]
        + const2 * (x_current[2, -1] - 2 * x_current[1, -2] + x_current[0, -3])
    )


def simulate_wave(
    initial_x: ndarray, delta_x: float, delta_t: float, c: float, time_steps
):
    """
    repeats update_wave for a chosen amount of time_steps.
    Should return ndarray of state at each step?
    """
    None

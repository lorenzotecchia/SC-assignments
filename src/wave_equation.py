import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


def update_wave(x_current: ndarray, x_old: ndarray, const: float) -> ndarray:
    """
    Update the wave function given the two previous steps.
    The boundaries are kept fixed at zero.

    x_current: values of the wave function at time t, for each x.
    x_old: values of the wave function at time t - dt, for each x.
    const: constant term that accounts both for wave speed and time and space's step size.

    Returns the values of the wave function at time t + dt, for each x.
    """
    x_new = np.zeros_like(x_current)
    x_new[1:-1] = (
        2 * x_current[1:-1]
        - x_old[1:-1]
        + const * (x_current[2:] - 2 * x_current[1:-1] + x_current[0:-2])
    )
    return x_new


def simulate_wave(
    initial_x: ndarray,
    delta_x: float,
    delta_t: float,
    c: float,
    time_steps: int,
):
    """
    Repeats update_wave for a chosen amount of time_steps,
    with the wave being at rest at time t = 0.

    initial_x: initial condition, i.e. the values of the wave function at time = 0
    delta_x: space step size
    delta_t: time step size
    c: propagation speed of the wave
    time_steps: number of time steps to run the simulation

    Return ndarray of states at each step
    """
    constant = c * delta_t / delta_x
    constant2 = constant * constant

    wave_states = np.zeros((int(time_steps), int(initial_x.shape[0])))

    wave_states[0, :] = initial_x
    wave_states[1, :] = initial_x  # zero initial velocity

    for i in range(2, time_steps):
        x_old = wave_states[i - 2, :]
        x_current = wave_states[i - 1, :]

        wave_states[i, :] = update_wave(x_current, x_old, constant2)

    return wave_states


def animate(wave_states, delta_x):
    fig, ax = plt.subplots()
    time_steps = wave_states.shape[0]
    n_points = int(wave_states.shape[1])
    x = delta_x * np.array(range(n_points))

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(wave_states.min(), wave_states.max())

    (line,) = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return (line,)

    def update(i):
        y = wave_states[i]
        line.set_data(x, y)

        return (line,)

    ani = animation.FuncAnimation(
        fig=fig, func=update, init_func=init, frames=time_steps, interval=30
    )
    plt.savefig("output/plots/wave_sim.eps", fomrat="eps")
    plt.show()

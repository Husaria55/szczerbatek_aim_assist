import numpy as np

# constants
AIR_DENSITY = 1.225
GRAVITY_VECTOR = np.array([0, 0, -9.81])
WIND_VECTOR = np.array([0, 0, 0])


def calculate_state_derivative(
    t: float, state: np.ndarray, mass: float, cd: float, area: float
) -> np.ndarray:
    # state
    x, y, z, v_x, v_y, v_z = state
    v_payload = np.array([v_x, v_y, v_z])
    v_relative = v_payload - WIND_VECTOR
    v_rel_magnitude = np.linalg.norm(v_relative)
    drag_vector = -0.5 * AIR_DENSITY * cd * area * v_rel_magnitude * v_relative

    a_vector = (drag_vector / mass) + GRAVITY_VECTOR

    return np.concatenate([v_payload, a_vector])


def rk4_step(
    t: float, dt: float, state: np.ndarray, mass: float, cd: float, area: float
) -> np.ndarray:
    k1 = calculate_state_derivative(t, state, mass, cd, area)
    k2 = calculate_state_derivative(t + dt / 2, state + k1 * (dt / 2), mass, cd, area)
    k3 = calculate_state_derivative(t + dt / 2, state + k2 * (dt / 2), mass, cd, area)
    k4 = calculate_state_derivative(t + dt, state + k3 * dt, mass, cd, area)

    next_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state


def simulate_drop(
    initial_state: np.ndarray, mass: float, cd: float, area: float, dt: float = 0.01
) -> tuple[list[float], list[np.ndarray]]:
    time_history = []
    state_history = []

    current_time = 0.0
    current_state = initial_state

    while current_state[2] > 0:
        state_history.append(current_state)
        time_history.append(current_time)
        current_state = rk4_step(current_time, dt, current_state, mass, cd, area)
        current_time += dt

    return time_history, state_history

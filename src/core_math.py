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

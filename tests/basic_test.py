import numpy as np
import pytest

from szczerbatek_aim_assist.core_math import simulate_drop


def test_standard_payload_drop():
    # input
    mass = 2.0
    cd = 0.45
    area = 0.03

    initial_state = np.array([0.0, 0.0, 50.0, 25.0, 0.0, 0.0])

    times, states = simulate_drop(initial_state, mass, cd, area)

    final_time = times[-1]
    final_state = states[-1]

    assert final_time == pytest.approx(3.39, rel=0.01)
    assert final_state[0] == pytest.approx(71.3, rel=0.01)

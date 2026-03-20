import numpy as np
import pytest
from szczerbatek_aim_assist.core_math import DropSolver, SimulationEnvironment

env = SimulationEnvironment(wind_model=lambda pos, t: np.array([5.0, 0.0, 0.0]))


def test_drop_solver_simple_case():
    target_position = np.array([100.0, 0.0, 0.0])
    mass = 2.0
    cd = 0.45
    area = 0.03
    solver = DropSolver(mass, cd, area)
    approach_altitude = 50.0
    velocity_vector = np.array([20.0, 5.0, 0.0])
    release_point = solver.calculate_release_point(
        target_position, approach_altitude, velocity_vector, env
    )
    assert release_point is not None
    assert release_point == pytest.approx(np.array([39.14, -14.7]), rel=0.01)

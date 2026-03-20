from typing import Callable
from scipy.optimize import minimize
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SimulationEnvironment:
    # constants
    air_density: float = 1.225
    gravity_vector: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    target_elevation: float = 0.0
    wind_model: Callable[[np.ndarray, float], np.ndarray] = field(
        default_factory=lambda: create_constant_wind(np.array([0.0, 0.0, 0.0]))
    )


def create_constant_wind(
    wind_vector: np.ndarray,
) -> Callable[[np.ndarray, float], np.ndarray]:
    def constant_wind(position: np.ndarray, time: float) -> np.ndarray:
        return wind_vector

    return constant_wind


def create_shear_wind(
    base_wind_vector: np.ndarray, shear_exponent: float
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Wind model where wind speed increases with altitude following a power law. The base_wind_vector is the wind at 10m altitude."""

    def shear_wind(position: np.ndarray, time: float) -> np.ndarray:
        height = position[2]
        if height < 0:
            height = 0
        return base_wind_vector * ((height / 10.0) ** shear_exponent)

    return shear_wind


def calculate_state_derivative(
    t: float,
    state: np.ndarray,
    mass: float,
    cd: float,
    area: float,
    env: SimulationEnvironment | None = None,
) -> np.ndarray:
    if env is None:
        env = SimulationEnvironment()
    # state
    v_payload = state[3:]
    v_relative = v_payload - env.wind_model(state[:3], t)
    v_rel_magnitude = np.linalg.norm(v_relative)
    drag_vector = -0.5 * env.air_density * cd * area * v_rel_magnitude * v_relative

    a_vector = (drag_vector / mass) + env.gravity_vector

    return np.concatenate([v_payload, a_vector])


def rk4_step(
    t: float,
    dt: float,
    state: np.ndarray,
    mass: float,
    cd: float,
    area: float,
    env: SimulationEnvironment | None = None,
) -> np.ndarray:
    if env is None:
        env = SimulationEnvironment()
    k1 = calculate_state_derivative(t, state, mass, cd, area, env)
    k2 = calculate_state_derivative(
        t + dt / 2, state + k1 * (dt / 2), mass, cd, area, env
    )
    k3 = calculate_state_derivative(
        t + dt / 2, state + k2 * (dt / 2), mass, cd, area, env
    )
    k4 = calculate_state_derivative(t + dt, state + k3 * dt, mass, cd, area, env)

    next_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return next_state


def simulate_drop(
    initial_state: np.ndarray,
    mass: float,
    cd: float,
    area: float,
    dt: float = 0.01,
    env: SimulationEnvironment | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if env is None:
        env = SimulationEnvironment()
    time_history = []
    state_history = []

    current_time = 0.0
    current_state = initial_state

    while current_state[2] > env.target_elevation:
        state_history.append(current_state)
        time_history.append(current_time)
        current_state = rk4_step(current_time, dt, current_state, mass, cd, area, env)
        current_time += dt

    return np.array(time_history), np.array(state_history)


class DropSolver:
    def __init__(self, mass: float, cd: float, area: float) -> None:
        self.mass = mass
        self.area = area
        self.cd = cd

    def calculate_release_point(
        self,
        target_position: np.ndarray,
        approach_altitude: float,
        approach_velocity: np.ndarray,
        env: SimulationEnvironment | None = None,
    ) -> np.ndarray:
        initial_state = np.concatenate(
            [np.array([0.0, 0.0, approach_altitude]), approach_velocity]
        )
        times, positions = simulate_drop(
            initial_state, self.mass, self.cd, self.area, env=env
        )
        displacement = positions[-1, :2]
        drop_position = target_position[:2] - displacement
        return drop_position


class ShootingSolver:
    def __init__(self, mass: float, cd: float, area: float) -> None:
        self.mass = mass
        self.area = area
        self.cd = cd

    def _objective_function(
        self,
        guess_xy: np.ndarray,
        target_position: np.ndarray,
        approach_altitude: float,
        approach_velocity: np.ndarray,
        env: SimulationEnvironment | None = None,
    ):
        inital_state = np.concatenate(
            [guess_xy, [approach_altitude], approach_velocity]
        )
        times, positions = simulate_drop(
            inital_state, self.mass, self.cd, self.area, env=env
        )
        final_position = positions[-1, :3]
        return np.linalg.norm(final_position - target_position)

    def calculate_release_point(
        self,
        target_position: np.ndarray,
        approach_altitude: float,
        approach_velocity: np.ndarray,
        env: SimulationEnvironment | None = None,
    ) -> np.ndarray:
        if env is None:
            env = SimulationEnvironment()
        time_to_fall = np.sqrt(2 * approach_altitude / -env.gravity_vector[2])
        initial_guess = target_position[:2] - approach_velocity[:2] * time_to_fall
        result = minimize(
            fun=self._objective_function,
            x0=initial_guess,
            args=(target_position, approach_altitude, approach_velocity, env),
            method="Nelder-Mead",
            tol=0.1,
        )
        if not result.success:
            print("Optimization failed:", result.message)

        return result.x

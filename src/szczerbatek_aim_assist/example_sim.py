import matplotlib.pyplot as plt
import numpy as np

import payload_trajectory_solver.core_math as rk

# cd, are guesses
mass = 0.158  # kg
cd_payload = 1.0
area_payload = 0.0104
cd_parachute = 0.8
area_parachute = 0.159
deploy_time = 1.5
opening_duration = 0.5

drag_area_func = rk.create_drag_area_model(
    cd_payload,
    area_payload,
    cd_parachute,
    area_parachute,
    deploy_time,
    opening_duration,
)

env = rk.SimulationEnvironment(
    wind_model=rk.create_shear_wind(np.array([5.0, 5.0, 0.0]), 1 / 7)
)

# env = rk.SimulationEnvironment(
#     wind_model=rk.create_logarithmic_wind(np.array([5.0, 5.0, 0.0]), 60, 0.03)
# )

initial_state = np.array([0.0, 0.0, 60.0, 23.0, 0.0, 0.0])
times, states = rk.simulate_drop(initial_state, mass, drag_area_func, dt=0.01, env=env)

print(f"Final State (Landing): {states[-1]}")
print(f"Time to land: {times[-1]:.2f} seconds")

# Plotting 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(states[:, 0], states[:, 1], states[:, 2], color="b", label="Payload Path")
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("3D Trajectory of Dropped Object with Parachute")
ax.legend()
plt.show()


# finding release point
target_position = np.array([0.0, 0.0, 0.0])
approach_altitude = 60.0
velocity_vector = np.array([23.0, 0.0, 0.0])

solver = rk.ShootingSolver(mass, drag_area_func, release_latency=0.5)

release_point = solver.calculate_release_point(
    target_position, approach_altitude, velocity_vector, env
)

print(f"Optimal command coordinate for target at {target_position}: {release_point}")

plt.figure()
plt.plot(target_position[0], target_position[1], "ro", markersize=8, label="Target")
plt.plot(
    release_point[0],
    release_point[1],
    "go",
    markersize=8,
    label="Optimal Drop Command Point",
)

# taking delay into account
actual_physical_drop_point = release_point + (
    velocity_vector[:2] * solver.release_latency
)
plt.plot(
    actual_physical_drop_point[0],
    actual_physical_drop_point[1],
    "mo",
    markersize=6,
    label="Actual Physical Drop Point",
)

# Run the simulation from the physical drop point
optimal_initial_state = np.concatenate(
    [actual_physical_drop_point, [approach_altitude], velocity_vector]
)
_, check_states = rk.simulate_drop(
    optimal_initial_state, mass, drag_area_func, dt=0.01, env=env
)

# Plot the payload's path over the ground
plt.plot(check_states[:, 0], check_states[:, 1], "b--", label="Payload Ground Track")

plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Shooting Solver Result & Trajectory Verification")
plt.legend()
plt.grid(True)
plt.axis("equal")  # Ensures the X and Y axes have the same scale
plt.show()

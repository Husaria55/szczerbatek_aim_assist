import numpy as np
import szczerbatek_aim_assist.core_math as rk
import matplotlib.pyplot as plt

# input
mass = 0.2
cd = 0.5
area = 0.03
z = 50
v_x = 25
x = 0
y = 0
env = rk.SimulationEnvironment(wind_vector=np.array([0.0, 10.0, 0.0]))
initial_state = np.array([0.0, 0.0, 50.0, 25.0, 0.0, 0.0])

times, states = rk.simulate_drop(initial_state, mass, cd, area, env=env)

print(states[-1])
print(times[-1])

# plotting 3D trajectory
x_vals = [state[0] for state in states]
y_vals = [state[1] for state in states]
z_vals = [state[2] for state in states]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x_vals, y_vals, z_vals)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_zlabel("Z Position (m)")
ax.set_title("3D Trajectory of Dropped Object")
plt.show()


# example finding release point for a target at (100, 0, 0)
# wirh non zero wind

target_position = np.array([100.0, 100.0, 0.0])
solver = rk.DropSolver(mass, cd, area)
approach_altitude = 50.0
velocity_vector = np.array([25.0, 0.0, 0.0])
release_point = solver.calculate_release_point(
    target_position, approach_altitude, velocity_vector, env
)
print(f"Release point for target at {target_position}: {release_point}")
# wisualkization of release point and target
plt.figure()
plt.plot(target_position[0], target_position[1], "ro", label="Target")
plt.plot(release_point[0], release_point[1], "go", label="Release Point")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.title("Release Point and Target Location")
plt.legend()
plt.grid()
plt.show()

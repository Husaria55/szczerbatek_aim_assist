import numpy as np
import szczerbatek_aim_assist.core_math as rk
import matplotlib.pyplot as plt

# input
mass = 2.0
cd = 0.45
area = 0.03
z = 50
v_x = 25
x = 0
y = 0

initial_state = np.array([0.0, 0.0, 50.0, 25.0, 0.0, 0.0])

times, states = rk.simulate_drop(initial_state, mass, cd, area)

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

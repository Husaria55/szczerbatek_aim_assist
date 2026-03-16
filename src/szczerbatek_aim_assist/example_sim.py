import numpy as np
import szczerbatek_aim_assist.core_math as rk

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

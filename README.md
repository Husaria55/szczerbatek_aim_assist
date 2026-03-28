# payload_trajectory_solver

## How to use it
In the [core_math.py](./src/payload_trajectory_solver/core_math.py) there are:
- SimulationEnvironment class for setting the constants and wind
- simulate_drop function for basic trajectory calculation
- DropSolver class for finding optimal drop position by using displacement
- ShootingSolver class that uses minimize from scipy - useful if the wind depends on x,y (e.g. terrain)


## Dev tools
- poetry
- ruff


## Update after meeting
### Payload
1. length 50.5mm, diameter 120mm, mass 150g
2. length  140mm, diameter 65mm, mass 255g
Shape more or less cylindrical, first beacon, second bottle of water


### Parachute
Import new consideration is that we drop payload wirh parachute.
Diameter 45cm
By KLIMA (german firm)


### Probable initial condition
- drop altitude 60m
- airplane speed 23m/s

### other input
- wind vector
- wind profile
- air density
- mechanism latency

### Simulation goal
Find the optimal drop position to reach the target


### How it will work
1. the parachutes needs time to open and start working, in this phase the payload is moving nearly horizontally with ~23m/s speed
2. parachutes works as a gigantic air brake, which completely slows down the horizontal velocity
3. now the main mover of the payload is wind, with it the payload drifts to the landing spot


### How I simulate it
1. from t=0s to t= ~1.5s (we have to find out during tests how long it takes for parachute to open)
- cd and refernce area of payload
2. somewhere between transition
3. from t= ~1.5s to landing
- cd and refernce area of parachute
Trajectory calculated using rk4


### What we need to find out during tests:
opening time, cd for payload and parachute
this parameters should we setup in this way that the simualtion gives sensible prediction of where we should drop to reach the target

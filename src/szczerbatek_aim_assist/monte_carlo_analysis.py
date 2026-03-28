import matplotlib.pyplot as plt
import numpy as np

import szczerbatek_aim_assist.core_math as rk

# done with AI
# --- 1. Base Simulation Parameters ---
cd_values_to_test = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
initial_state = np.array([0.0, 0.0, 60.0, 23.0, 0.0, 0.0])
area_parachute = 0.159  # 45cm diameter

# Deployment timings
deploy_time = 1.5
opening_duration = 0.5

# Wind Models
wind_models = [
    {
        "name": "Negligible Wind",
        "func": rk.create_constant_wind(np.array([0.0, 0.0, 0.0])),
    },
    {
        "name": "Medium Wind (5m/s shear)",
        "func": rk.create_shear_wind(np.array([5.0, 5.0, 0.0]), 1 / 7),
    },
    {
        "name": "Strong Wind (10m/s shear)",
        "func": rk.create_shear_wind(np.array([10.0, 10.0, 0.0]), 1 / 7),
    },
]

# Payload Definitions (Adding approximate frontal areas for the freefall phase)
payloads = [
    {
        "name": "Payload 1 (158g, Wide)",
        "mass": 0.158,
        "cd_payload": 0.8,
        "area_payload": 0.0113,  # ~120mm diameter
    },
    {
        "name": "Payload 2 (263g, Narrow)",
        "mass": 0.263,
        "cd_payload": 0.8,
        "area_payload": 0.0033,  # ~65mm diameter
    },
]

# --- 2. Run Simulations ---
# Store results in a nested dictionary for easier plotting
results = {
    wind["name"]: {p["name"]: {"x": [], "y": []} for p in payloads}
    for wind in wind_models
}

for wind in wind_models:
    env = rk.SimulationEnvironment(wind_model=wind["func"])

    for payload in payloads:
        for test_cd in cd_values_to_test:
            # Create a specific drag profile for this exact combination
            drag_func = rk.create_drag_area_model(
                cd_payload=payload["cd_payload"],
                area_payload=payload["area_payload"],
                cd_parachute=test_cd,  # This is where we sweep the C_D
                area_parachute=area_parachute,
                deploy_time=deploy_time,
                opening_duration=opening_duration,
            )

            # Run the drop
            times, states = rk.simulate_drop(
                initial_state=initial_state,
                mass=payload["mass"],
                drag_area_func=drag_func,
                dt=0.01,
                env=env,
            )

            # Record final X, Y
            final_pos = states[-1][:3]
            results[wind["name"]][payload["name"]]["x"].append(final_pos[0])
            results[wind["name"]][payload["name"]]["y"].append(final_pos[1])


# --- 3. Plotting Results ---
for wind_name, payload_data in results.items():
    plt.figure(figsize=(8, 6))
    plt.title(f"Landing Footprint Sweep - {wind_name}")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")

    # We will use a colormap to show how C_D changes the landing spot
    markers = ["o", "s"]  # Circle for Payload 1, Square for Payload 2

    for idx, (payload_name, coords) in enumerate(payload_data.items()):
        # Scatter plot with color mapped to the C_D values
        sc = plt.scatter(
            coords["x"],
            coords["y"],
            c=cd_values_to_test,
            cmap="viridis",
            marker=markers[idx],
            s=80,
            edgecolor="k",
            label=payload_name,
        )

        # Connect the dots with a thin line to show the trajectory trend
        plt.plot(coords["x"], coords["y"], linestyle="--", color="gray", alpha=0.5)

    # Add a colorbar so we know which point corresponds to which C_D
    cbar = plt.colorbar(sc)
    cbar.set_label("Parachute $C_D$ Value")

    plt.legend()
    plt.grid(True)
    plt.axis("equal")  # Keep X and Y scale 1:1 so drift angles look realistic
    plt.show()

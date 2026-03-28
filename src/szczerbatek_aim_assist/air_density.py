# Air density calculation module ()
def calculate_air_density(
    temperature_c: float, pressure_hpa: float, relative_humidity_percent: float
) -> float:
    """
    Calculates accurate air density accounting for temperature, pressure, and humidity.

    Args:
        temperature_c: Temperature in Celsius (e.g., 20.0)
        pressure_hpa: Absolute station pressure in hPa or mbar (e.g., 1013.25)
        relative_humidity_percent: Relative humidity from 0 to 100 (e.g., 50.0)

    Returns:
        Air density in kg/m^3
    """
    # Constants
    R_d = 287.058  # Gas constant for dry air, J/(kg*K)
    R_v = 461.495  # Gas constant for water vapor, J/(kg*K)

    # Conversions
    temp_k = temperature_c + 273.15
    pressure_pa = pressure_hpa * 100.0
    rh_decimal = relative_humidity_percent / 100.0

    # Calculate saturation vapor pressure (Tetens formula)
    exponent = (7.5 * temperature_c) / (temperature_c + 237.3)
    p_sat = 6.1078 * (10**exponent) * 100.0  # Multiplied by 100 to convert hPa to Pa

    # Calculate actual vapor pressure
    p_v = rh_decimal * p_sat

    # Calculate dry air pressure
    p_d = pressure_pa - p_v

    # Calculate total humid air density
    density = (p_d / (R_d * temp_k)) + (p_v / (R_v * temp_k))

    return density


rho = calculate_air_density(25.0, 1013.25, 60.0)
print(f"Air density: {rho:.4f} kg/m^3")

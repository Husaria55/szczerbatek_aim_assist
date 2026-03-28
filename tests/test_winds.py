import numpy as np
from numpy.testing import assert_array_almost_equal

# Assuming your functions are in a file/module named core_math.py
from szczerbatek_aim_assist.core_math import (
    create_logarithmic_wind,
    create_shear_wind,
)


class TestShearWind:
    def test_shear_wind_at_reference_height(self):
        """At exactly 10m, the wind should equal the base wind vector."""
        base_wind = np.array([5.0, 2.0, 0.0])
        wind_func = create_shear_wind(base_wind, shear_exponent=1 / 7)
        pos = np.array([0.0, 0.0, 10.0])

        result = wind_func(pos, time=0.0)
        assert_array_almost_equal(result, base_wind)

    def test_shear_wind_at_ground(self):
        """At 0m, the wind should be zero."""
        base_wind = np.array([5.0, 2.0, 0.0])
        wind_func = create_shear_wind(base_wind, shear_exponent=1 / 7)
        pos = np.array([100.0, 50.0, 0.0])

        result = wind_func(pos, time=0.0)
        assert_array_almost_equal(result, np.zeros(3))

    def test_shear_wind_negative_altitude(self):
        """Below ground (negative Z), the wind should safely return zero without crashing."""
        base_wind = np.array([5.0, 2.0, 0.0])
        wind_func = create_shear_wind(base_wind, shear_exponent=1 / 7)
        pos = np.array([0.0, 0.0, -5.0])

        result = wind_func(pos, time=0.0)
        assert_array_almost_equal(result, np.zeros(3))

    def test_shear_wind_scaling(self):
        """Test if the power law scaling works correctly at an arbitrary height."""
        base_wind = np.array([10.0, 0.0, 0.0])
        wind_func = create_shear_wind(
            base_wind, shear_exponent=0.5
        )  # Square root scale
        pos = np.array([0.0, 0.0, 40.0])  # (40/10)^0.5 = 4^0.5 = 2.0 scale factor

        expected_wind = np.array([20.0, 0.0, 0.0])
        result = wind_func(pos, time=0.0)
        assert_array_almost_equal(result, expected_wind)


class TestLogarithmicWind:
    def test_log_wind_at_reference_height(self):
        """At z_ref (default 60m), the wind should equal the reference vector."""
        ref_wind = np.array([10.0, -5.0, 0.0])
        wind_func = create_logarithmic_wind(ref_wind, z_ref=60.0, z0=0.03)
        pos = np.array([0.0, 0.0, 60.0])

        result = wind_func(pos, time=0.0)
        assert_array_almost_equal(result, ref_wind)

    def test_log_wind_at_roughness_length(self):
        """At exactly z0 (surface roughness), wind velocity must be zero."""
        ref_wind = np.array([10.0, 0.0, 0.0])
        wind_func = create_logarithmic_wind(ref_wind, z_ref=60.0, z0=0.03)
        pos = np.array([0.0, 0.0, 0.03])

        result = wind_func(pos, time=0.0)
        assert_array_almost_equal(result, np.zeros(3))

    def test_log_wind_below_roughness_length(self):
        """Below z0, the formula would yield negative speeds/errors. It should safely clamp to zero."""
        ref_wind = np.array([10.0, 0.0, 0.0])
        wind_func = create_logarithmic_wind(ref_wind, z_ref=60.0, z0=0.03)
        pos = np.array([0.0, 0.0, 0.01])

        result = wind_func(pos, time=0.0)
        assert_array_almost_equal(result, np.zeros(3))

    def test_log_wind_scaling(self):
        """Test if the logarithmic interpolation evaluates correctly at an arbitrary height."""
        ref_wind = np.array([10.0, 0.0, 0.0])
        z_ref = 60.0
        z0 = 0.03
        wind_func = create_logarithmic_wind(ref_wind, z_ref=z_ref, z0=z0)

        z_test = 10.0
        pos = np.array([0.0, 0.0, z_test])

        # Manually calculate the expected scale
        expected_scale = np.log(z_test / z0) / np.log(z_ref / z0)
        expected_wind = ref_wind * expected_scale

        result = wind_func(pos, time=0.0)
        assert_array_almost_equal(result, expected_wind)

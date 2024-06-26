import unittest

import numpy as np
import pandas as pd
import xarray as xr

import stglib


class TestUtils(unittest.TestCase):
    def test_rotate(self):
        expected = (-np.sqrt(2) / 2, np.sqrt(2) / 2)
        result = stglib.aqd.aqdutils.rotate(0, 1, -45)
        np.testing.assert_almost_equal(expected, result)

        expected = (np.sqrt(2) / 2, np.sqrt(2) / 2)
        result = stglib.aqd.aqdutils.rotate(0, 1, 45)
        np.testing.assert_almost_equal(expected, result)
        print(expected, result)

        expected = (1, 0)
        result = stglib.aqd.aqdutils.rotate(0, 1, 90)
        np.testing.assert_almost_equal(expected, result)

        expected = (np.sqrt(2) / 2, -np.sqrt(2) / 2)
        result = stglib.aqd.aqdutils.rotate(0, 1, 135)
        np.testing.assert_almost_equal(expected, result)

        expected = (0, -1)
        result = stglib.aqd.aqdutils.rotate(0, 1, 180)
        np.testing.assert_almost_equal(expected, result)

        expected = (-1, 0)
        result = stglib.aqd.aqdutils.rotate(0, 1, 270)
        np.testing.assert_almost_equal(expected, result)

    def test_spcon_salinity(self):
        # Test the the roundtrip conversion results in the same as the input value.
        # This is an simplified conversion so we relax the relative tolerance.

        spcon = np.array([10000, 20000])
        salinity = np.array([5.63, 11.92])

        expected = salinity
        result = stglib.utils.salinity_from_spcon(spcon)
        np.testing.assert_allclose(expected, result, rtol=1e-3)

        expected = spcon
        result = stglib.utils.spcon_from_salinity(salinity)
        np.testing.assert_allclose(expected, result, rtol=1e-3)

        expected = spcon
        result = stglib.utils.spcon_from_salinity(
            stglib.utils.salinity_from_spcon(spcon)
        )
        np.testing.assert_allclose(expected, result, rtol=1e-3)


class TestClip(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset()
        self.ds["time"] = xr.DataArray(
            pd.date_range("2000-01-01 00:00", "2000-01-30 00:00", freq="15min"),
            dims="time",
        )
        self.ds.attrs["Deployment_date"] = "2000-01-01 00:05"
        self.ds.attrs["Recovery_date"] = "2000-01-29 23:00"

    def test_clip(self):
        expected = xr.Dataset()
        expected["time"] = xr.DataArray(
            pd.date_range("2000-01-01 00:15", "2000-01-29 23:00", freq="15min"),
            dims="time",
        )
        result = stglib.utils.clip_ds(self.ds)

        np.testing.assert_array_equal(expected["time"], result["time"])

    def test_clip_good_dates(self):
        expected = xr.Dataset()
        expected["time"] = xr.DataArray(
            pd.date_range("2000-01-10 15:45", "2000-01-19 00:00", freq="15min"),
            dims="time",
        )
        self.ds.attrs["good_dates"] = ["2000-01-10 15:41", "2000-01-19 00:00"]
        result = stglib.utils.clip_ds(self.ds)

        np.testing.assert_array_equal(expected["time"], result["time"])

    def test_clip_multiple_good_dates(self):
        expected = xr.Dataset()
        expected["time"] = xr.DataArray(
            np.hstack(
                [
                    pd.date_range("2000-01-10 15:45", "2000-01-19 00:00", freq="15min"),
                    pd.date_range("2000-01-21 15:45", "2000-01-23 00:00", freq="15min"),
                ]
            ),
            dims="time",
        )
        self.ds.attrs["good_dates"] = [
            "2000-01-10 15:41",
            "2000-01-19 00:00",
            "2000-01-21 15:41",
            "2000-01-23 00:00",
        ]
        result = stglib.utils.clip_ds(self.ds)

        np.testing.assert_array_equal(expected["time"], result["time"])

    def test_clip_good_ens(self):
        expected = xr.Dataset()
        expected["time"] = xr.DataArray(
            pd.date_range("2000-01-01 00:15", "2000-01-01 00:30", freq="15min"),
            dims="time",
        )
        self.ds.attrs["good_ens"] = [1, 3]
        result = stglib.utils.clip_ds(self.ds)

        np.testing.assert_array_equal(expected["time"], result["time"])


class TestClock(unittest.TestCase):
    def setUp(self):
        self.ds = xr.Dataset()
        self.ds["time"] = xr.DataArray(
            pd.date_range("2000-01-01 00:00", "2000-01-30 00:00", freq="15min"),
            dims="time",
        )
        self.ds.attrs["AQDAverageInterval"] = 120

    def test_clock_shift(self):
        origtime = self.ds["time"]
        result = stglib.utils.shift_time(
            self.ds, self.ds.attrs["AQDAverageInterval"] / 2
        )
        np.testing.assert_array_equal(
            result["time"],
            origtime
            + np.timedelta64(int(self.ds.attrs["AQDAverageInterval"] / 2), "s"),
        )

    def test_clock_error(self):
        origtime = self.ds["time"]
        self.ds.attrs["ClockError"] = 10
        result = stglib.utils.shift_time(self.ds, 0)
        np.testing.assert_array_equal(
            result["time"], origtime - np.timedelta64(self.ds.attrs["ClockError"], "s")
        )

    def test_clock_drift(self):
        origtime = self.ds["time"]
        self.ds.attrs["ClockDrift"] = 30
        result = stglib.utils.shift_time(self.ds, 0)
        assert result["time"][0] == origtime[0]
        assert result["time"][-1] == origtime[-1] - np.timedelta64(
            self.ds.attrs["ClockDrift"], "s"
        )

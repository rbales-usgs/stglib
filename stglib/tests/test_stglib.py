import unittest
import stglib
import xarray as xr
import numpy as np


class TestTimes(unittest.TestCase):

    def test_epic_time_conversion(self):
        bbvcf = xr.open_dataset(
            ('http://geoport.whoi.edu/thredds/dodsC/silt/usgs/Projects/'
             'stellwagen/CF-1.6/CHINCOTEAGUE/10191Aaqd-a.nc'))
        bbvepic = xr.open_dataset(
            ('http://stellwagen.er.usgs.gov/thredds/dodsC/TSdata/'
             'CHINCOTEAGUE/10191Aaqd-a.nc'), decode_times=False)

        difftime = bbvcf['time'].values - stglib.utils.epic_to_datetime(
           bbvepic['time'], bbvepic['time2']).values

        # values should be equal to at least 1 ms
        assert np.all(difftime <= np.timedelta64(1, 'ms'))


class TestAqd(unittest.TestCase):

    def setUp(self):
        self.T = np.array([[2896, 2896, 0],
                           [-2896, 2896, 0],
                           [-2896, -2896, 5792]])/4096

        self.T_orig = self.T.copy()

        self.vel1 = np.expand_dims([0.23, 0.23, 0.23], 1)
        self.vel2 = np.expand_dims([-0.52, -0.52, -0.52], 1)
        self.vel3 = np.expand_dims([0.12, 0.12, 0.12], 1)

        self.h = np.expand_dims([0, 10, 230], 1)
        self.p = np.expand_dims([0, -5, 5], 1)
        self.r = np.expand_dims([0, 3, -3], 1)

    def test_coord_transform(self):
        # Using Nortek example m-file and compare to Matlab results
        # http://www.nortekusa.com/lib/forum-attachments/coordinate-transformation/view

        u, v, w = stglib.aqd.qaqc.coord_transform(self.vel1,
                                                  self.vel2,
                                                  self.vel3,
                                                  self.h,
                                                  self.p,
                                                  self.r,
                                                  self.T,
                                                  self.T_orig,
                                                  'BEAM')

        result = np.hstack((u, v, w))
        expected = np.array([[0.530273437500000, -0.205039062500000, 0.374726562500000],
                             [0.510589752632478, -0.266778740685713, 0.363012589777355],
                             [-0.144471300248944, 0.544447107731532, 0.382565448778586]])

        np.testing.assert_allclose(result, expected)

    def test_coord_transform_downlooking(self):

        T = self.T.copy()
        T[1, :] = -T[1, :]
        T[2, :] = -T[2, :]

        u, v, w = stglib.aqd.qaqc.coord_transform(self.vel1,
                                                  self.vel2,
                                                  self.vel3,
                                                  self.h,
                                                  self.p,
                                                  self.r,
                                                  T,
                                                  self.T_orig,
                                                  'BEAM')

        result = np.hstack((u, v, w))
        expected = np.array([[-0.530273437500000, -0.205039062500000, -0.374726562500000],
                             [-0.581528098781915, -0.135532612145337, -0.327271926208413],
                             [0.457413978956800, -0.281857021448140, -0.418306112347528]])

        np.testing.assert_allclose(result, expected)

    def test_xyz_to_enu(self):

        u, v, w = stglib.aqd.qaqc.coord_transform(self.vel1,
                                                  self.vel2,
                                                  self.vel3,
                                                  self.h,
                                                  self.p,
                                                  self.r,
                                                  self.T,
                                                  self.T_orig,
                                                  'XYZ')

        result = np.hstack((u, v, w))
        expected = np.array([[0.520000000000000, 0.230000000000000, 0.120000000000000],
                             [0.558771983800901, 0.142329791902643, 0.0722225758067118],
                             [-0.495456501337512, 0.253945766296246, 0.166536491684568]])

        np.testing.assert_allclose(result, expected)

    def test_xyz_to_enu_downlooking(self):

        T = self.T.copy()
        T[1, :] = -T[1, :]
        T[2, :] = -T[2, :]

        u, v, w = stglib.aqd.qaqc.coord_transform(self.vel1,
                                                  self.vel2,
                                                  self.vel3,
                                                  self.h,
                                                  self.p,
                                                  self.r,
                                                  T,
                                                  self.T_orig,
                                                  'XYZ')

        result = np.hstack((u, v, w))
        expected = np.array([[-0.520000000000000, 0.230000000000000, -0.120000000000000],
                             [-0.479197782595360, 0.308957928704944, -0.112314217470635],
                             [0.144416971478138, -0.548502906329893, -0.126444850020645]])

        np.testing.assert_allclose(result, expected)


class TestWavesUtils(unittest.TestCase):

    def test_polar2compass(self):
        polar = [90, 80, 0, -80, -100, -180, 180, 100, 260, 280, 365]
        result = stglib.waves.polar2compass(polar)
        expected = [0, 10, 90, 170, 190, 270, 270, 350, 190, 170, 85]

        np.testing.assert_equal(result, expected)

    def test_to2from(self):
        todir = [0, 45, 90, 135, 180, 225, 270, 315, 360, -180]
        result = stglib.waves.to2from(todir)
        expected = [180, 225, 270, 315, 0, 45, 90, 135, 180, 0]

        np.testing.assert_equal(result, expected)


class TestWaves(unittest.TestCase):
    """Test waves against published Chincoteague data.
    Use the first published burst"""

    def setUp(self):
        self.bbv = xr.open_dataset(
            ('http://stellwagen.er.usgs.gov/thredds/dodsC/TSdata/'
             'CHINCOTEAGUE/10191Aaqdwvs_diwasp-cal.nc'), decode_times=False)

        self.frequency = self.bbv['frequency'].squeeze()
        self.pspec = self.bbv['pspec'].isel(time=0).squeeze()

        self.m0 = stglib.waves.make_moment(
            self.frequency,
            self.pspec,
            0)

        self.m2 = stglib.waves.make_moment(
            self.frequency,
            self.pspec,
            2)

    def test_make_Hs(self):
        result = stglib.waves.make_Hs(self.m0)
        expected = self.bbv['wh_4061'][0].squeeze()

        # can't expect 1e-7 agreement for wave heights, so relax rtol
        np.testing.assert_allclose(result, expected, rtol=1e-2)

    def test_make_Tm(self):
        result = stglib.waves.make_Tm(self.m0, self.m2)
        expected = self.bbv['wp_4060'][0].squeeze()

        np.testing.assert_allclose(result, expected)

    def test_make_Tp(self):
        result = stglib.waves.make_Tp(self.pspec)
        expected = self.bbv['wp_peak'][0].squeeze()

        np.testing.assert_allclose(result, expected)


if __name__ == '__main__':
    unittest.main()
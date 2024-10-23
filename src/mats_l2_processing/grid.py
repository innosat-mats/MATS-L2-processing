from abc import ABC, abstractmethod
import numpy as np
from mats_l1_processing.pointing import pix_deg
from mats_l2_processing.util import get_image, center_grid, cart2sph, sph2cart, geoid_radius, multiprocess
from scipy.spatial.transform import Rotation as R
from skyfield import api as sfapi
from skyfield.framelib import itrs
from mats_utils.geolocation.coordinates import col_heights
import logging


class Grid(ABC):
    def __init__(self):
        # Main grid attributes
        self.edges = None
        self.columns = None
        self.rows = None
        self.ecef_to_local = None
        self.stepsize = None
        self.top_alt = None

        # Main atmospheric quantity attributes
        self.ret_qty = []
        self.aux_qty = []
        self.scales = []
        self.bounds = []

        # Derived attributes
        self.npoints = None
        self.atm_shape = None
        self.centers = None
        self.is_regular_grid = None
        self.lims = None
        self.alt = None

    @abstractmethod
    def calc_los_image(self, image):
        pass

    def verify(self, metadata, processes=1):
        _ = multiprocess(self.cal_los_inage, metadata, self.fwdm_vars, processes, [])
        logging.info("Grid built and verified!")

    def vec2atm(self, vec):
        return [vec[i * self.npoints:(i + 1) * self.npoints].reshape(self.atm_shape[1:]) * self.scales[i]
                for i in range(len(self.ret_qty))]

    def atm2vec(self, atm):
        return np.concatenate([atm[i].flatten() / self.scales[i] for i in len(atm)])

    def _calc_lims(self):
        lims = np.zeros((2, len(self.ret_qty) * self.npoints))
        for i in range(len(self.ret_qty)):
            for j in range(2):
                lims[j, i * self.npoints: (i + 1) * self.npoints] = self.bounds[i][j] / self.scales[i]
        return lims

    def reset_invalid(self, vec):
        assert not np.isnan(vec).any(), "Nan's found in solution! Abort!"
        res = np.maximum(vec, self.lims[0, :])
        res = np.minimum(res, self.lims[1, :])
        return res

    def _calc_tp_heights_image(self, image):
        res = np.empty((len(self.columns), len(self.rows)))
        for c, col in enumerate(self.columns):
            cs = col_heights(image, col, 10, spline=True)
            res[c, :] = np.array(cs(self.rows))
        return res

    def _calc_tp_heights(self, metadata, processes):
        return multiprocess(self._calc_tp_heights_image, metadata, self.TP_heights_vars, processes, [], stack=True)


class Rad_along_corotating_3D_grid(Grid):
    def __init__(self, metadata, conf, const, columns, rows, grid_proto, processes=1, verify=False):
        super().__init__()

        # Initialize basic grid
        self.columns = columns
        self.rows = rows
        self.stepsize = conf.STEP_SIZE
        self.top_alt = conf.TOP_ALT
        self.timescale = sfapi.load.timescale()
        self.ecef_to_local = self.generate_local_transform(metadata)
        self.edges = self._generate_grid(metadata, grid_proto)

        # Set derived attributes
        self.centers = [center_grid(g) for g in self.edges]
        self.alt, self.lat, self.lon = self.get_alt_lat_lon()
        dims = np.array([len(x) for x in self.centers])
        self.npoints = np.prod(dims)
        self.atm_shape = [len(self.ret_qty), *dims]

        self.is_regular_grid = True
        for axis_edges in self.edges:
            widths = np.diff(axis_edges)
            if not all(np.abs(widths - np.mean(widths)) < 0.001 * np.abs(np.mean(widths))):
                self.is_regular_grid = False
                break

        # Set atmospheric quantity data
        self.ret_qty = conf.RET_QTY
        self.aux_qty = conf.AUX_QTY
        self.scales = conf.SCALES
        self.bounds = conf.BOUNDS
        self.lims = self._calc_lims()

        # Calculate tangent point heights
        self.TP_heights_vars = const.TP_VARS
        self.TP_heights = self._calc_tp_heights(metadata, processes)

        # Verify that all LOS fit into grid
        if verify:
            self.fwdm_vars = const.NEEDED_VARS
            self.verify(metadata, processes=processes)

    def generate_local_transform(self, data):
        """Calculate the transform from ecef to the local coordinate system.

        Detailed description

        Args:
            df (pandas dataframe): data.

        Returns:
            ecef_to_local
        """
        first = 0
        mid = int((data["size"] - 1) / 2)
        last = data["size"] - 1

        # eci_to_ecef = eci_to_ecef_transform(data['EXPDate'][mid])
        eci_to_ecef = R.from_matrix(itrs.rotation_at(self.timescale.from_datetime(data['EXPDate'][mid])))

        posecef_first = eci_to_ecef.apply(data["afsTangentPointECI"][first, :]).astype('float32')
        posecef_mid = eci_to_ecef.apply(data["afsTangentPointECI"][mid, :]).astype('float32')
        posecef_last = eci_to_ecef.apply(data["afsTangentPointECI"][last, :]).astype('float32')

        observation_normal = np.cross(posecef_first, posecef_last)
        observation_normal = observation_normal / np.linalg.norm(observation_normal)  # normalize vector

        posecef_mid_unit = posecef_mid / np.linalg.norm(posecef_mid)  # unit vector for central position
        ecef_to_local = R.align_vectors([[1, 0, 0], [0, 1, 0]], [posecef_mid_unit, observation_normal])[0]

        return ecef_to_local

    def _generate_grid(self, data, grid_proto):
        # columns, rows, ecef_to_local, top_alt, stepsize, timescale
        lims = self.grid_limits(data)
        if grid_proto is None:
            grid_proto = [lims[i][2] for i in range(3)]
        result = []
        for i in range(3):
            if (type(grid_proto[i]) is int) or (type(grid_proto[i]) is float):
                grid_len = int(grid_proto[i])
                assert grid_len > 0, "Malformed grid_spec parameter!"
                result.append(np.linspace(lims[i][0], lims[i][1], grid_len))
            elif isinstance(grid_proto[i], np.ndarray):
                assert len(grid_proto[i].shape) == 1
                result.append(grid_proto[i][np.logical_and(grid_proto[i] > lims[i][0], grid_proto[i] < lims[i][1])])
            else:
                raise ValueError(f"Malformed grid_spec parameter: {type(grid_proto[i])}")
        return result

    def grid_limits(self, data):
        first = 0
        mid = int((data["size"] - 1) / 2)
        last = data["size"] - 1

        mid_date = data['EXPDate'][mid]
        current_ts = self.timescale.from_datetime(mid_date)
        localR = np.linalg.norm(sfapi.wgs84.latlon(data["TPlat"][mid], data["TPlon"][mid],
                                                   elevation_m=0).at(current_ts).position.m)

        # Rotation between satellite pointing and channel pointing
        rot_sat_channel = R.from_quat(data['qprime'][mid, :])

        q = data['afsAttitudeState'][mid, :]  # Satellite pointing in ECI
        rot_sat_eci = R.from_quat(np.roll(q, -1))  # Rotation matrix for q (should go straight to ecef?)

        eci_to_ecef = R.from_matrix(itrs.rotation_at(current_ts))
        satpos_eci = data['afsGnssStateJ2000'][mid, 0:3]
        satpos_ecef = eci_to_ecef.apply(satpos_eci)

        # change to do only used columns and rows
        if len(self.columns) == 1:
            mid_col = self.columns[0]
        else:
            left_col = self.columns[0]
            right_col = self.columns[-1]
            mid_col = int((left_col + right_col) / 2)
            # mid_col = int(data["NCOL"][0] / 2) - 1
            # right_col = data["NCOL"][0] - 1

        irow = self.rows[0]
        poslocal_sph = []

        get_los_vars = ['CCDSEL', 'NCSKIP', 'NCBINCCDColumns', 'NRSKIP', 'NRBIN', 'NCOL', 'EXPDate', 'TPlat', 'TPlon']
        # Which localR to use in get_step_local_grid?
        for idx in [first, mid, last]:
            image = get_image(data, idx, get_los_vars)
            los_ecef = self.get_los_ecef(image, mid_col, irow, rot_sat_channel, rot_sat_eci, eci_to_ecef)
            poslocal_sph.append(self._get_steps_in_local_grid(image, satpos_ecef, los_ecef, localR=None))

        if len(self.columns) > 1:
            for col in [left_col, right_col]:
                for idx in [first, mid, last]:
                    image = get_image(data, idx, get_los_vars)
                    los_ecef = self.get_los_ecef(image, col, irow, rot_sat_channel, rot_sat_eci, eci_to_ecef)
                    poslocal_sph.append(self._get_steps_in_local_grid(image, satpos_ecef, los_ecef, localR=localR))

        poslocal_sph = np.vstack(poslocal_sph)
        # max_rad = poslocal_sph[:, 0].max()
        min_rad = poslocal_sph[:, 0].min()
        max_along = poslocal_sph[:, 2].max()
        min_along = poslocal_sph[:, 2].min()
        max_across = poslocal_sph[:, 1].max()
        min_across = poslocal_sph[:, 1].min()
        if len(self.columns) < 2:
            max_across = max_across + 0.2
            min_across = min_across - 0.2

        nalt = int(len(self.rows / 2)) + 2
        nacross = int(len(self.columns / 2)) + 2
        nalong = int(data["size"] / 2) + 2

        if (nacross - 2) < 2:
            nacross = 2
        if (nalong - 2) < 2:
            nalong = 2

        return (min_rad - 10e3, localR + self.top_alt + 10e3, nalt), (min_across - 0.1, max_across + 0.1, nacross), \
            (min_along - 0.6, max_along + 0.6, nalong)

    def _get_steps_in_local_grid(self, image, satpos_ecef, los_ecef, localR=None):
        if localR is None:
            date = image['EXPDate']
            current_ts = self.timescale.from_datetime(date)
            localR = np.linalg.norm(sfapi.wgs84.latlon(image["TPlat"], image["TPlon"],
                                                       elevation_m=0).at(current_ts).position.m)

        s_steps = self._generate_steps(localR, satpos_ecef, los_ecef)

        posecef = (np.expand_dims(satpos_ecef, axis=0).T +
                   s_steps * np.expand_dims(los_ecef, axis=0).T).astype('float32')
        poslocal = self.ecef_to_local.apply(posecef.T)  # convert to local (for middle alongtrack measurement)
        poslocal_sph = cart2sph(poslocal)
        poslocal_sph = np.array(poslocal_sph).T
        return poslocal_sph

    def _generate_steps(self, localR, satpos, losvec):
        distance_top_of_atmosphere = self._find_top_of_atmosphere(localR, satpos, losvec)
        steps = np.arange(distance_top_of_atmosphere[0], distance_top_of_atmosphere[1], self.stepsize)
        return steps

    def _find_top_of_atmosphere(self, localR, satpos, losvec):
        sat_radial_pos = np.linalg.norm(satpos)
        los_zenith_angle = np.arccos(np.dot(satpos, losvec) / sat_radial_pos)
        #  solving quadratic equation to find distance start and end of atmosphere
        b = 2 * sat_radial_pos * np.cos(los_zenith_angle)
        root = np.sqrt(b**2 + 4 * ((self.top_alt + localR)**2 - sat_radial_pos**2))
        distance_top_1 = (-b - root) / 2
        distance_top_2 = (-b + root) / 2
        return [distance_top_1, distance_top_2]

    @staticmethod
    def get_los_ecef(image, icol, irow, rot_sat_channel, rot_sat_eci, eci_to_ecef):
        # Angles for single pixel
        x, y = pix_deg(image, icol, irow)
        # Rot matrix between pixel pointing and channels pointing
        rotation_channel_pix = R.from_euler('xyz', [0, y, x], degrees=True).apply([1, 0, 0])
        los_eci = np.array(rot_sat_eci.apply(rot_sat_channel.apply(rotation_channel_pix)))
        los_ecef = eci_to_ecef.apply(los_eci)
        return los_ecef

    def calc_los_image(self, image):
        # tic = time.time()
        #   columns, rows, ecef_to_local, timescale, top_alt, stepsize = common_args
        rot_sat_channel = R.from_quat(image['qprime'])  # Rotation between satellite pointing and channel pointing
        q = image['afsAttitudeState']  # Satellite pointing in ECI
        rot_sat_eci = R.from_quat(np.roll(q, -1))  # Rotation matrix for q (should go straight to ecef?)

        current_ts = self.timescale.from_datetime(image["EXPDate"])
        localR = np.linalg.norm(sfapi.wgs84.latlon(image["TPlat"], image["TPlon"],
                                                   elevation_m=0).at(current_ts).position.m)
        eci_to_ecef = R.from_matrix(itrs.rotation_at(current_ts))
        satpos_eci = image['afsGnssStateJ2000'][0:3]
        satpos_ecef = eci_to_ecef.apply(satpos_eci)
        los = []
        for column in self.columns:
            los_col = []
            for row in self.rows:
                los_ecef = self.get_los_ecef(image, column, row, rot_sat_channel, rot_sat_eci, eci_to_ecef)
                posecef_i_sph = self._get_steps_in_local_grid(image, satpos_ecef, los_ecef, localR=localR)
                los_col.append(posecef_i_sph)
            los.append(los_col.copy())
        # toc = time.time()
        # logging.log(15, f"LOS calculated for image {image['num_image']} in {toc - tic:.3f} s.")
        return los

    def get_alt_lat_lon(self):
        rr, acrr, alongg = np.meshgrid(*self.centers, indexing="ij")
        lxx, lyy, lzz = sph2cart(rr, acrr, alongg)
        glgrid = self.ecef_to_local.inv().apply(np.dstack((lxx.flatten(), lyy.flatten(), lzz.flatten()))[0, :, :])
        latt = np.arcsin(glgrid[:, 2].reshape(rr.shape) / rr)
        altt = rr - geoid_radius(latt)
        latt = np.rad2deg(latt)[0, :, :]
        lonn = np.rad2deg(np.arctan2(glgrid[:, 1], glgrid[:, 0]).reshape(rr.shape))[0, :, :]
        return altt, latt, lonn

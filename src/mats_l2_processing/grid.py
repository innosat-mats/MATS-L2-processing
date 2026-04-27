from abc import ABC, abstractmethod
import numpy as np
# from mats_l1_processing.pointing import pix_deg
from mats_l2_processing.pointing import Pointing, faster_heights
from mats_l2_processing.util import get_image, center_grid, cart2sph, sph2cart, \
    geoid_radius, multiprocess, make_grid_proto, grid_from_proto, unnest
from mats_l2_processing.io import write_gen_ncdf, append_gen_ncdf
from mats_l2_processing.obs import get_row_col

from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interpn
from skyfield import api as sfapi
from skyfield.framelib import itrs
# from mats_utils.geolocation.coordinates import col_heights
import logging


class Grid(ABC):
    def __init__(self, all_metadata, conf, const, processes):
        metadata = all_metadata[0]

        # Main grid attributes
        self.points = None
        self.rows, self.columns = get_row_col(conf, metadata)

        # self.ecef_to_local = None
        self.stepsize = conf.STEP_SIZE
        self.top_alt = conf.TOP_ALT
        self.timescale = sfapi.load.timescale()
        self.pointing = Pointing(all_metadata, conf, const)

        # Constants
        self.ref_alt = 80e3
        self.fwdm_vars = const.NEEDED_DATA

        # Main atmospheric quantity attributes
        self.ret_qty = conf.RET_QTY
        self.aux_qty = conf.AUX_QTY
        self.scales = conf.SCALES
        self.bounds = conf.BOUNDS

        # Set geometric parameters of observations
        self.img_time = metadata["time_s"]
        self.valid_time = np.mean(metadata["time_s"])

        self.ncpar = const.ncpar

        self.alt = None

    def _set_derived(self, metadata, processes, combine_images, verify):
        # self.centers = [center_grid(g) for g in self.edges]
        dims = np.array([len(x) for x in self.points])
        self.npoints = np.prod(dims)

        self.combine_images = combine_images
        if combine_images:
            self.atm_shape = [len(self.ret_qty), *dims]
            self.all_points = self.npoints
        else:
            self.atm_shape = [len(self.ret_qty), len(self.img_time), *dims]
            self.all_points = self.npoints * len(self.img_time)

        self.lims = self._calc_lims()
        # Obs self.TP_heights = self._calc_tp_heights(metadata, processes)

        self.is_regular_grid = True
        for axis_points in self.points:
            widths = np.diff(axis_points)
            if not all(np.abs(widths - np.mean(widths)) < 0.001 * np.abs(np.mean(widths))):
                self.is_regular_grid = False
                break

        if verify:
            self.verify(metadata, processes)

    def calc_los_image(self, image, common_args):
        # tic = time.time()
        rot_sat_channel = R.from_quat(image['qprime'])  # Rotation between satellite pointing and channel pointing
        q = image['afsAttitudeState']  # Satellite pointing in ECI
        rot_sat_eci = R.from_quat(np.roll(q, -1))  # Rotation matrix for q (should go straight to ecef?)

        current_ts = self.timescale.from_datetime(image["time"])
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
                posecef_own_grid = self._get_steps_in_own_grid(self._get_steps_in_ecef(image, satpos_ecef,
                                                                                       los_ecef, localR=localR))
                los_col.append(posecef_own_grid)
            los.append(los_col.copy())
        # toc = time.time()
        # logging.log(15, f"LOS calculated for image {image['num_image']} in {toc - tic:.3f} s.")
        return los

    def get_los_ecef(self, image, icol, irow, rot_sat_channel, rot_sat_eci, eci_to_ecef):
        # Angles for single pixel
        x, y = self.pointing.pix_deg(image, icol, irow)
        # Rot matrix between pixel pointing and channels pointing
        rotation_channel_pix = R.from_euler('xyz', [0, y, x], degrees=True).apply([1, 0, 0])
        los_eci = np.array(rot_sat_eci.apply(rot_sat_channel.apply(rotation_channel_pix)))
        los_ecef = eci_to_ecef.apply(los_eci)
        return los_ecef

    def _get_steps_in_ecef(self, image, satpos_ecef, los_ecef, localR=None):
        if localR is None:
            date = image['time']
            current_ts = self.timescale.from_datetime(date)
            localR = np.linalg.norm(sfapi.wgs84.latlon(image["TPlat"], image["TPlon"],
                                                       elevation_m=0).at(current_ts).position.m)

        s_steps = self._generate_steps(localR, satpos_ecef, los_ecef)
        return (np.expand_dims(satpos_ecef, axis=0).T +
                s_steps * np.expand_dims(los_ecef, axis=0).T).astype('float32').T

    @abstractmethod
    def _get_steps_in_own_grid(self, steps_in_ecef):
        pass

    @abstractmethod
    def write_grid_ncdf(self, fname, attributes=[]):
        pass

    @abstractmethod
    def write_atm_ncdf(self, fname, atm, atm_suffix="", atm_suffix_long=""):
        pass

    # Obs @abstractmethod
    #     def write_obs_ncdf(self, fname, obs, channels, obs_suffix="", obs_suffix_long=""):
    #     pass

    def verify(self, metadata, processes=1):
        minmax = np.stack([np.full((2,), points[int(len(points) / 2.0)]) for points in self.points])
        gridl = np.stack([np.array([points[0], points[-1]]) for points in self.points])

        ndims = len(self.points)

        pos = multiprocess(self.calc_los_image, metadata, self.fwdm_vars, processes, [])
        for los in unnest(pos):
            minmax[:, 0] = [np.minimum(minmax[i, 0], np.min(los[:, i])) for i in range(ndims)]
            minmax[:, 1] = [np.maximum(minmax[i, 1], np.max(los[:, i])) for i in range(ndims)]

        minmax, gridl = [(arr - self.offsets[:, np.newaxis]) / self.scalings[:, np.newaxis] for arr in [minmax, gridl]]
        logging.info("Grid verification:")
        for i in range(ndims):
            logging.info(f"Dimension {i}: grid {gridl[i, 0]:.3e} to {gridl[i, 1]:.3e}," +
                         f" los {minmax[i, 0]:.3e} to {minmax[i, 1]:.3e}.")

        OK = all([np.logical_and(minmax[i, 0] > gridl[i, 0], minmax[i, 1] < gridl[i, 1]) for i in range(ndims)])
        if OK:
            logging.info("Grid built and verified!")
        else:
            raise ValueError("LOS outside grid found! Abort")

    def vec2atm(self, vec):
        return [vec[i * self.all_points:(i + 1) * self.all_points].reshape(self.atm_shape[1:]) * self.scales[i]
                for i in range(len(self.ret_qty))]

    def atm2vec(self, atm):
        return np.concatenate([atm[i].flatten() / self.scales[i] for i in range(len(atm))])

    def get_imvec(self, vec, num_im):
        return np.concatenate([vec[self.all_points * j + self.npoints * num_im:
                                   self.all_points * j + self.npoints * (num_im + 1)] for j in range(self.atm_shape[0])])

    def imvecs2vec(self, imvecs):
        qty_vecs = [np.concatenate([imvec[self.npoints * j:self.npoints * (j + 1)] for imvec in imvecs])
                    for j in range(self.atm_shape[0])]
        return np.concatenate(qty_vecs)

    def _calc_lims(self):
        lims = np.zeros((2, len(self.ret_qty) * self.all_points))
        for i in range(len(self.ret_qty)):
            for j in range(2):
                lims[j, i * self.all_points: (i + 1) * self.all_points] = self.bounds[i][j] / self.scales[i]
        return lims

    def reset_invalid(self, vec):
        assert not np.isnan(vec).any(), "Nan's found in solution! Abort!"
        res = np.maximum(vec, self.lims[0, :])
        res = np.minimum(res, self.lims[1, :])
        return res

    @staticmethod
    def generate_local_transform(data, timescale):
        # Calculate the transform from ecef to the local coordinate system.

        first = 0
        mid = int((data["size"] - 1) / 2)
        last = data["size"] - 1
        center_row, center_col = [int(np.round(data[var][0] / 2.0)) for var in ["NROW", "NCOL"]]

        # posecef_first, posecef_mid, posecef_last = [data["afsTangentPointECEF"][idx, :3] for idx in [first, mid, last]]
        posecef_first, posecef_mid, posecef_last = [np.array([data[f"TPECEF{c}"][idx, center_row, center_col]
                                                              for c in ['x', 'y', 'z']]) for idx in [first, mid, last]]
        observation_normal = np.cross(posecef_first, posecef_last)
        observation_normal = observation_normal / np.linalg.norm(observation_normal)  # normalize vector

        posecef_mid_unit = posecef_mid / np.linalg.norm(posecef_mid)  # unit vector for central position
        ecef_to_local = R.align_vectors([[1, 0, 0], [0, 1, 0]], [posecef_mid_unit, observation_normal])[0]
        return ecef_to_local

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

    def _set_points(self, name, conf, const, lims, scaling=1.0, offset=0.0):
        spec = getattr(conf, name, None)
        if (type(spec) is not dict) or ("method" not in spec.keys()):
            raise ValueError(f"Configuration variable {name} must be a dictionary with a 'method' key!")

        use_lims = not spec["ignore_lims"] if "ignore_lims" in spec.keys() else True

        if spec["method"] == "fill_lims":
            assert use_lims, "Cannot ignore limits with 'fill_lims' grid specification method!"
            coord = np.linspace(lims[0], lims[1], int(spec["num_points"]))
        elif spec["method"] == "array":
            coord = spec["array"]
        elif spec["method"] == "intervals":
            assert type(spec.get("intervals")) is list and all([len(x) == 3 for x in spec["intervals"]]), \
                "Malformed 'intervals' spec for conf. variable {name}!"
            coord = np.concatenate([np.arange(*intv) for intv in spec["intervals"]])
        elif spec["method"] == "auto_along":
            nlc_pad = 2 * spec.get("nlc_img_pad", 0.0) * np.mean(np.diff(self.img_time))
            tomo_len = const.sat_speed_approx * (self.img_time[-1] - self.img_time[0] - nlc_pad)
            full_grid_hlen = tomo_len / 2 + spec["full_grid_pad"]
            hgrid = np.arange(0, full_grid_hlen + spec["full_grid_step"], spec["full_grid_step"])
            hgrid = np.concatenate([hgrid, hgrid[-1] + spec["grid_tail"]])
            coord = np.sort(np.unique(np.concatenate([-1.0 * hgrid, hgrid])))
        else:
            raise ValueError(f"Invalid 'method' value in configuration variable {name}!")

        coord = scaling * coord + offset
        if use_lims:
            coord = coord[np.logical_and(coord > lims[0], coord < lims[1])]
        return coord

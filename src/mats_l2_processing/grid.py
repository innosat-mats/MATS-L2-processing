from abc import ABC, abstractmethod
import numpy as np
# from mats_l1_processing.pointing import pix_deg
from mats_l2_processing.pointing import Pointing, faster_heights
from mats_l2_processing.util import get_image, center_grid, cart2sph, sph2cart, \
    geoid_radius, multiprocess, make_grid_proto, grid_from_proto
from mats_l2_processing.io import write_gen_ncdf, append_gen_ncdf
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interpn
from skyfield import api as sfapi
from skyfield.framelib import itrs
# from mats_utils.geolocation.coordinates import col_heights
import logging


class Grid(ABC):
    def __init__(self, metadata, conf, const, processes):
        # Main grid attributes
        self.edges = None
        self.columns = None
        self.rows = None
        # self.ecef_to_local = None
        self.stepsize = conf.STEP_SIZE
        self.top_alt = conf.TOP_ALT
        self.timescale = sfapi.load.timescale()
        self.pointing = Pointing(metadata, conf, const)

        # Constants
        self.ref_alt = 80e3

        # Main atmospheric quantity attributes
        self.ret_qty = conf.RET_QTY
        self.aux_qty = conf.AUX_QTY
        self.scales = conf.SCALES
        self.bounds = conf.BOUNDS

        # Set geometric parameters of observations
        self.TP_heights_vars = const.TP_VARS
        self.img_time = metadata["EXPDate_s"]
        self.valid_time = np.mean(self.img_time)

        self.ncpar = const.ncpar

        self.alt = None

    def _set_derived(self, metadata, processes, combine_images, verify):
        self.centers = [center_grid(g) for g in self.edges]
        dims = np.array([len(x) for x in self.centers])
        self.npoints = np.prod(dims)

        self.combine_images = combine_images
        if combine_images:
            self.atm_shape = [len(self.ret_qty), *dims]
            self.all_points = self.npoints
        else:
            self.atm_shape = [len(self.ret_qty), len(self.img_time), *dims]
            self.all_points = self.npoints * len(self.img_time)

        self.lims = self._calc_lims()
        self.TP_heights = self._calc_tp_heights(metadata, processes)

        self.is_regular_grid = True
        for axis_edges in self.edges:
            widths = np.diff(axis_edges)
            if not all(np.abs(widths - np.mean(widths)) < 0.001 * np.abs(np.mean(widths))):
                self.is_regular_grid = False
                break

    def calc_los_image(self, image, common_args):
        # tic = time.time()
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
            date = image['EXPDate']
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

    @abstractmethod
    def write_obs_ncdf(self, fname, obs, channels, obs_suffix="", obs_suffix_long=""):
        pass

    def verify(self, metadata, processes=1):
        _ = multiprocess(self.calc_los_image, metadata, self.fwdm_vars, processes, [])
        logging.info("Grid built and verified!")

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

    def _calc_tp_heights_image(self, image, common_args):
        # res = np.empty((len(self.columns), len(self.rows)))
        # for c, col in enumerate(self.columns):
        #     cs = col_heights(image, col, 10, spline=True)
        #     res[c, :] = np.array(cs(self.rows))
        #heights = faster_heights(image, self.pointing)
        #breakpoint()
        return faster_heights(image, self.pointing, cols=self.columns, rows=self.rows).T

    def _calc_tp_heights(self, metadata, processes):
        return multiprocess(self._calc_tp_heights_image, metadata, self.TP_heights_vars, processes, [], stack=True)

    @staticmethod
    def generate_local_transform(data, timescale):
        # Calculate the transform from ecef to the local coordinate system.

        first = 0
        mid = int((data["size"] - 1) / 2)
        last = data["size"] - 1

        posecef_first, posecef_mid, posecef_last = [data["afsTangentPointECEF"][idx, :3] for idx in [first, mid, last]]

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


class Rad_along_3D_grid(Grid):
    def __init__(self, metadata, conf, const, processes=1, verify=False):
        super().__init__(metadata, conf, const, processes)

        # Initialize basic grid
        row_range = (0, metadata["NROW"][0]) if conf.ROW_RANGE[0] < 0 else conf.ROW_RANGE
        self.columns, self.rows = [np.arange(r[0], r[1], 1) for r in [conf.COL_RANGE, row_range]]

        self.ecef_to_local = self.generate_local_transform(metadata, self.timescale)
        self.local_geoid_radius = geoid_radius(np.deg2rad(np.mean(metadata["TPlat"])))
        # self.edges = self._generate_grid(metadata, conf)
        lims = self.grid_limits(metadata)
        grid_proto = [make_grid_proto(conf.ALT_GRID, scaling=1e3, offset=self.local_geoid_radius),
                      make_grid_proto(conf.ACROSS_GRID, scaling=1e3 / self.local_geoid_radius),
                      make_grid_proto(conf.ALONG_GRID, scaling=1e3 / self.local_geoid_radius)]
        self.edges = [grid_from_proto(p, l) for p, l in zip(grid_proto, lims)]
        # Set derived attributes
        self._set_derived(metadata, processes, True, verify)

        # Set geolocation attributes
        self.alt, self.lat, self.lon = self.get_alt_lat_lon()

    def _generate_grid(self, data, conf):
        lims = self.grid_limits(data)
        grid_proto = [make_grid_proto(conf.ALT_GRID, scaling=1e3, offset=self.local_geoid_radius),
                      make_grid_proto(conf.ACROSS_GRID, scaling=1e3 / self.local_geoid_radius),
                      make_grid_proto(conf.ALONG_GRID, scaling=1e3 / self.local_geoid_radius)]
        # if grid_proto is None:
        #     grid_proto = [lims[i][2] for i in range(3)]
        #result = []
        #for i in range(3):
        #    if (type(grid_proto[i]) is int) or (type(grid_proto[i]) is float):
        #        grid_len = int(grid_proto[i])
        #        assert grid_len > 0, "Malformed grid_spec parameter!"
        #        result.append(np.linspace(lims[i][0], lims[i][1], grid_len))
        #    elif isinstance(grid_proto[i], np.ndarray):
        #        assert len(grid_proto[i].shape) == 1
        #        result.append(grid_proto[i][np.logical_and(grid_proto[i] > lims[i][0], grid_proto[i] < lims[i][1])])
        #    else:
        #        raise ValueError(f"Malformed grid_spec parameter: {type(grid_proto[i])}")
        return [grid_from_proto(p, l) for p, l in zip(lims, grid_proto)]

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

        irow = self.rows[0]
        poslocal_sph = []

        get_los_vars = ['channel', 'NCSKIP', 'NCBINCCDColumns', 'NRSKIP', 'NRBIN', 'NCOL', 'EXPDate', 'TPlat', 'TPlon']
        # Which localR to use in get_step_local_grid?
        for idx in [first, mid, last]:
            image = get_image(data, idx, get_los_vars)
            los_ecef = self.get_los_ecef(image, mid_col, irow, rot_sat_channel, rot_sat_eci, eci_to_ecef)
            # poslocal_sph.append(self._get_steps_in_local_grid(image, satpos_ecef, los_ecef, localR=None))
            steps_in_ecef = self._get_steps_in_ecef(image, satpos_ecef, los_ecef, localR=None)
            poslocal_sph.append(self._get_steps_in_own_grid(steps_in_ecef))

        if len(self.columns) > 1:
            for col in [left_col, right_col]:
                for idx in [first, mid, last]:
                    image = get_image(data, idx, get_los_vars)
                    los_ecef = self.get_los_ecef(image, col, irow, rot_sat_channel, rot_sat_eci, eci_to_ecef)
                    steps_in_ecef = self._get_steps_in_ecef(image, satpos_ecef, los_ecef, localR=localR)
                    poslocal_sph.append(self._get_steps_in_own_grid(steps_in_ecef))

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

    # def _get_steps_in_local_grid(self, image, satpos_ecef, los_ecef, localR=None):
    #    if localR is None:
    #        date = image['EXPDate']
    #        current_ts = self.timescale.from_datetime(date)
    #        localR = np.linalg.norm(sfapi.wgs84.latlon(image["TPlat"], image["TPlon"],
    #                                                   elevation_m=0).at(current_ts).position.m)
    #
    #    s_steps = self._generate_steps(localR, satpos_ecef, los_ecef)
    #
    #    posecef = (np.expand_dims(satpos_ecef, axis=0).T +
    #               s_steps * np.expand_dims(los_ecef, axis=0).T).astype('float32')
    #    poslocal = self.ecef_to_local.apply(posecef.T)  # convert to local (for middle alongtrack measurement)
    #    poslocal_sph = cart2sph(poslocal)
    #    poslocal_sph = np.array(poslocal_sph).T
    #    return poslocal_sph

    def _get_steps_in_own_grid(self, steps_in_ecef):
        poslocal = self.ecef_to_local.apply(steps_in_ecef)
        return np.array(cart2sph(poslocal)).T

    def get_alt_lat_lon(self):
        rr, acrr, alongg = np.meshgrid(*self.centers, indexing="ij")
        lxx, lyy, lzz = sph2cart(rr, acrr, alongg)
        glgrid = self.ecef_to_local.inv().apply(np.dstack((lxx.flatten(), lyy.flatten(), lzz.flatten()))[0, :, :])
        latt = np.arcsin(glgrid[:, 2].reshape(rr.shape) / rr)
        altt = rr - geoid_radius(latt)
        latt = np.rad2deg(latt)[0, :, :]
        lonn = np.rad2deg(np.arctan2(glgrid[:, 1], glgrid[:, 0]).reshape(rr.shape))[0, :, :]
        return altt, latt, lonn

    def write_grid_ncdf(self, fname, attributes={}):
        # Define dimensions
        eff_radius = self.local_geoid_radius + np.mean(self.centers[0])
        dim_pars = {"radial_coord": ("Distance from the center of the Earth", "meter", self.centers[0]),
                    "acrosstrack_coord": ("Horizontal coordinate in the direction perpendicular to the orbital plane",
                                          "meter", self.centers[1] * eff_radius),
                    "alongtrack_coord": ("Horizontal coordinate in the direction of the satellite track", "meter",
                                         self.centers[2] * eff_radius),
                    "img_time": ("Acquisition time of individual MATS images", "Seconds since 2000.01.01 00:00 UTC",
                                 self.img_time),
                    "img_col": ("Column of (coadded) pixels in the image", None, self.columns),
                    "img_row": ("Row of (coadded) pixels in the image", None, self.rows),
                    "time": ("Valid time of L2 data", "Seconds since 2000.01.01 00:00 UTC",
                             self.valid_time * np.ones(1))}
        dims_4D = ("time", "radial_coord", "acrosstrack_coord", "alongtrack_coord")
        dims_2D = ("acrosstrack_coord", "alongtrack_coord")

        # Define coordinate variables
        ncvars = {"altitude": ("Altitude", "meter", self.alt, dims_4D),
                  "longitude": ("Longitude", "degree_east", self.lon, dims_2D),
                  "latitude": ("Latitude", "degree_north", self.lat, dims_2D),
                  "TPheight": ("Tangent point height", "meter", self.TP_heights, ("img_time", "img_col", "img_row"))}

        write_gen_ncdf(fname, dim_pars, ncvars, attributes)

    def write_atm_ncdf(self, fname, atm, atm_suffix="", atm_suffix_long=""):
        ncvars = {}
        dims = ("time", "radial_coord", "acrosstrack_coord", "alongtrack_coord")
        for i, qty in enumerate(self.ret_qty):
            ncvars[f"{qty}{atm_suffix}"] = (f"{self.ncpar[qty][0]}{atm_suffix_long}", self.ncpar[qty][1], atm[i],
                                            dims)
        append_gen_ncdf(fname, ncvars)

    def write_obs_ncdf(self, fname, obs, channels, obs_suffix="", obs_suffix_long="", attributes={}):
        ncvars = {}
        dims = ("img_time", "img_col", "img_row")
        for i, chn in enumerate(channels):
            ncvars[f"{chn}{obs_suffix}"] = (f"{self.ncpar[chn][0]}{obs_suffix_long}", self.ncpar[chn][1],
                                            obs[i, :, :, :], dims)
        append_gen_ncdf(fname, ncvars, attributes=attributes)

    def interpolate_from_3D(self, ext_coords, ext_data):
        # Prepare coordinates
        ret_coords = np.zeros(list(self.alt.shape) + [3])
        ret_coords[..., 0] = self.alt
        ret_coords[..., 1] = self.lat[np.newaxis, :, :]
        ret_coords[..., 2] = self.lon[np.newaxis, :, :]

        # Interpolate
        res = []
        for extd in ext_data:
            res.append(interpn(ext_coords, extd, ret_coords))
        return res

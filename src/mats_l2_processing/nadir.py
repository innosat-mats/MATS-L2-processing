from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
from itertools import product, chain
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation as R
from multiprocessing import Pool
from types import SimpleNamespace

from mats_l2_processing.io import write_gen_ncdf
from mats_l2_processing.util import wgs842ecef, ecef2wgs84, cart2sph, sph2cart, geoid_radius


class Nadir_grid(ABC):
    def __init__(self, nadir_data, conf, const, processes, mask=None):

        # Main grid attributes
        # self.rows, self.columns = get_row_col(conf, metadata)
        # self.ref_alt = 80e3
        ims_shape = nadir_data["img"].shape
        self.num_simages = ims_shape[0]
        self.im_shape = (ims_shape[2], ims_shape[1])
        self.num_obs = np.prod(ims_shape)
        self.img_time = nadir_data["time"]
        # self.im = nadir_data["im"][:]
        self.scales = [conf.RAD_SCALE]
        self.combine_images = True
        self.ret_qty = ["Radiance"]

        if mask is None:
            mask = np.zeros(self.im_shape, dtype=bool)
        else:
            assert mask.shape == self.im_shape, "Incorrectly shaped nadir mask! "
        self.valid_pix = np.logical_not(mask)
        self.valid_obs = np.broadcast_to(self.valid_pix[np.newaxis, :, :], (self.num_simages,) + self.im_shape)

        # Variables to be set for every Nadir_grid
        self.points = None  # coordinate value 1d arrays: alt, 1st hor. coord, 2nd hor. coord.
        self.lon = None  # Longitude, 3d array, order of dims as above
        self.lat = None  # Latitude, 3d array
        self.alt = None  # Altitude, 1d array
        self.mask = None  # Image mask
        self.atm_shape = None  # Grid shape
        self.geo_coords = None  # Geolocated coordinates
        self.nalt = None  # Number of altitude layers
        self.reg_points = None  # Grid points rescaled so that the same reg. weigths would work across different grids

    def _set_derived(self):
        self.npoints = int(np.prod(self.atm_shape))

    def _jacobian_img(self, idx):
        gc = self.geo_coords[idx % self.nalt, idx // self.nalt, :, :, :]
        coords0 = np.stack([np.searchsorted(self.points[i + 1], gc[..., i].flatten(), sorter=None) - 1
                            for i in range(2)], axis=-1)
        # numrows = gc.shape[1]
        numpoints_im = coords0.shape[0]
        numpoints_grid = int(np.prod(self.atm_shape[1:]))

        coords = np.zeros((numpoints_im, 4, 2))
        dists = np.zeros((numpoints_im, 2, 2))
        iw = np.zeros((numpoints_im, 4))

        dists[:, :, 1] = np.stack([gc[..., i].flatten() - self.points[i + 1][coords0[:, i]] for i in range(2)], axis=-1)
        dists[:, :, 0] = np.stack([self.points[i + 1][coords0[:, i] + 1] - gc[..., i].flatten() for i in range(2)],
                                  axis=-1)

        # res = np.zeros(self.im_shape)
        norms = np.prod(dists[:, :, 0] + dists[:, :, 1], axis=1)
        for i, idx in enumerate(product([0, 1], repeat=2)):
            for j in range(2):
                coords[..., i, j] = coords0[..., j] + idx[j]
            iw[:, i] = dists[..., 0, idx[0]] * dists[..., 1, idx[1]]
            # res += iw[..., i] * idata[coords[..., i, 0], coords[..., i, 1]]
        # res /= norms[np.newaxis, :]
        iw *= self.scales[0] / norms[:, np.newaxis]

        cols = coords[..., 0] * self.atm_shape[-1] + coords[..., 1]
        rows = np.broadcast_to(np.arange(numpoints_im)[:, np.newaxis], cols.shape)
        valid = np.broadcast_to(self.valid_pix.flatten()[:, np.newaxis], iw.shape).flatten()
        return coo_matrix((iw.flatten()[valid], (rows.flatten()[valid], cols.flatten()[valid])),
                          shape=(numpoints_im, numpoints_grid))

    def calc_jacobian(self, nproc):
        num_alt_img = self.nalt * self.num_simages
        if nproc == 1:  # Simple serial processing for easier debugging
            img_jacs = [self._jacobian_img(i) for i in range(num_alt_img)]
        else:
            with Pool(processes=nproc) as pool:
                img_jacs = pool.starmap(self._jacobian_img, [(j,) for j in range(num_alt_img)])

        alts_jacs = []
        while len(img_jacs) > 0:
            alts_jacs.append(sp.hstack([img_jacs.pop(0) for _ in range(self.nalt)]))

        jac = sp.vstack(alts_jacs)
        valid_points = np.zeros(self.npoints)
        valid_points[np.unique(jac.col)] = 1

        return jac.tocsr(), valid_points.reshape(self.atm_shape)

    def vec2atm(self, vec):
        return vec.reshape(self.atm_shape) * self.scales[0]

    def atm2vec(self, atm):
        return atm.flatten() / self.scales[0]


class Nadir_grid_lonlat(Nadir_grid):
    def __init__(self, nadir_data, conf, const, processes=1, mask=None):
        super().__init__(nadir_data, conf, const, processes, mask=mask)
        self.alt = nadir_data["alt"]
        self.nalt = len(self.alt)
        self.points = [self.alt]
        self.sampling_factor = conf.NADIR_SAMPLING_FACTOR
        cnames = ["lon", "lat"]
        for coord in cnames:
            lims = (np.min(nadir_data[coord]), np.max(nadir_data[coord]))
            ref_img = nadir_data[coord][0, 0, :, :]
            im_step = max([np.abs(np.mean(np.diff(ref_img, axis=i))) for i in range(2)])
            self.points.append(np.arange(lims[0] - im_step * 1.1, lims[1] + im_step * 1.1,
                                         im_step * self.sampling_factor))

        self.atm_shape = tuple([len(x) for x in self.points])
        self.lon = np.broadcast_to(self.points[1][np.newaxis, :, np.newaxis], self.atm_shape)
        self.lat = np.broadcast_to(self.points[2][np.newaxis, np.newaxis, :], self.atm_shape)
        self.geo_coords = np.stack([np.swapaxes(nadir_data[name], 2, 3) for name in ["lon", "lat"]], axis=-1)
        self.reg_points = [self.points[i] * f for i, f in enumerate([1, np.cos(np.deg2rad(np.mean(self.lat))), 1])]
        self._set_derived()

    def write_nadir_L2_ncdf(self, obs, sol, fname, mask=True, hot_pix=None):
        dim_pars = {"alt": ("Altitude", "meter", self.points[0] * 1e3),
                    "lon": ("Longitude", "degree_east", self.points[1]),
                    "lat": ("Longitude", "degree_north", self.points[2]),
                    "time": ("time", "seconds since 2000.01.01 00:00 UTC", self.img_time),
                    "img_col": ("Column of (coadded) pixels in the image", None, np.arange(self.im_shape[0])),
                    "img_row": ("Row of (coadded) pixels in the image", None, np.arange(self.im_shape[1]))}
        im_dims = ("time", "img_col", "img_row")
        rad_unit = "ph/cm^2/s/srad"
        imgs = obs[0, ...]
        if mask:
            imgs = np.where(self.valid_obs, imgs, np.nan)
        ncvars = {"ImageCalibrated": ("MATS images", rad_unit, imgs, im_dims),
                  "pix_lon": ("Geolocated longitude for each pixel", "degree_east",
                              self.geo_coords[..., 0], ("alt", ) + im_dims),
                  "pix_lat": ("Geolocated latitude for each pixel", "degree_north",
                              self.geo_coords[..., 1], ("alt", ) + im_dims),
                  "ImageLayer": ("Radiance originating from each altitude layer", rad_unit, sol, ("alt", "lon", "lat"))}
        if hot_pix is not None:
            ncvars["HotPixels"] = ("Hot pixel map that was subtracted from each image", rad_unit,
                                   hot_pix, ("img_col", "img_row"))
        write_gen_ncdf(fname, dim_pars, ncvars, {})


class Nadir_grid_rotated(Nadir_grid):
    def __init__(self, nadir_data, conf, const, processes=1, mask=None):
        super().__init__(nadir_data, conf, const, processes, mask=mask)
        self.alt = nadir_data["alt"] * 1e3
        self.nalt = len(self.alt)
        self.points = [self.alt]
        self.sampling_factor = conf.NADIR_SAMPLING_FACTOR
        geo_lon, geo_lat = [np.swapaxes(nadir_data[name], 2, 3) for name in ["lon", "lat"]]
        geo_alt = np.broadcast_to(self.alt[:, np.newaxis, np.newaxis, np.newaxis], geo_lat.shape)

        # Geolocation coordinates as ECEF cartesians
        geo_ecef = np.stack(wgs842ecef(geo_lat, geo_lon, geo_alt), axis=-1)
        self.ecef_to_local = self.generate_local_transform(geo_ecef)

        # Geolocation corrdinates in local grid
        geo_shape = list(geo_ecef.shape)
        geo_shape[-1] = 2
        rotated = self.ecef_to_local.apply(geo_ecef.reshape(-1, 3))
        del geo_ecef
        self.geo_coords = np.stack(cart2sph(rotated)[1:3], axis=-1).reshape(geo_shape)

        # Define local grid
        for i in range(2):
            coord = self.geo_coords[0, :, :, :, i]
            lims = (np.min(coord), np.max(coord))
            ref_img = self.geo_coords[0, int(geo_shape[1] / 2), :, :, i]
            im_step = max([np.abs(np.mean(np.diff(ref_img, axis=j))) for j in range(2)])
            self.points.append(np.arange(lims[0] - im_step * 2.1, lims[1] + im_step * 2.1,
                                         im_step * self.sampling_factor))

        self.atm_shape = tuple([len(x) for x in self.points])
        self.lon, self.lat = self.get_grid_lonlat()
        scale = 180 / np.pi
        self.reg_points = [self.points[i] * f for i, f in enumerate([1, scale, scale])]
        self.mean_rad = geoid_radius(np.deg2rad(self.lat.mean())) + self.alt[1]
        self._set_derived()

    @staticmethod
    def generate_local_transform(geo_ecef):
        # Calculate the transform from ecef to the local coordinate system.

        num_img, num_rows, num_cols = geo_ecef.shape[1:4]

        # First, middle, and last image indices
        ref_img = (0, int((num_img - 1) / 2), num_img - 1)
        center_row, center_col = [int(np.round(s / 2.0)) for s in [num_rows, num_cols]]
        ref_ecef = np.stack([geo_ecef[1, idx, center_row, center_col, :] for idx in ref_img], axis=0)

        orb_plane_normal = np.cross(ref_ecef[0, :], ref_ecef[2, :])
        orb_plane_normal = orb_plane_normal / np.linalg.norm(orb_plane_normal)  # normalize vector

        ecef_mid_unit = ref_ecef[1, :] / np.linalg.norm(ref_ecef[1, :])  # unit vector for central position
        ecef_to_local = R.align_vectors([[1, 0, 0], [0, 1, 0]], [ecef_mid_unit, -orb_plane_normal])[0]
        return ecef_to_local

    def get_grid_lonlat(self):
        points = self.points.copy()
        points[0] = np.ones_like(points[0])  # Use unit radius
        grid = np.stack(sph2cart(*np.meshgrid(*points, indexing='ij')), axis=-1)  # In local cartesian coordinates
        _, lon, lat = cart2sph(self.ecef_to_local.inv().apply(grid.reshape(-1, 3)))
        return [np.rad2deg(arr.reshape(grid.shape[:-1])) for arr in [lon, lat]]

    def write_nadir_L2_ncdf(self, obs, sol, fname, mask=True, hot_pix=None):
        dim_pars = {"alt": ("Altitude", "kilometer", self.points[0] * 1e-3),
                    "across": ("Acrosstrack coordinate", "kilometer", self.points[1] * self.mean_rad * 1e-3),
                    "along": ("Alongtrack coordinate", "kilometer", self.points[2] * self.mean_rad * 1e-3),
                    "time": ("time", "seconds since 2000.01.01 00:00 UTC", self.img_time),
                    "img_col": ("Column of (coadded) pixels in the image", None, np.arange(self.im_shape[0])),
                    "img_row": ("Row of (coadded) pixels in the image", None, np.arange(self.im_shape[1]))}
        im_dims = ("time", "img_col", "img_row")
        sol_dims = ("alt", "across", "along")
        rad_unit = "ph/cm^2/s/srad"
        imgs = obs[0, ...]
        if mask:
            imgs = np.where(self.valid_obs, imgs, np.nan)
        ncvars = {"ImageCalibrated": ("MATS images", rad_unit, imgs, im_dims),
                  "pix_lon": ("Geolocated longitude for each pixel", "degree_east",
                              self.geo_coords[..., 0], ("alt", ) + im_dims),
                  "pix_lat": ("Geolocated latitude for each pixel", "degree_north",
                              self.geo_coords[..., 1], ("alt", ) + im_dims),
                  "ImageLayer": ("Radiance originating from each altitude layer", rad_unit, sol, sol_dims),
                  "lon": ("Longitude of ImageLayer pixels", "degree_east", self.lon, sol_dims),
                  "lat": ("Latitude of ImageLayer pixels", "degree_north", self.lat, sol_dims),
                  }
        if hot_pix is not None:
            ncvars["HotPixels"] = ("Hot pixel map that was subtracted from each image", rad_unit,
                                   hot_pix, ("img_col", "img_row"))
        write_gen_ncdf(fname, dim_pars, ncvars, {})


class Nadir_forward_model():
    def __init__(self, grid):
        self.grid = grid

        obs = {name: np.arange(grid.im_shape[i]) for i, name in enumerate(["columns", "rows"])}
        obs["valid_obs"] = grid.valid_obs
        self.obs = SimpleNamespace(**obs)

        self.channels = ["NADIR"]
        self.sparse = True
        self.debug_nan = False

    def prepare_obs(self, imgs):
        return np.swapaxes(imgs, 1, 2)[np.newaxis, ...]

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interpn
from skyfield.framelib import itrs

# from mats_l1_processing.pointing import pix_deg
from mats_l2_processing.util import get_image, cart2sph, sph2cart, geoid_radius, \
    make_grid_proto, grid_from_proto
from mats_l2_processing.io import write_gen_ncdf, append_gen_ncdf
from mats_l2_processing.grid import Grid

import logging


class Alt_1D_stacked_grid(Grid):
    def __init__(self, metadata, conf, const, column, processes=1, verify=False):
        super().__init__(metadata, conf, const, processes)

        # Initialize basic grid
        row_range = (0, metadata["NROW"][0]) if conf.ROW_RANGE[0] < 0 else conf.ROW_RANGE
        self.rows = np.arange(row_range[0], row_range[1], 1)
        self.columns = np.array([column])

        self.local_geoid_radius = geoid_radius(np.deg2rad(np.mean(metadata["TPlat"])))
        lims = self.grid_limits(metadata)
        print(f"Grid limits: {lims}")
        # lims = (0, 1e8)
        grid_proto = make_grid_proto(conf.ALT_GRID, scaling=1e3)
        self.edges = [grid_from_proto(grid_proto, lims)]

        # Set derived attributes
        self._set_derived(metadata, processes, False, verify)

        # Set geolocation attributes
        self.alt = np.broadcast_to(self.centers[0][np.newaxis, :], self.atm_shape[1:])

        if conf.GEOLOCATE_1D_FROM_TP:
            self.lat, self.lon = self._get_lat_lon(metadata)
        else:
            self.lat = np.broadcast_to(metadata["TPlat"][:, np.newaxis], self.atm_shape[1:])
            self.lon = np.broadcast_to(metadata["TPlon"][:, np.newaxis], self.atm_shape[1:])

    def grid_limits(self, data):
        mid = int((data["size"] - 1) / 2)

        mid_date = data['EXPDate'][mid]
        current_ts = self.timescale.from_datetime(mid_date)

        # Rotation between satellite pointing and channel pointing
        rot_sat_channel = R.from_quat(data['qprime'][mid, :])

        q = data['afsAttitudeState'][mid, :]  # Satellite pointing in ECI
        rot_sat_eci = R.from_quat(np.roll(q, -1))  # Rotation matrix for q (should go straight to ecef?)

        eci_to_ecef = R.from_matrix(itrs.rotation_at(current_ts))
        satpos_eci = data['afsGnssStateJ2000'][mid, 0:3]
        satpos_ecef = eci_to_ecef.apply(satpos_eci)

        get_los_vars = ['channel', 'NCSKIP', 'NCBINCCDColumns', 'NRSKIP', 'NRBIN', 'NCOL', 'EXPDate', 'TPlat', 'TPlon']
        im_min, im_max = [], []
        for idx in range(data["size"]):
            for row in [self.rows[0], self.rows[-1]]:
                image = get_image(data, idx, get_los_vars)
                los_ecef = self.get_los_ecef(image, self.columns[0], row, rot_sat_channel, rot_sat_eci, eci_to_ecef)
                steps_in_ecef = self._get_steps_in_ecef(image, satpos_ecef, los_ecef, localR=None)
                steps_in_own = self._get_steps_in_own_grid(steps_in_ecef)
                im_min.append(steps_in_own.min())
                im_max.append(steps_in_own.max())

        return (min(im_min) - 1e3, max(im_max) + 1e3)

    def _get_steps_in_own_grid(self, pos):
        radial_coord = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)
        geoid_rad = geoid_radius(np.arcsin(pos[:, 2] / radial_coord))  # Get geoid radius for each point from latitude
        return radial_coord - geoid_rad

    def write_grid_ncdf(self, fname, attributes={}):
        # Define dimensions
        # eff_radius = self.local_geoid_radius + np.mean(self.centers[0])
        dim_pars = {"alt_coord": ("Altitude", "meter", self.centers[0]),
                    "img_time": ("Acquisition time of individual MATS images", "Seconds since 2000.01.01 00:00 UTC",
                                 self.img_time),
                    "img_col": ("Column of (coadded) pixels in the image", None, self.columns),
                    "img_row": ("Row of (coadded) pixels in the image", None, self.rows),
                    # "time": ("Valid time of L2 data", "Seconds since 2000.01.01 00:00 UTC",
                    #         self.valid_time * np.ones(1))
                    }
        dims = ("img_time", "alt_coord")

        # Define coordinate variables
        ncvars = {"altitude": ("Altitude", "meter", self.alt, dims),
                  "longitude": ("Longitude", "degree_east", self.lon, dims),
                  "latitude": ("Latitude", "degree_north", self.lat, dims),
                  "TPheight": ("Tangent point height", "meter", self.TP_heights[:, 0, :], ("img_time", "img_row"))}

        write_gen_ncdf(fname, dim_pars, ncvars, attributes)

    def write_atm_ncdf(self, fname, atm, atm_suffix="", atm_suffix_long=""):
        ncvars = {}
        dims = ("img_time", "alt_coord")
        # dims = ("time", "radial_coord", "acrosstrack_coord", "alongtrack_coord")
        for i, qty in enumerate(self.ret_qty):
            ncvars[f"{qty}{atm_suffix}"] = (f"{self.ncpar[qty][0]}{atm_suffix_long}", self.ncpar[qty][1], atm[i],
                                            dims)
        append_gen_ncdf(fname, ncvars)

    def write_obs_ncdf(self, fname, obs, channels, obs_suffix="", obs_suffix_long="", attributes={}):
        ncvars = {}
        dims = ("img_time", "img_col", "img_row")
        # dims = ("img_time", "alt_coord")
        for i, chn in enumerate(channels):
            ncvars[f"{chn}{obs_suffix}"] = (f"{self.ncpar[chn][0]}{obs_suffix_long}", self.ncpar[chn][1],
                                            obs[i, :, :, :], dims)
        append_gen_ncdf(fname, ncvars, attributes=attributes)

    def _get_lat_lon(self, metadata):
        shape = metadata["TPECEFz"].shape[:2]
        tp_ecef = np.stack([metadata[name][:, :, self.columns[0]].flatten()
                            for name in ["TPECEFx", "TPECEFy", "TPECEFz"]], axis=1)
        tpr, tplon, tplat = [arr.reshape(shape) for arr in cart2sph(tp_ecef)]
        tpalt = tpr - geoid_radius(tplat)

        lon, lat = [np.zeros((shape[0], len(self.centers[0]))) for _ in range(2)]
        for im in range(shape[0]):
            lat[im, :], lon[im, :] = [np.interp(self.centers[0], tpalt[im, :], arr[im, :])
                                      for arr in [tplat, tplon]]
        return np.rad2deg(lat), np.rad2deg(lon)

    def interpolate_from_3D(self, ext_coords, ext_data):
        # Prepare coordinates
        ret_coords = np.zeros(list(self.alt.shape) + [3])
        ret_coords[..., 0] = self.alt
        ret_coords[..., 1] = self.lat
        ret_coords[..., 2] = self.lon

        # Interpolate
        res = []
        for extd in ext_data:
            res.append(interpn(ext_coords, extd, ret_coords))
        return res

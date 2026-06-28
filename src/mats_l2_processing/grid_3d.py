import numpy as np
import netCDF4 as nc
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interpn, CubicSpline
from skyfield import api as sfapi
from skyfield.framelib import itrs
# from mats_l1_processing.pointing import pix_deg
from mats_l2_processing.util import get_image, cart2sph, sph2cart, geoid_radius, ecef2wgs84
from mats_l2_processing.io import write_gen_ncdf, append_gen_ncdf
from mats_l2_processing.grid import Grid
from mats_l2_processing.pointing import calc_track_ecef
import sys
import logging


class Alt_along_3D_grid(Grid):
    def __init__(self, all_metadata, conf, const, processes=1, verify=False, copy_points_from=None):
        super().__init__(all_metadata, conf, const, processes)
        metadata = all_metadata[0]

        # Initialize basic grid
        if copy_points_from is None:
            self.ecef_to_local = self.generate_local_transform(metadata, self.timescale)
            self.local_geoid_radius = geoid_radius(np.deg2rad(np.mean(metadata["TPlat"])))
            self.ref_rad = const.MEAN_EARTH_RADIUS + self.ref_alt
            self.track_spline = None  # To be updated below!
            self.track_lims = None  # To be updated below!
            # lims = self.grid_limits(metadata)
            lims = (None, None, None)
            self.scalings = np.array([1e3, 1e3 / self.ref_rad, 1e3 / self.ref_rad])
            self.offsets = np.zeros(3)
            self.points = [self._set_points(name, conf, const, lims=lims[i], scaling=self.scalings[i], offset=self.offsets[i])
                           for i, name in enumerate(["ALT_GRID", "ACROSS_GRID", "ALONG_GRID"])]
            if conf.STRICT_ALONGTRACK:
                tp_track = calc_track_ecef(metadata, self.pointing.chn_map(conf.CHANNELS[0]), processes, const.TP_VARS,
                                           c_col=conf.CENTRAL_COLUMN, ny=10, ref_alt=self.ref_alt)
                alt, across, along = cart2sph(self.ecef_to_local.apply(tp_track))
                if along[0] > along[-1]:
                    along, across = [np.flip(arr) for arr in [along, across]]
                    logging.info("Grid flipped!")
                else:
                    logging.info("Grid not flipped!")
                self.track_spline = CubicSpline(along, across, extrapolate=True)
                self.track_lims = np.empty((3, 2))
                self.track_lims[0, :] = (along.min(), along.max())
                self.track_lims[1, :] = [self.track_spline(x) for x in self.track_lims[0, :]]
                self.track_lims[2, :] = [self.track_spline(x, 1) for x in self.track_lims[0, :]]
                self.along_vtime = np.interp(self.points[2], along, self.img_time, left=np.nan, right=np.nan)
                valid_duration = (self.img_time[-conf.TOMO_PADDING_IMG] - self.img_time[conf.TOMO_PADDING_IMG])
                if valid_duration / const.NOMINAL_IMG_STEP - 10 > len(self.img_time) - 2 * conf.TOMO_PADDING_IMG:
                    valid_duration = 0
                self.along_valid = np.abs(self.along_vtime - np.mean(self.img_time)) < valid_duration / 2

        else:
            egrid = copy_points_from
            self.ecef_to_local = egrid.ecef_to_local
            self.local_geoid_radius = egrid.local_geoid_radius
            self.scalings, self.offsets = egrid.scalings, egrid.offsets
            self.points = egrid.points
            self.track_spline = egrid.track_spline

        # Set derived attributes
        self._set_derived(metadata, processes, True, verify)

        # Set geolocation attributes
        self.alt = np.broadcast_to(self.points[0][:, np.newaxis, np.newaxis], self.atm_shape[1:])
        self.lat, self.lon = self.get_lat_lon()

    def grid_limits(self, data):
        first = 0
        mid = int((data["size"] - 1) / 2)
        last = data["size"] - 1

        mid_date = data['time'][mid]
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

        get_los_vars = ['channel', 'NCSKIP', 'NCBINCCDColumns', 'NRSKIP', 'NRBIN', 'NCOL', 'time', 'TPlat', 'TPlon']
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
        min_alt = poslocal_sph[:, 0].min()
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

        return (min_alt - 5e3, self.top_alt + 5e3, nalt), (min_across - 0.1, max_across + 0.1, nacross), \
            (min_along - 0.6, max_along + 0.6, nalong)

    def _get_steps_in_own_grid_simple(self, pos):
        poslocal_sph = np.array(cart2sph(self.ecef_to_local.apply(pos))).T
        geoid_rad = geoid_radius(np.arcsin(pos[:, 2] / poslocal_sph[:, 0]))
        poslocal_sph[:, 0] -= geoid_rad
        return poslocal_sph

    def _get_steps_in_own_grid(self, pos):
        lat, lon, alt = ecef2wgs84(pos)
        nx, ny, nz = sph2cart(np.ones_like(lat), lon, lat)
        del lon, lat
        poslocal_sph = np.array(cart2sph(self.ecef_to_local.apply(np.stack((nx, ny, nz), axis=1)))).T
        poslocal_sph[:, 0] = alt
        if self.track_spline is not None:
            poslocal_sph[:, 1] = poslocal_sph[:, 1] - self.interp_track(poslocal_sph[:, 2])
        return poslocal_sph

    def get_lat_lon(self):
        acrr, alongg = np.meshgrid(*self.points[1:], indexing="ij")
        if self.track_spline is not None:
            track_offs = self.interp_track(self.points[2])
            np.save("track.npy", track_offs)
            acrr = acrr + track_offs[np.newaxis, :]

        # Cartesian unit vectors in directions of horizontal slice of local grid
        lxx, lyy, lzz = sph2cart(np.ones_like(acrr), acrr, alongg)
        glgrid = self.ecef_to_local.inv().apply(np.dstack((lxx.flatten(), lyy.flatten(), lzz.flatten()))[0, :, :])
        latt = np.arcsin(glgrid[:, 2].reshape(acrr.shape))
        latt = np.rad2deg(latt)
        lonn = np.rad2deg(np.arctan2(glgrid[:, 1], glgrid[:, 0]).reshape(acrr.shape))
        return [np.broadcast_to(arr[np.newaxis, :, :], self.atm_shape[1:]) for arr in [latt, lonn]]

    def interp_track(self, xi):
        return np.where(xi < self.track_lims[0, 0],
                        self.track_lims[1, 0] + self.track_lims[2, 0] * (xi - self.track_lims[0, 0]),
                        np.where(xi > self.track_lims[0, 1],
                        self.track_lims[1, 1] + self.track_lims[2, 1] * (xi - self.track_lims[0, 1]),
                        self.track_spline(xi))
                       )

    def write_grid_ncdf(self, fname, attributes={}):
        # Define dimensions
        #eff_radius = self.local_geoid_radius + np.mean(self.points[0])
        eff_radius = self.ref_rad
        dim_pars = {"alt_coord": ("Altitude", "meter", self.points[0]),
                    "acrosstrack_coord": ("Horizontal coordinate in the direction perpendicular to the orbital plane",
                                          "meter", self.points[1] * eff_radius),
                    "alongtrack_coord": ("Horizontal coordinate in the direction of the satellite track", "meter",
                                         self.points[2] * eff_radius),
                    "img_time": ("Acquisition time of individual MATS images", "Seconds since 2000.01.01 00:00 UTC",
                                 self.img_time),
                    "img_col": ("Column of (coadded) pixels in the image", None, self.columns),
                    "img_row": ("Row of (coadded) pixels in the image", None, self.rows),
                    "time": ("Valid time of L2 data", "Seconds since 2000.01.01 00:00 UTC",
                             self.valid_time * np.ones(1))}
        dims_4D = ("time", "alt_coord", "acrosstrack_coord", "alongtrack_coord")
        # dims_2D = ("acrosstrack_coord", "alongtrack_coord")

        # Define coordinate variables
        ncvars = {"altitude": ("Altitude", "meter", self.alt, dims_4D),  # For compatibility with old scripts
                  "longitude": ("Longitude", "degree_east", self.lon, dims_4D),
                  "latitude": ("Latitude", "degree_north", self.lat, dims_4D)}
                 # "TPheight": ("Tangent point height", "meter", self.TP_heights, ("img_time", "img_col", "img_row"))}

        write_gen_ncdf(fname, dim_pars, ncvars, attributes)

    def write_grid_ncdf_exp(self, fname, attributes={}, track_width=None):
        # Define dimensions
        #eff_radius = self.local_geoid_radius + np.mean(self.points[0])
        eff_radius = self.ref_rad
        dim_pars = {"alt_coord": ("Altitude", "meter", self.points[0]),
                    "acrosstrack_coord": ("Horizontal coordinate in the direction perpendicular to the orbital plane",
                                          "meter", self.points[1] * eff_radius),
                    "alongtrack_coord": ("Horizontal coordinate in the direction of the satellite track", "meter",
                                         self.points[2] * eff_radius),
                    "img_time": ("Acquisition time of individual MATS images", "Seconds since 2000.01.01 00:00 UTC",
                                 self.img_time),
                    "img_col": ("Column of (coadded) pixels in the image", None, self.columns),
                    "img_row": ("Row of (coadded) pixels in the image", None, self.rows),
                    "time": ("Valid time of L2 data", "Seconds since 2000.01.01 00:00 UTC",
                             self.valid_time * np.ones(1)),
}
        dims_4D = ("time", "alt_coord", "acrosstrack_coord", "alongtrack_coord")
        # dims_2D = ("acrosstrack_coord", "alongtrack_coord")

        # Define coordinate variables
        ncvars = {"altitude": ("Altitude", "meter", self.alt, dims_4D),  # For compatibility with old scripts
                  "longitude": ("Longitude", "degree_east", self.lon, dims_4D),
                  "latitude": ("Latitude", "degree_north", self.lat, dims_4D),
                  "alongtrack_valid_time": ("Approximate valid time for each alongtrack coordinate",
                                            "Seconds since 2000.01.01 00:00 UTC", self.along_vtime, ("alongtrack_coord",)),
                  "alongtrack_valid_data": ("Valid data flag for alongtrack coordinates",
                                            "Seconds since 2000.01.01 00:00 UTC", self.along_valid, ("alongtrack_coord",))}
                 # "TPheight": ("Tangent point height", "meter", self.TP_heights, ("img_time", "img_col", "img_row"))}
        if track_width is not None:
            ncvars["track_width"] = ("Acrosstrack width of valid data region", "meter", track_width, ("img_time"))

        write_gen_ncdf(fname, dim_pars, ncvars, attributes)

    def write_to_grids(self, idx, fname, sza=None):
        try:
            with nc.Dataset(fname, 'r+') as nf:
                nf["longitude"][idx, :, :] = self.lon[0, :, :]
                nf["latitude"][idx, :, :] = self.lat[0, :, :]
                nf["chunk_img_time"][idx, :len(self.img_time)] = self.img_time
                nf["valid_time_alongtrack"][idx, :] = self.along_vtime
                nf["valid_data_alongtrack"][idx, :] = self.along_valid
                nf["valid_chunk"][idx] = 1
                if sza is not None:
                    nf["sza"][idx, :] = np.interp(self.along_vtime, self.img_time, sza, left=np.nan, right=np.nan)
        except Exception:
            print(f"WARN: data write failed for chunk {idx}!")

    def write_atm_ncdf(self, fname, atm, atm_suffix="", atm_suffix_long=""):
        ncvars = {}
        dims = ("time", "alt_coord", "acrosstrack_coord", "alongtrack_coord")
        for i, qty in enumerate(self.ret_qty):
            ncvars[f"{qty}{atm_suffix}"] = (f"{self.ncpar[qty][0]}{atm_suffix_long}", self.ncpar[qty][1], atm[i],
                                            dims)
        append_gen_ncdf(fname, ncvars)

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

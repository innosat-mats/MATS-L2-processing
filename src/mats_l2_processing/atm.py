import numpy as np
import netCDF4 as nc
from re import sub
from scipy.interpolate import LinearNDInterpolator, interpn
from scipy.signal import savgol_filter
from os.path import expanduser

from mats_l2_processing.util import running_mean, seconds2DT


def apr_from_1D(geo, geo3d, ver_1D, along_dists, decay_range=None, hw=4):
    stack = ver_1D.copy()
    stack = np.maximum(stack, 0.0)

    # Get rid of spurious top-altitude maxima
    if decay_range is not None:
        decay_range = [x * 1e3 for x in decay_range]
        stack = np.where(geo["alt"] > decay_range[1], 0.0, stack)
        stack = np.where(np.logical_and(geo["alt"] < decay_range[1], geo["alt"] > decay_range[0]),
                         stack * ((geo["alt"] - decay_range[1]) / (decay_range[1] - decay_range[0])) ** 2, stack)
    # plot_map(geo["alt"], stack, "VER_apr_map_decayed")

    # Since ND interpolator cannot extrapolate, extend the ends of input data to cover the 3d array
    ealts = geo["alt"].copy()
    ealts[:, 0] = 0
    ealts[:, -1] = 2e5
    ealong = along_dists.copy()
    ealong[0] = geo3d["alongtrack_grid_c"][0] - 1.0
    ealong[-1] = geo3d["alongtrack_grid_c"][-1] + 1.0

    # Interpolate on 3D grid
    ealong = np.broadcast_to(ealong[:, np.newaxis], ealts.shape)
    points = np.stack([ealong.flatten(), ealts.flatten()], axis=1)
    interp = LinearNDInterpolator(points, stack.flatten())
    res = interp(np.broadcast_to(geo3d["alongtrack_grid_c"][np.newaxis, np.newaxis, :], geo3d["alt"].shape),
                 geo3d["alt"])

    # Apply running mean to smoothen the data
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j, :] = running_mean(res[i, j, :], hw)
    for i in range(res.shape[0]):
        for j in range(res.shape[2]):
            res[i, :, j] = running_mean(res[i, :, j], 4)
    for i in range(res.shape[1]):
        for j in range(res.shape[2]):
            res[:, i, j] = running_mean(res[:, i, j], 3)

    # plot_map(geo3d["alt"][:, 19, :].T, res[:, 19, :].T, "VER_apr_map_slice", along_coord=geo3d["alongtrack_grid_c"])
    return res


class gridded_data():
    def __init__(self, spec, extend_previous={}, from_grid=None, from_L2_ncdf=None, ncdf_preload=None):
        if not ((from_grid is None) ^ (from_L2_ncdf is None)):
            raise ValueError("gridded_data requires either grid or L2 ncdf file for initialization!")
        self.data = extend_previous.copy()
        if from_grid is not None:
            self.centers = from_grid.centers
            self.alt = from_grid.alt
            self.lat = from_grid.lat
            self.lon = from_grid.lon
            self.valid_time = from_grid.valid_time
        if from_L2_ncdf is not None:
            with nc.Dataset(from_L2_ncdf, 'r') as nf:
                self.alt = nf["alt"][:]
                self.lat = nf["lat"][:]
                self.lon = nf["lon"][:]
                self.valid_time = nf["time"][0]
                self.centers = [nf[coord_name][:] for coord_name in nf["alt"].dimensions]
                self.ncdf_file = from_L2_ncdf
                for var in ncdf_preload:
                    self.data[var] = nf[var][:]
        self.shape = self.alt.shape

        methods = {"L2": self._L2, "jawara": self._model, "msis": self._model,
                   "running_mean": self._running_mean, "savgol": self._savgol,
                   "fill": self._fill, "eval": self._eval, "alt_taper": self._alt_taper}
        for names, method, gargs in spec:
            if method in methods.keys():
                output = methods[method](method, gargs)
            else:
                raise ValueError(f"Invalid gridded data method {method}!")
            if len(output) != len(names):
                raise ValueError(f"Gridded data method {method} should have generated {len(names)} variables " +
                                 f"to set {names}, but {len(output)} variables were generated.")
            for name, out in zip(names, output):
                self.data[name] = out

    def _verify_arguments(self, method, gargs, needed):
        for key in needed:
            if key not in gargs.keys():
                raise ValueError(f"Gridded data method '{method}' is missing a required argument {key}!")

    def _verify_shape(self, shape, prefix="Loaded variable"):
        if shape != self.shape:
            raise ValueError(f"{prefix} has shape {shape}, but retrieval grid has shape {self.shape}! Abort!")

    def _L2(self, method, gargs):
        self._verify_arguments("L2", gargs, ["path", "var"])

        res = []
        with nc.Dataset(expanduser(gargs["path"]), 'r') as nf:
            for vname in gargs["var"]:
                if vname not in nf.variables:
                    raise ValueError(f"The file {gargs['path']} has no variable called {vname}!")
                vdata = nf[vname][0, ...].data
                self._verify_shape(vdata.shape, prefix=f"Variable {vname} in file {gargs['path']}")
                res.append(vdata)
        return res

    def _model(self, method, gargs):
        self._verify_arguments(method, gargs, ["path", "var"])

        if method == 'jawara':
            coord_names = ["time", "height", "latitude", "longitude"]
            with nc.Dataset(expanduser(gargs["path"]), 'r') as nf:
                coords = {name: nf[name][:] for name in coord_names}
                time_idx = np.argmin(np.abs(coords["time"] - self.valid_time))
                if np.abs(self.valid_time - coords["time"][time_idx]) > 3600:
                    raise ValueError("Time of observation is outside JAWARA dataset time range! Abort!")
                rvars = {}
                lon = coords["longitude"]
                for name in gargs["var"]:
                    nd = nf[name][time_idx, ...]
                    rvars[name] = np.concatenate((nd[:, :, lon >= 180.0], nd[:, :, lon < 180.0],
                                                  nd[:, :, lon == 180.0]), axis=2)
                lon = np.concatenate((lon[lon >= 180.0] - 360.0, lon[lon < 180.0], [180.0]), axis=0)
            if "p" in rvars.keys():
                rvars["p"] *= 1e2
            # longitude = np.where(coords["longitude"] < 180.0, coords["longitude"], coords["longitude"] - 360.0)
            model_coords = (coords["height"], coords["latitude"], lon)

        elif method == 'msis':
            month = seconds2DT(self.valid_time).month
            coord_names = ["z", "lat"]
            with nc.Dataset(expanduser(gargs["path"]), 'r') as nf:
                coords = {name: nf[name][:] for name in coord_names}
                rvars = {name: nf[name][month, ...].T for name in gargs["var"]}
            if "o2" in rvars.keys():
                rvars["o2"] *= 1e-6
            model_coords = (coords["z"] * 1e3, coords["lat"], np.array([-181.0, 361.0]))
            shape = [len(c) for c in model_coords]
            rvars = {name: np.broadcast_to(data[..., np.newaxis], shape) for name, data in rvars.items()}
        else:
            raise ValueError(f"Invalid model type {method}!")

        ret_coords = np.stack([self.alt, self.lat, self.lon], axis=-1)
        return [interpn(model_coords, rvars[name], ret_coords) for name in gargs['var']]

    def _running_mean(self, method, gargs):
        self.verify_arguments("running_mean", gargs, ["input", "half_widths"])

        res = self.data[gargs["input"]].copy()
        hw = gargs["half_widths"]
        if hw[0] > 0:
            for i in range(res.shape[0]):
                for j in range(res.shape[1]):
                    res[i, j, :] = running_mean(res[i, j, :], hw[0])
        if hw[1] > 0:
            for i in range(res.shape[0]):
                for j in range(res.shape[2]):
                    res[i, :, j] = running_mean(res[i, :, j], hw[1])
        if hw[1] > 0:
            for i in range(res.shape[1]):
                for j in range(res.shape[2]):
                    res[:, i, j] = running_mean(res[:, i, j], hw[2])
        return [res]

    def _savgol(self, method, gargs):
        self._verify_arguments("savgol", gargs, ["input", "polyorder", "width"])

        res = []
        for in_var in gargs["input"]:
            data = self.data[in_var].copy()
            ndims = len(data.shape)
            # if not (len(gargs["polyorder"]) == ndims) and (len(gargs["width"]) == ndims):
            #     raise ValueError("Invalid Savitzky-Golay filter specification!")
            for d in range(ndims):
                if gargs["polyorder"][d] > 0:
                    data = savgol_filter(data, gargs["width"][d], gargs["polyorder"][d], axis=d)
            res.append(data)
        return res

    def _alt_taper(self, method, gargs):
        self._verify_arguments("fill", gargs, ["input", "boundary_alt", "scale_height", "reference"])
        exponent = - (self.alt - gargs["boundary_alt"]) / gargs["scale_height"]
        factor = np.exp(np.where(exponent < 0.0, exponent, 0.0))
        if gargs["reference"] == "local":
            return [qty * factor for qty in gargs["input"]]
        elif gargs["reference"] == "boundary":  # TODO: fix for other grids!
            boundary_idx = np.argmin(np.abs(self.centers[0] - gargs["boundary_alt"]))
            return [np.where(exponent < 0, self.data[qty][boundary_idx, :, :][np.newaxis, :, :] * factor,
                             self.data[qty]) for qty in gargs["input"]]
        else:
            raise ValueError(f"alt_taper reference can be 'local' or 'boundary', not {gargs['reference']}!")

    def _fill(self, method, gargs):
        self._verify_arguments("fill", gargs, ["value"])
        return [gargs["value"] * np.ones(tuple(self.shape))]

    def _eval(self, method, gargs):
        self._verify_arguments("eval", gargs, ["expr"])
        expanded = [sub(r'<(\w+)>', r'sd["\1"]', string) for string in gargs["expr"]]
        return [eval(string, {"sd": self.data}) for string in expanded]

    def write_to_ncdf(self, fname):
        pass

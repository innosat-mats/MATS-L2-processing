import numpy as np
import pandas
import datetime as DT
import netCDF4 as nc
import logging
from mats_l2_processing.grids import geoid_radius
from mats_l2_processing.util import DT2seconds


def read_ncdf(fname, vnames):
    res = {}
    with nc.Dataset(fname, 'r') as nf:
        for v in vnames:
            res[v] = nf[v][:]
    return res


def write_aux(jb, y, fx, tan_alts, atm_apr, prefix):
    # Centre grid (the working grid for Jacobian contains grid edges)
    # rad_grid_c, alongtrack_grid_c, acrosstrack_grid_c = [center_grid(jb[x]) for x in
    #                                                     ["rad_grid", "alongtrack_grid", "acrosstrack_grid"]]
    # Create alt/lat/lon grid for grid centres
    # alt, lon, lat, rr = localgrid_to_lat_lon_alt_3D(altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local)
    # alts, lats, lons = get_alt_lat_lon(rad_grid_c, acrosstrack_grid_c,
    #                                   alongtrack_grid_c, jb["ecef_to_local"])
    # Save results
    tan_alts = tan_alts.reshape((-1, jb["data"]["NCOL"][0], len(jb["rows"])))
    yg, fxg = [arr.reshape((2, -1, jb["data"]["NCOL"][0], len(jb["rows"]))) for arr in [y, fx]]
    img_time = DT2seconds(jb["data"]["EXPDate"].data).astype(float)
    # sp.save_npz(f"{args.out_np}_K.npz", K)
    np.savez_compressed(f"{prefix}_aux.npz", y=yg, fx=fxg, radial_coord=jb["rad_grid_c"],
                        alongtrack_coord=jb["alongtrack_grid_c"], acrosstrack_coord=jb["acrosstrack_grid_c"],
                        tan_alts=tan_alts, altitude=jb["alt"], latitude=jb["lat"], longitude=jb["lon"],
                        img_time=img_time, VER_bgr=atm_apr[0], T_bgr=atm_apr[1])


def write_iter(jb, fx, atm, prefix):
    fxg = fx.reshape((2, -1, jb["data"]["NCOL"][0], jb["data"]["NROW"][0]))
    # sp.save_npz(f"{args.out_np}_K.npz", K)
    np.savez_compressed(f"{prefix}_iter.npz", fx=fxg, VER=atm[0], T=atm[1], alongtrack_coord=jb["alongtrack_grid_c"],
                        acrosstrack_coord=jb["acrosstrack_grid_c"], altitude=jb["alt"])


def read_L1_ncdf(filename, var=None, start_img=None, stop_img=None, start_time=None, stop_time=None,
                 time_var='EXPDate', read_ncattrs=True):
    res = {}
    with nc.Dataset(filename, 'r') as nf:
        # Read in global attributes
        if type(read_ncattrs) is list:
            attrs = read_ncattrs
        else:
            attrs = nf.ncattrs() if read_ncattrs else []
        for attr in attrs:
            res[attr] = nf.getncattr(attr)
        if (start_img is not None) or (stop_img is not None):
            start = 0 if start_img is None else start_img
            stop = -1 if stop_img is None else stop_img
        elif (start_time is not None) or (stop_time is not None):
            assert len(nf[time_var].shape) == 1, f"Invalid time variable {time_var} selected."
            time = DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc) +\
                nf[time_var][:] * DT.timedelta(0, 1)
            start = 0 if start_time is None else np.nanargmin(np.abs(time - start_time))
            stop = -1 if stop_time is None else np.nanargmin(np.abs(time - stop_time))
        else:
            start, stop = 0, -1

        real_stop = stop if stop >= 0 else len(time) + stop
        assert start < real_stop, f"The selected time interval has no images! Read data for {time[0]} to {time[-1]}"
        if start_time < time[0] - DT.timedelta(1, 0) or stop_time > time[-1] + DT.timedelta(1, 0):
            print("WARNING: Data in the input file does not completely cover the selected time interval.")

        if var is None:
            var = nf.variables

        for name in var:
            v = nf[name]
            if hasattr(v, "units") and (v.units == "Seconds since 2000.01.01 00:00 UTC"):
                res[v.name] = DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc) +\
                    v[start:stop, ...] * DT.timedelta(0, 1)
            else:
                res[name] = v[start:stop, ...]

        res["size"] = res[var[0]].shape[0]
        logging.info(f"Read {res['size']} images.")
    return res


def write_L2_ncdf(jb, outfile, atm_vars=[], obs_vars=[], tan_alts=None, atts=[]):
    # atm_vars should be provided as list of tuples:
    # (atm, name_suffix, <long name or None for empty>))
    #
    # obs_vars should be provided as list of tuples:
    # (data, name_suffix, <long name or None for empty>))

    # Parsing coordinate data
    from_jb = [("radial_coord", "rad_grid_c"), ("acrosstrack_coord", "acrosstrack_grid_c"),
               ("alongtrack_coord", "alongtrack_grid_c"), ("altitude", "alt"), ("longitude", "lon"),
               ("latitude", "lat"), ("img_col", "columns"), ("img_row", "rows")]
    res = {}
    for item in from_jb:
        to_name, from_name = item if type(item) is tuple else (item, item)
        res[to_name] = jb[from_name]
    res["img_time"] = DT2seconds(jb["data"]["EXPDate"].data).astype(float)
    earth_radius = geoid_radius(0)
    res["acrosstrack_coord"] *= earth_radius
    res["alongtrack_coord"] *= earth_radius

    # Define dimensions
    dim_pars = {"radial_coord": ("Distance from the center of the Earth", "meter"),
                "acrosstrack_coord": ("Horizontal coordinate in the direction perpendicular to the orbital plane",
                                      "meter"),
                "alongtrack_coord": ("Horizontal coordinate in the direction of the satellite track", "meter"),
                "img_time": ("Acquisition time of individual MATS images", "Seconds since 2000.01.01 00:00 UTC"),
                "img_col": ("Column of (coadded) pixels in the image", None),
                "img_row": ("Row of (coadded) pixels in the image", None)}
    dims_4D = ("time", "radial_coord", "acrosstrack_coord", "alongtrack_coord")
    dims_2D = ("acrosstrack_coord", "alongtrack_coord")

    # Define coordinate variables
    ncvars = {"altitude": ("Altitude", "meter", res["altitude"], dims_4D),
              "longitude": ("Longitude", "degree_east", res["longitude"], dims_2D),
              "latitude": ("Latitude", "degree_north", res["latitude"], dims_2D)}

    # Write file
    with nc.Dataset(outfile, 'w') as nf:
        # Create dimensions and dimension variables
        for dim, param in dim_pars.items():
            nf.createDimension(dim, len(res[dim]))
            ncvar = nf.createVariable(dim, 'f8', (dim,))
            ncvar.long_name = param[0]
            if param[1] is not None:
                ncvar.units = param[1]
            ncvar[:] = res[dim]

        # Create a dummy time dimension and a variable for valid time of the L2 data product
        nf.createDimension("time", 1)
        ncvar = nf.createVariable("time", 'f8', ("time",))
        ncvar[0] = np.nanmean(res["img_time"][:])
        ncvar.long_name = "Valid time of L2 data"
        ncvar.units = "Seconds since 2000.01.01 00:00 UTC"

        # Write coordinate variables
        for name, param in ncvars.items():
            ncvar = nf.createVariable(name, 'f8', param[3])
            ncvar.long_name = param[0]
            if param[1] is not None:
                ncvar.units = param[1]
            ncvar[:] = param[2]

        # Write tangent point heights
        if tan_alts is not None:
            ncvar = nf.createVariable("TPheight", 'f8', ("img_time", "img_col", "img_row"))
            ncvar.long_name = "Tangent point height"
            ncvar.units = "meter"
            ncvar[:] = tan_alts

        # Write ncdf attributes
        if len(atts) > 0:
            nf.setncatts(atts)

    if not (len(atm_vars) == 0 and len(obs_vars) == 0):
        append_L2_ncdf(outfile, atm_vars, obs_vars)


def append_L2_ncdf(ncfile, atm_vars, obs_vars):
    # atm_vars should be provided as list of tuples:
    # (atm, name_suffix, <long name or None for empty>))
    #
    # obs_vars should be provided as list of tuples:
    # (data, name_suffix, <long name or None for empty>))

    dims_4D = ("time", "radial_coord", "acrosstrack_coord", "alongtrack_coord")
    dims_obs = ("img_time", "img_col", "img_row")
    defaults = {"VER": ("Volume emission rate", "ph/cm^3/s", None, dims_4D),
                "T": ("Temperature", "Kelvin", None, dims_4D),
                "IR1": ("Infrared image channel 1", "ph/cm^2/s/srad", None, dims_obs),
                "IR2": ("Infrared image channel 2", "ph/cm^2/s/srad", None, dims_obs)}
    dim_names = ["time", "radial_coord", "acrosstrack_coord", "alongtrack_coord",
                 "img_time", "img_col", "img_row"]

    ncvars = {}

    with nc.Dataset(ncfile, 'r+') as nf:
        sizes = {name: len(nf.dimensions[name]) for name in dim_names}
        atm_shape = (sizes["time"], sizes["radial_coord"], sizes["acrosstrack_coord"], sizes["alongtrack_coord"])
        obs_shape = (2, sizes["img_time"], sizes["img_col"], sizes["img_row"])

        for data, suffix, long_name in atm_vars:
            for i, v in enumerate(["VER", "T"]):
                ncvars[f"{v}{suffix}"] = (None if long_name is None else f"{defaults[v][0]} {long_name}",
                                          defaults[v][1], data[i].reshape(atm_shape), defaults[v][3])

        for data, suffix, long_name in obs_vars:
            chn_data = data.reshape(obs_shape)
            for i, v in enumerate(["IR1", "IR2"]):
                ncvars[f"{v}{suffix}"] = (None if long_name is None else f"{defaults[v][0]} {long_name}",
                                          defaults[v][1], chn_data[i, ...], defaults[v][3])

        for name, param in ncvars.items():
            ncvar = nf.createVariable(name, 'f8', param[3])
            ncvar.long_name = param[0]
            if param[1] is not None:
                ncvar.units = param[1]
            ncvar[:] = param[2]


def L2_write_init(jb, prefix, atm_apr, y, fx, tan_alts, ret_alt_range, tp_alt_range):
    name = f"{prefix}_L2.nc"
    atm_vars = [(atm_apr, "_apr", "a priori")]
    obs_vars = [(y, "", ", MATS observation"), (fx, "_sim_apr", "forward model simulation based on apriori")]
    atts = {"ret_min_height": ret_alt_range[0], "ret_max_height": ret_alt_range[1], "TP_min_height": tp_alt_range[0],
            "TP_max_height": tp_alt_range[1]}
    write_L2_ncdf(jb, name, atm_vars=atm_vars, obs_vars=obs_vars, tan_alts=tan_alts, atts=atts)


def L2_write_iter(prefix, atm, fx, numit):
    name = f"{prefix}_L2.nc"
    if numit < 0:
        suffix = ""
        long_suffix = "final result"
    else:
        suffix = f"_it_{numit}"
        long_suffix = f"iteration {numit}"
    atm_vars = [(atm, suffix, long_suffix)]
    obs_vars = [(fx, f"_sim{suffix}", f"{long_suffix} forward model simulation")]
    append_L2_ncdf(name, atm_vars=atm_vars, obs_vars=obs_vars)


def write_ncdf_L1b(pdata, outfile, channel, version, im_calibrated=True):
    with nc.Dataset(outfile, 'w') as nf:
        num_images = len(pdata)
        # Global parameters
        nf.date_created = DT.datetime.now().isoformat()
        nf.date_modified = DT.datetime.now().isoformat()
        nf.channel = channel
        nf.data_version = version

        # Create dimensions and their corresponding variables
        # Dimension variable parameters: (<len of variable>, <long name>, <dimension unlimited>)
        params = {"num_images": (num_images, "Image number", True),
                  "im_col": (pdata["NCOL"][0] + 1, "Column number", False),
                  "im_row": (pdata["NROW"][0], "Row number", False)}
        for dim, param in params.items():
            nf.createDimension(dim, None if param[2] else param[0])
            ncdf_create_var(np.arange(param[0]), nf, dim, (dim, ), np.int16, long_name=param[1])

        # Create variable for ImageCalibrated
        if im_calibrated:
            ncdf_create_var(np.stack(pdata["ImageCalibrated"].to_numpy(), axis=0),
                            nf, "ImageCalibrated", ("num_images", "im_row", "im_col"), np.float32)

        # Create variable for CalibrationErrors
        error = np.stack([np.stack(pdata["CalibrationErrors"][i], axis=0)
                          for i in range(len(pdata["CalibrationErrors"]))], axis=0)
        ncdf_create_var(error, nf, "CalibrationErrors", ("num_images", "im_row", "im_col"), np.int16)

        # Create variable for BadColumns
        bad_cols = np.zeros((num_images, params["im_col"][0]))
        for i, bcols in enumerate(pdata['BadColumns'].to_numpy()):
            for j in bcols:
                bad_cols[i, j] = 1
        ncdf_create_var(bad_cols, nf, "BadColumns", ("num_images", "im_col"), np.int8)

        # Handle the remaining variables automatically
        handled_vars = list(nf.variables.keys())
        if not im_calibrated:
            handled_vars.append("ImageCalibrated")
        for var in pdata.keys():
            if var not in handled_vars:
                data = pdata[var].to_numpy()
                ncdf_create_var(data, nf, var, ("num_images", ), type(data.flat[0]))


def ncdf_create_var(data, dset, name, dims, dtype, long_name=None, units=None):
    NCDF_TYPES = {np.int8: 'i1', np.int16: 'i2', np.int32: 'i4', np.int64: 'i8',
                  np.float32: 'f4', np.float64: 'f8', np.bool_: 'i1', str: 'str',
                  pandas._libs.tslibs.timestamps.Timestamp: 'timestamp',
                  np.datetime64: 'timestamp', np.ndarray: 'ndarray'}

    if dtype in NCDF_TYPES.keys():
        type_id = NCDF_TYPES[dtype]
    else:
        raise NotImplementedError(f"Input variable {name} is of type {dtype}, writing this to ncdf is not implemented.")
    if type_id == 'str':
        # dname = f"{name}_string"
        # cdata = nc.stringtochar(np.array(data, dtype=str))
        # dset.createDimension(dname, cdata.shape[-1])
        ncvar = dset.createVariable(name, str, dims)
        ncvar[:] = data
    elif type_id == 'timestamp':
        ncvar = dset.createVariable(name, 'f8', dims)
        ncvar.units = "Seconds since 2000.01.01 00:00 UTC"
        if dtype == np.datetime64:
            ncvar[:] = (data - np.datetime64("2000-01-01 00:00:00.0")) / np.timedelta64(1, "s")
        else:
            ncvar[:] = (data - DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc)) /\
                DT.timedelta(0, 1)
    elif type_id == 'ndarray':
        try:
            size = data[0].shape[0]
            dname = f"{name}_comp"
            if size > 0:
                ctype = type(data[0].flat[0])
                assert np.issubdtype(ctype, np.integer) or ctype == np.float32 or ctype == np.float64
                assert all([len(data[i].shape) == 1 and data[i].shape[0] == size for i in range(data.shape[0])])
                dset.createDimension(dname, size)
                ncvar = dset.createVariable(name, NCDF_TYPES[ctype], (*dims, dname))
                ncvar[:] = np.stack(data, axis=0)
            else:
                assert all([data[i].shape[0] == 0 for i in range(data.shape[0])])
                dset.createDimension(dname, 1)
                ncvar = dset.createVariable(name, 'i1', (*dims, dname))
        except Exception:
            raise NotImplementedError(f"Input variable {name} of type ndarray cannot be imported automatically.")
    else:
        ncvar = dset.createVariable(name, type_id, dims)
        ncvar[:] = data
    if long_name is not None:
        ncvar.long_name = long_name
    if units is not None:
        ncvar.units = units
    return ncvar


def read_multi_ncdf(files, var, offsets=None, numimg=None):
    if offsets is None:
        offsets = np.zeros(len(files))
    else:
        assert len(files) == len(offsets)

    data = {}

    for chn, file in enumerate(files):
        data[chn] = {}

        with nc.Dataset(file, 'r') as nf:
            if chn == 0:
                version = float(nf.getncattr("data_version"))
                if numimg is None:
                    numimg == len(nf[var[0]][:].shape[0])
            else:
                assert float(nf.getncattr("data_version")) == version
            for v in var:
                if len(nf[v][:].shape) > 1:
                    data[chn][v] = nf[v][offsets[chn]:(offsets[chn] + numimg), ...]
                else:
                    data[chn][v] = nf[v][offsets[chn]:(offsets[chn] + numimg)]
    return data, version


def read_chn_from_pandas(df, chn_name, offset, numimg, var):
    dfc = df[df["channel"] == chn_name].copy()
    dfc.reset_index(drop=True, inplace=True)
    res = {}
    for v in var:
        array = dfc[v].to_numpy()
        if type(array[0]) is np.ndarray:
            res[v] = np.stack(array[offset:(offset + numimg)], axis=0)
        else:
            res[v] = array[offset:(offset + numimg)]

    return res


def add_ncdf_vars(file, proto_var, new_vars):
    with nc.Dataset(file, 'r+') as nf:
        dims = nf[proto_var].dimensions
        for name, long_name, data in new_vars:
            if name in nf.variables:
                nf[name][:] = data
                continue
            ncvar = nf.createVariable(name, 'f8', dims)
            if long_name is not None:
                ncvar.long_name = long_name
            ncvar[:] = data

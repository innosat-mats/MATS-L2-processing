import numpy as np
import pandas
import datetime as DT
import netCDF4 as nc
import logging
from mats_l2_processing.util import DT2seconds, geoid_radius


def read_ncdf(fname, vnames, get_units=False):
    res = {}
    with nc.Dataset(fname, 'r') as nf:
        if vnames is None:
            vnames = nf.variables
        for v in vnames:
            res[v] = nf[v][:]
        if get_units:
            return res, {name: nf[name].units for name in vnames}
        else:
            return res
    return res


def read_L1_ncdf(filename, var=None, start_img=None, stop_img=None, start_time=None, stop_time=None,
                 time_var='time', custom_time_format=None, read_ncattrs=True, center_times=False):
    res = {}
    with nc.Dataset(filename, 'r') as nf:
        # Read in global attributes
        size = nf[time_var].shape[0]
        if type(read_ncattrs) is list:
            attrs = read_ncattrs
        else:
            attrs = nf.ncattrs() if read_ncattrs else []
        for attr in attrs:
            res[attr] = nf.getncattr(attr)

        # Handle time variable
        tdata = nf[time_var]
        assert len(nf[time_var].shape) == 1, f"Invalid time variable {time_var} selected."
        if custom_time_format is not None:
            assert len(custom_time_format) == 2 and len(custom_time_format) == 6, "Invalid custom_time_format!"
            time_origin = DT.datetime(*custom_time_format[0], tzinfo=DT.timezone.utc)
            time_step = custom_time_format[1]
        elif hasattr(tdata, "units") and (tdata.units == "Seconds since 2000.01.01 00:00 UTC"):
            time_origin = DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc)
            time_step = 1
        elif hasattr(tdata, "units") and (tdata.units == "nanoseconds since 2000-01-01"):
            time_origin = DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc)
            time_step = 1e-9
        else:
            ValueError(f"Invalid time variable {time_var}!")

        time = time_origin + (time_step * tdata[:]) * DT.timedelta(0, 1)
        if (start_img is not None) or (stop_img is not None):
            start = 0 if start_img is None else start_img
            stop = size if stop_img is None else stop_img
        elif (start_time is not None) or (stop_time is not None):
            start = 0 if start_time is None else np.nanargmin(np.abs(time - start_time))
            stop = size if stop_time is None else np.nanargmin(np.abs(time - stop_time)) + 1
        else:
            start, stop = 0, size

        assert stop > start, "Empty time interval selected for input data! Abort!"

        res["time"] = time[start:stop]
        res["time_s"] = time_step * tdata[start:stop]

        # Handle remaining variables
        if var is None:
            var = nf.variables
        #elif "afsTangentPointECEF" in nf.variables: # Handle legacy L1b file structure
        #    var = list(set(var) - set(["TPECEFx", "TPECEFy", "TPECEFz"]))
        #    var.append("afsTangentPointECEF")

        for name in var:
            v = nf[name]
            if name != time_var:
                if len(v.dimensions) > 0:
                    res[name] = v[start:stop, ...]
                else:
                    res[name] = np.array([v[:] for _ in range(stop - start)])

        #if "TPECEFx" not in var:
        #    for i, name in enumerate(["TPECEFx", "TPECEFy", "TPECEFz"]):
        #        res[name] = res["afsTangentPointECEF"][:, ]

        if center_times:
            res["time_s"] += res["TEXPMS"] * 5e-4
            res["time"] = np.array([time + exposure * 5e-4 * DT.timedelta(0, 1)
                                    for time, exposure in zip(res["time"], res["TEXPMS"])])

        res["size"] = res["time"].shape[0]
        logging.info(f"Read {res['size']} images.")
    return res


def write_gen_ncdf(fname, dim_spec, var_spec, attributes):
    with nc.Dataset(fname, 'w') as nf:
        # Create dimensions and dimension variables
        for dim, spec in dim_spec.items():
            nf.createDimension(dim, len(spec[2]))
            ncvar = nf.createVariable(dim, 'f8', (dim,))
            ncvar.long_name = spec[0]
            if spec[1] is not None:
                ncvar.units = spec[1]
            ncvar[:] = spec[2]

        # Write remaining variables
        for var, spec in var_spec.items():
            ncvar = nf.createVariable(var, 'f8', spec[3])
            ncvar.long_name = spec[0]
            if spec[1] is not None:
                ncvar.units = spec[1]
            ncvar[:] = spec[2]

        # Write attributes
        if len(attributes) > 0:
            nf.setncatts(attributes)


def append_gen_ncdf(fname, var_spec, attributes={}, overwrite=False):
    with nc.Dataset(fname, 'r+') as nf:
        for var, spec in var_spec.items():
            if var in nf.variables:
                if overwrite:
                    ncvar = nf[var]
                else:
                    raise RuntimeError(f"Variable {var} already exists!")
            else:
                ncvar = nf.createVariable(var, 'f8', spec[3])
            ncvar.long_name = spec[0]
            if spec[1] is not None:
                ncvar.units = spec[1]
            ncvar[:] = spec[2]

        # Write attributes
        if len(attributes) > 0:
            nf.setncatts(attributes)


def write_ncdf_L1b(pdata, outfile, channel, version, im_calibrated=True, var=None):
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
            handled_vars = handled_vars + ["ImageCalibrated", "CalibrationErrors", "BadColumns"]

        all_vars = pdata.keys() if var is None else var
        for var in all_vars:
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
        ncvar.units = "seconds since 2000.01.01 00:00 UTC"
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
            chn_version = float(nf.getncattr("data_version")) if "data_version" in nf.ncattrs() else 1.0
            if chn == 0:
                version = chn_version
                if numimg is None:
                    numimg == len(nf[var[0]][:].shape[0])
            else:
                assert chn_version == version
            for v in var:
                if type(nf[v][:]) is not np.ma.core.MaskedArray:
                    data[chn][v] = np.array([nf[v][:]] * numimg)
                elif len(nf[v][:].shape) > 1:
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


def add_ncdf_vars(file, proto_var, new_vars, units=[]):
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
            ncvar.units = nf[proto_var].units
        for v, unit in units:
            nf[v].units = unit


def ncdf_filter_dim(ifile, fdim, keep_idx, ofile, vectorize_scalars=False, idx_type='explicit'):

    with nc.Dataset(ifile, "r") as src, nc.Dataset(ofile, "w") as dst:
        assert fdim in src.dimensions, f"The file {ifile} has no dimension {fdim}!"
        if idx_type == 'interval':
            assert len(keep_idx) == 2, "idx_type is 'interval', so keep_idx should have length 2 (start/end of intv.)"
            keep_idx = range(keep_idx[0], keep_idx[1])
        elif idx_type != 'explicit':
            raise ValueError("idx_type must be 'explicit' or 'interval'!")

        # Copy global attributes
        dst.setncatts(src.__dict__)

        # Copy dimensions
        for name, dim in src.dimensions.items():
            if name == fdim:  # the dimension you're filtering
                dst.createDimension(name, len(keep_idx))
            else:
                dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

        # Copy variables
        for name, var in src.variables.items():
            if vectorize_scalars and len(var.dimensions) == 0:
                dims = (fdim,)
                vals = np.full(src.dimensions[fdim].size, var[:])
            else:
                dims = var.dimensions
                vals = var[:]
            dst.createVariable(name, var.datatype, dims)
            dst[name].setncatts(var.__dict__)  # copy attributes

            if fdim in var.dimensions:
                axis = var.dimensions.index(fdim)
                dst[name][:] = np.take(vals, keep_idx, axis=axis)
            elif len(var.dimensions) == 0:
                dst[name][0] = var[0]
            else:
                dst[name][:] = var[:]

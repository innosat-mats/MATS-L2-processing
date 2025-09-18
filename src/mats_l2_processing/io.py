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
                 time_var='EXPDate', read_ncattrs=True):
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
        if (start_img is not None) or (stop_img is not None):
            start = 0 if start_img is None else start_img
            stop = size if stop_img is None else stop_img
        elif (start_time is not None) or (stop_time is not None):
            assert len(nf[time_var].shape) == 1, f"Invalid time variable {time_var} selected."
            time = DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc) +\
                nf[time_var][:] * DT.timedelta(0, 1)
            start = 0 if start_time is None else np.nanargmin(np.abs(time - start_time))
            stop = size if stop_time is None else np.nanargmin(np.abs(time - stop_time)) + 1
            print("start/stop from times")
        else:
            start, stop = 0, size

        #real_stop = stop if stop >= 0 else len(time) + stop
        #assert start < real_stop, f"The selected time interval has no images! Read data for {time[0]} to {time[-1]}"
        #if start_time < time[0] - DT.timedelta(1, 0) or stop_time > time[-1] + DT.timedelta(1, 0):
        #    print("WARNING: Data in the input file does not completely cover the selected time interval.")

        if var is None:
            var = nf.variables

        for name in var:
            v = nf[name]
            if hasattr(v, "units") and (v.units == "Seconds since 2000.01.01 00:00 UTC"):
                res[v.name] = DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc) +\
                    v[start:stop, ...] * DT.timedelta(0, 1)
                res[f"{v.name}_s"] = v[start:stop, ...]
            elif hasattr(v, "units") and (v.units == "nanoseconds since 2000-01-01"):
                res[v.name] = DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc) +\
                    (v[start:stop, ...].data * 1e-9) * DT.timedelta(0, 1)
                res[f"{v.name}_s"] = v[start:stop, ...]
            else:
                if len(v.dimensions) > 0:
                    res[name] = v[start:stop, ...]
                else:
                    res[name] = np.array([v[:] for _ in range(stop - start)])

        res["size"] = res[var[0]].shape[0]
        print(var[0], start, stop)
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


def append_gen_ncdf(fname, var_spec, attributes={}):
    with nc.Dataset(fname, 'r+') as nf:
        for var, spec in var_spec.items():
            ncvar = nf.createVariable(var, 'f8', spec[3])
            ncvar.long_name = spec[0]
            if spec[1] is not None:
                ncvar.units = spec[1]
            ncvar[:] = spec[2]

        # Write attributes
        if len(attributes) > 0:
            nf.setncatts(attributes)


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
            handled_vars = handled_vars + ["ImageCalibrated", "CalibrationErrors", "BadColumns"]
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
        for v, unit in units:
            nf[v].units = unit

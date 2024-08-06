import numpy as np
from numba import jit
import datetime as DT
import netCDF4 as nc
import logging


def geoid_radius(latitude):
    '''
    Function from GEOS5 class.
    GEOID_RADIUS calculates the radius of the geoid at the given latitude
    [Re] = geoid_radius(latitude) calculates the radius of the geoid (km)
    at the given latitude (degrees).
    ----------------------------------------------------------------
            Craig Haley 11-06-04
    ---------------------------------------------------------------
    '''
    DEGREE = np.pi / 180.0
    EQRAD = 6378.14 * 1000
    FLAT = 1.0 / 298.257
    Rmax = EQRAD
    Rmin = Rmax * (1.0 - FLAT)
    Re = np.sqrt(1./(np.cos(latitude)**2/Rmax**2
                + np.sin(latitude)**2/Rmin**2))
    return Re

def sph2cart(r, phi, theta):
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z

@jit(nopython=True,cache=True)
def cart2sph(pos):
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    longitude = np.arctan2(y, x)
    latitude = np.arcsin(z / radius)

    return radius,longitude,latitude


def localgrid_to_lat_lon_alt_3D(radius_grid,acrosstrack_grid,alongtrack_grid,ecef_to_local):
    arrayshape = (len(radius_grid),len(acrosstrack_grid),len(alongtrack_grid))
    non_uniform_ecef_grid_x = np.zeros(arrayshape)
    non_uniform_ecef_grid_y = np.zeros(arrayshape)
    non_uniform_ecef_grid_z = np.zeros(arrayshape)

    non_uniform_ecef_grid_altitude = np.zeros(arrayshape)
    non_uniform_ecef_grid_r = np.zeros(arrayshape)
    non_uniform_ecef_grid_lat = np.zeros(arrayshape)
    non_uniform_ecef_grid_lon = np.zeros(arrayshape)

    for i in range(arrayshape[0]):
        for j in range(arrayshape[1]):
            for k in range(arrayshape[2]):
                non_uniform_ecef_grid_x[i,j,k],non_uniform_ecef_grid_y[i,j,k],non_uniform_ecef_grid_z[i,j,k]=ecef_to_local.inv().apply(sph2cart(radius_grid[i],acrosstrack_grid[j],alongtrack_grid[k]))
                ecef_r_lon_lat = cart2sph(np.array([[non_uniform_ecef_grid_x[i,j,k],non_uniform_ecef_grid_y[i,j,k],non_uniform_ecef_grid_z[i,j,k]]]))
                non_uniform_ecef_grid_r[i,j,k] = ecef_r_lon_lat[0][0]
                non_uniform_ecef_grid_lon[i,j,k] = ecef_r_lon_lat[1][0]
                non_uniform_ecef_grid_lat[i,j,k] = ecef_r_lon_lat[2][0]
                non_uniform_ecef_grid_altitude[i,j,k] = non_uniform_ecef_grid_r[i,j,k]-geoid_radius(non_uniform_ecef_grid_lat[i,j,k])

    return non_uniform_ecef_grid_altitude,non_uniform_ecef_grid_lon,non_uniform_ecef_grid_lat,non_uniform_ecef_grid_r


def center_grid(grid):
    return (grid[:-1] + grid[1:]) / 2


def get_alt_lat_lon(radius_grid, acrosstrack_grid, alongtrack_grid, ecef_to_local):
    rr, acrr, alongg = np.meshgrid(radius_grid, acrosstrack_grid, alongtrack_grid, indexing="ij")
    lxx, lyy, lzz = sph2cart(rr, acrr, alongg)
    glgrid = ecef_to_local.inv().apply(np.dstack((lxx.flatten(), lyy.flatten(), lzz.flatten()))[0, :, :])
    latt = np.arcsin(glgrid[:, 2].reshape(rr.shape) / rr)
    altt = rr - geoid_radius(latt)
    latt = np.rad2deg(latt)[0, :, :]
    lonn = np.rad2deg(np.arctan2(glgrid[:, 1], glgrid[:, 0]).reshape(rr.shape))[0, :, :]
    return altt, latt, lonn


def DT2seconds(dts):
    return (dts - DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc)) / DT.timedelta(0, 1)


def write_aux(jb, y, fx, tan_alts, atm_apr, prefix):
    # Centre grid (the working grid for Jacobian contains grid edges)
    #rad_grid_c, alongtrack_grid_c, acrosstrack_grid_c = [center_grid(jb[x]) for x in
    #                                                     ["rad_grid", "alongtrack_grid", "acrosstrack_grid"]]
    # Create alt/lat/lon grid for grid centres
    # alt, lon, lat, rr = localgrid_to_lat_lon_alt_3D(altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local)
    #alts, lats, lons = get_alt_lat_lon(rad_grid_c, acrosstrack_grid_c,
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

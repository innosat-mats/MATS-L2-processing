import numpy as np
import pandas
import datetime as DT
import netCDF4 as nc
import logging
from mats_l2_processing.util import DT2seconds, geoid_radius


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
    # Legacy
    fxg = fx.reshape((2, -1, jb["data"]["NCOL"][0], jb["data"]["NROW"][0]))
    # sp.save_npz(f"{args.out_np}_K.npz", K)
    np.savez_compressed(f"{prefix}_iter.npz", fx=fxg, VER=atm[0], T=atm[1], alongtrack_coord=jb["alongtrack_grid_c"],
                        acrosstrack_coord=jb["acrosstrack_grid_c"], altitude=jb["alt"])


def write_L2_ncdf(jb, outfile, atm_vars=[], obs_vars=[], tan_alts=None, atts=[]):
    #  Legacy
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
    # Legacy
    name = f"{prefix}_L2.nc"
    atm_vars = [(atm_apr, "_apr", "a priori")]
    obs_vars = [(y, "", ", MATS observation"), (fx, "_sim_apr", "forward model simulation based on apriori")]
    atts = {"ret_min_height": ret_alt_range[0], "ret_max_height": ret_alt_range[1], "TP_min_height": tp_alt_range[0],
            "TP_max_height": tp_alt_range[1]}
    write_L2_ncdf(jb, name, atm_vars=atm_vars, obs_vars=obs_vars, tan_alts=tan_alts, atts=atts)


def L2_write_iter(prefix, atm, fx, numit):
    # Legacy
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

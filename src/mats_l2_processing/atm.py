import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator, CloughTocher2DInterpolator
from mats_l2_processing.regularisation import Sa_inv_tikhonov
from mats_l2_processing.util import multiprocess, get_image, running_mean, geoid_radius
# from mats_l2_processing.forward_model import calc_K_image_1D
# from mats_l2_processing.inverse_model import limit_alt
# from mats_l2_processing.oem import mkl_solve_1D
from mats_utils.geolocation.coordinates import col_heights
from matplotlib import pyplot as plt
import sys
import netCDF4 as nc


def get_background(grid, mean_time, clim_data, t_apr=None, full_ver=None):
    alt, lat, lon = grid.alt / 1000, grid.lat, grid.lon
    month = mean_time.month

    # ver = 2e3 * 1e6 * stats.norm.pdf(alt, 88, 4.5) + 2e2 * 1e6 * np.exp(-(alt - 60) / 20)
    T, o2 = [np.zeros_like(alt) for c in range(2)]
    so2 = clim_data.o2.sel(month=month)
    sT = clim_data.T.sel(month=month)

    # min_lat, max_lat = np.min(jb["lat"]), np.max(jb["lat"])
    # center_lat_range = (2 * min_lat + max_lat) / 3, (min_lat + 2 * max_lat) / 3
    # lat_averaged = ver_apr["VER"][0, ...].copy()

    # lat_averaged[:, ver_apr["latitude"] < center_lat_range[0]] = np.nan
    # lat_averaged[:, ver_apr["latitude"] > center_lat_range[1]] = np.nan
    # lat_averaged = np.broadcast_to(np.nanmean(lat_averaged, axis=1)[:, np.newaxis], ver_apr["VER"][0, ...].shape)
    # sver = RectBivariateSpline(ver_apr["altitude"], ver_apr["latitude"], lat_averaged)
    if t_apr is not None:
        sT = RectBivariateSpline(t_apr["altitude"], t_apr["latitude"], t_apr["T"][0, ...])
    # sT = RectBivariateSpline(t_apr["z"], t_apr["lat"], t_apr["T"][0, ...])

    for i in range(T.shape[1]):
        for j in range(T.shape[2]):
            # T[:, i, j] = sT(alt[:, i, j], lat[i, j])[:, 0]
            T[:, i, j] = sT.interp(lat=lat[i, j], z=alt[:, i, j])
            o2[:, i, j] = so2.interp(lat=lat[i, j], z=alt[:, i, j])
            # ver[:, i, j] = sver(alt[:, i, j], lat[i, j])[:, 0]
    o2 *= 1e-6

    # Experimental sideload of VER apriori from result:
    if full_ver is not None:
        with nc.Dataset(full_ver, 'r') as nf:
            vert = nf["VER"][0, :, :, :].data
        assert vert.shape == T.shape, f"VER:{vert.shape}, T:{T.shape}"
        ver = vert.copy()
        assert not np.isnan(ver).any(), "ver a priori has NaN's!"

    assert not np.isnan(T).any(), "T a priori has NaN's!"
    assert not np.isnan(o2).any(), "o2 a priori has NaN's!"
    return o2, T, None if full_ver is None else ver


def plot_profiles(alt, ver, idxs, fname):
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    for idx in idxs:
        ax.plot(ver[idx, :], alt[idx, :] * 1e-3, label=f"Image {idx}")
    ax.set_xscale('linear')
    plt.xlabel("VER")
    plt.ylabel("Altitude, km")
    plt.legend()
    plt.savefig(f"{fname}.png", dpi=400)


def plot_map(alt, ver, fname, along_coord=None):
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    if along_coord is None:
        along_coord = np.arange(0, alt.shape[0])
        xlabel = "Profile number"
    else:
        xlabel = "Alongtrack coordinate"
    pc = ax.pcolor(np.broadcast_to(along_coord[:, np.newaxis], alt.shape), alt * 1e-3, ver, cmap="inferno")
    ax.set_xscale('linear')
    plt.xlabel(xlabel)
    plt.ylabel("Altitude, km")
    plt.colorbar(pc)
    plt.savefig(f"{fname}.png", dpi=400)


def VER_1D(geo_3d, data_3d, mean_time, clim_data, rt_data, conf, columns, processes):

    # Initialize geometry
    geo = initialize_1D_geo(geo_3d, conf, columns)

    # Prepare observations
    obs_data = np.zeros((geo["size"], len(columns), len(geo["rows"])))
    hwidth = int((conf.MEDCOLS_1D_APR - 1) / 2)
    for i, c in enumerate(columns):
        col_range = [int(np.where(geo_3d["columns"] == c)[0][0] + pad) for pad in [-hwidth, hwidth + 1]]
        obs_data[:, i, :] = np.median(data_3d[conf.CHANNEL_1D_APR][:, :, col_range[0]:col_range[1]], axis=2)
    obs_data *= 1e13

    # Prepare temperature and o2 data
    month = mean_time.month
    so2 = clim_data.o2.sel(month=month)
    sT = clim_data.T.sel(month=month)
    T, o2, alts = [np.zeros((obs_data.shape[0], len(geo["rad_grid_c"]))) for c in range(3)]
#    alt = geo["rad_grid_c"][np.newaxis, :] - np.array([geoid_radius(np.deg2rad(lat))
#                                                       for lat in geo["data"]["TPlat"]])[:, np.newaxis]
    for j in range(obs_data.shape[0]):
        T[j, :] = sT.interp(lat=geo["data"]["TPlat"][j], z=geo["alt"][j, :] * 1e-3)
        o2[j, :] = so2.interp(lat=geo["data"]["TPlat"][j], z=geo["alt"][j, :] * 1e-3)
    o2 *= 1e-6
    # Generate regularisation matrices
    Sainv, _ = Sa_inv_tikhonov([geo["rad_grid_c"]], conf.SA_WEIGHTS_1D_APR[0],
                               diff_weights=[conf.SA_WEIGHTS_1D_APR[1]], volume_factors=True,
                               store_terms=True, aspect_ratio=conf.ASPECT_RATIO)
    Seinv = sp.diags(np.ones(len(geo["rows"])), 0).astype('float32') / (conf.RAD_SCALE ** 2 * len(geo["rows"]))
    xsc = conf.VER_SCALE  # np.ones(len(geo["rows"]))

    # Calculate tangent alts and valid altitude range
    tan_alts = np.zeros_like(obs_data)
    for j in range(tan_alts.shape[0]):
        image = get_image(geo["data"], j, geo["image_vars"])
        for i, col in enumerate(geo["columns"]):
            colh = col_heights(image, col, 10, spline=True)
            tan_alts[j, i, :] = np.array(colh(geo["rows"]))

    y_ar, valid_alt = limit_alt(geo, obs_data.reshape(-1, obs_data.shape[-1])[np.newaxis, ...],
                                tan_alts.reshape(-1, obs_data.shape[-1]), conf.TP_ALT_RANGE)
    y_ar = y_ar.reshape(tan_alts.shape)
    valid_alt = valid_alt.reshape(tan_alts.shape)

    # Generate Jacobians
    jac_cv = ["edges", "is_regular_grid", "columns", "rows", "ecef_to_local", "rad_grid", "stepsize", "top_alt",
              "excludeK", "timescale"]
    jac = multiprocess(calc_K_image_1D, geo["data"], geo["image_vars"], processes,
                       [geo[x] for x in jac_cv] + [xsc, o2, [np.zeros_like(T), T], rt_data, valid_alt])

    ver0 = np.zeros(len(geo["rad_grid_c"]))
    res = np.zeros_like(T)
    for j in range(len(jac)):
        res[j, :] = mkl_solve_1D(ver0, ver0, y_ar[j, 0, :], jac[j][0], jac[j][1], Sainv, Seinv)
    res *= xsc

    plot_profiles(geo["alt"], res, list(range(0, 91, 10)), "VER_apr_profiles")
    plot_map(geo["alt"], res, "VER_apr_map")

    return res, np.array([jac[j][2] for j in range(len(jac))]), geo


def apr_from_1D(geo, geo3d, ver_1D, along_dists, decay_range=None, hw=4):
    stack = ver_1D.copy()
    stack = np.maximum(stack, 0.0)

    # Get rid of spurious top-altitude maxima
    if decay_range is not None:
        decay_range = [x * 1e3 for x in decay_range]
        stack = np.where(geo["alt"] > decay_range[1], 0.0, stack)
        stack = np.where(np.logical_and(geo["alt"] < decay_range[1], geo["alt"] > decay_range[0]),
                         stack * ((geo["alt"] - decay_range[1]) / (decay_range[1] - decay_range[0])) ** 2, stack)
    plot_map(geo["alt"], stack, "VER_apr_map_decayed")

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


    plot_map(geo3d["alt"][:, 19, :].T, res[:, 19, :].T, "VER_apr_map_slice", along_coord=geo3d["alongtrack_grid_c"])
    return res

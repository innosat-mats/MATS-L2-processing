# %%
import numpy as np
from mats_utils.geolocation.coordinates import col_heights  # , findheight
from scipy.spatial.transform import Rotation as R
import scipy.sparse as sp
from skyfield import api as sfapi
from skyfield.framelib import itrs
from mats_l2_processing.grids import get_los_ecef, get_steps_in_local_grid
from mats_l2_processing.util import multiprocess, print_times
from sorcery import dict_of
import time
from itertools import product, chain
import logging
# %%


def prepare_profiles(ch, vnames, col, rows):
    cs = col_heights(ch, col, 10, spline=True)
    tanalts = np.array(cs(rows))
    profiles = [np.array(np.stack(ch[vname])[rows, col]) * 1e13 for vname in vnames]
    return profiles, tanalts


def select_data(df, num_profiles, start_index=0):
    if num_profiles + start_index > len(df):
        raise ValueError('Selected profiles out of bounds')
    df = df.loc[start_index:start_index + num_profiles - 1]
    df = df.reset_index(drop=True)


def grad_path2grid(pathGrad, gridVar, posidx, cumulative=False):
    res = np.zeros((pathGrad.shape[0], *gridVar.shape))
    for idx in range(pathGrad.shape[1]):
        res[:, posidx[0][idx], posidx[1][idx], posidx[2][idx]] += pathGrad[:, idx]
    return res


def grad_path2grid_weights(pathGrad, gridVar, pWeights, cumulative=False):
    res = np.zeros((pathGrad.shape[0], *gridVar.shape))
    it = np.nditer(pWeights[1], flags=["multi_index"])
    for w in it:
        coord = pWeights[0][it.multi_index[0], it.multi_index[1], :]
        res[:, coord[0], coord[1], coord[2]] += w * pathGrad[:, it.multi_index[0]]
    return res


def calc_rad2(pos, path_step, o2s, atm, rt_data, edges):
    interppos_func = interppos_trilinear
    # times = [time.time()]
    # titles = []
    VER, Temps = [np.array(arr) for arr in atm]
    startT = np.linspace(100, 600, 501)
    pathtemps, o2, pathVER, pWeights = interppos_func(pos, [Temps, o2s, VER], edges)
    # ##  o2, _ = interppos_func(pos, o2s, edges)
    # ##  pathVER, _ = interppos_func(pos, VER, edges)
    # times.append(time.time())
    # titles.append("Path interpolation")
    sigmas, sigmas_pTgrad, emissions, emissions_pTgrad = interp_T(pathtemps, startT, [rt_data[name] for name in
        ["sigma", "sigma_grad", "emission", "emission_grad"]])
    # #emissions, emissions_pTgrad = interp_and_grad_T(pathtemps, startT, rt_data["emission"], rt_data["emission_grad"])
    # times.append(time.time())
    # titles.append("Temp interpolation")
    exp_tau = np.exp(-(sigmas * o2).cumsum(axis=1) * path_step * 1e2) * (path_step / 4 / np.pi * 1e6)
    del sigmas
    # times.append(time.time())
    # titles.append("exp_tau")
    # #emissions, emissions_pTgrad = interp_and_grad_T(pathtemps, startT, rt_data["emission"], rt_data["emission_grad"])
    # #times.append(time.time())
    # #titles.append("Emission interpolation")
    grad_Temps = grad_path2grid_weights(rt_data["filters"] @ (exp_tau * emissions_pTgrad) * pathVER, Temps, pWeights)
    # times.append(time.time())
    # titles.append("Temp gradiant (emissions)")
    del emissions_pTgrad
    path_tau_em = exp_tau * emissions
    grad_Temps -= grad_path2grid_weights(rt_data["filters"] @ (np.flip(np.cumsum(np.flip(path_tau_em * pathVER, axis=1), axis=1), axis=1) * sigmas_pTgrad) * o2, Temps, pWeights)
    # times.append(time.time())
    # titles.append("Temp gradient (tau)")
    del sigmas_pTgrad, o2
    path_tau_em = rt_data["filters"] @ path_tau_em
    res = np.sum(path_tau_em * pathVER, axis=1)
    grad_VER = grad_path2grid_weights(path_tau_em, VER, pWeights)
    # times.append(time.time())
    # titles.append("VER gradient")
    # print_times(times, titles)
    return res, grad_VER, grad_Temps


def calc_rad_ng(pos, path_step, o2s, atm, rt_data, edges):
    interppos_func = interppos_trilinear
    VER, Temps = [np.array(arr) for arr in atm]
    startT = np.linspace(100, 600, 501)
    pathtemps, o2, pathVER, _ = interppos_func(pos, [Temps, o2s, VER], edges)
    # ## o2, _ = interppos_func(pos, o2s, edges)
    # ## pathVER, _ = interppos_func(pos, VER, edges)
    sigmas, emissions = interp_T(pathtemps, startT, [rt_data[name] for name in ["sigma", "emission"]])
    exp_tau = np.exp(-(sigmas * o2).cumsum(axis=1) * path_step * 1e2) * (path_step / 4 / np.pi * 1e6)
    del sigmas
    path_tau_em = rt_data["filters"] @ (exp_tau * emissions)
    res = np.sum(path_tau_em * pathVER, axis=1)
    return res


def interp_T(x, xs, ys):
    step = xs[1] - xs[0]
    ix = np.array(np.floor((x - xs[0]) / step), dtype=int)
    w0 = (xs[ix + 1] - x) / step
    w1 = (x - xs[ix]) / step
    return [w0 * y[ix, :].T + w1 * y[ix + 1, :].T for y in ys]


def interppos_regular_nearest(pos, inArray, edges, edge_steps):
    iz, iy, ix = [np.array(np.floor((pos[:, i] - edges[i][0])[:, :, :, np.newaxis] / edge_steps[i]),
                           dtype=int) for i in range(3)]
    iw = np.ones_like(iz)
    # for c in range(3):
    #     print(c, edges[c].shape, edges[c][0], np.diff(edges[c]).mean(), edges[c][-1])
    #     print(c, pos[:, c])
    return inArray[iz, iy, ix], (iz, iy, ix, iw)


def interppos_rectilinear_nearest(pos, inArray, edges):
    crd = np.stack([np.searchsorted(edges[i], pos[:, i], sorter=None) - 1 for i in range(3)], axis=-1)[:, np.newaxis, :]
    return inArray[crd[:, 0, 0], crd[:, 0, 1], crd[:, 0, 2]], (crd, np.ones((pos.shape[0], 1)))


def interppos_trilinear(pos, data, edges):
    num_pos = pos.shape[0]
    coords0 = np.stack([np.searchsorted(edges[i], pos[:, i], sorter=None) - 1 for i in range(3)], axis=-1)

    coords, dists, iw = np.zeros((num_pos, 8, 3), dtype=int), np.zeros((num_pos, 3, 2)), np.zeros((num_pos, 8))
    dists[:, :, 1] = np.stack([pos[:, i] - edges[i][coords0[:, i]] for i in range(3)], axis=-1)
    dists[:, :, 0] = np.stack([edges[i][coords0[:, i] + 1] - pos[:, i] for i in range(3)], axis=-1)

    idata = np.stack(data, axis=0)
    res = np.zeros((len(data), num_pos))

    norms = np.prod(dists[:, :, 0] + dists[:, :, 1], axis=1)
    for i, idx in enumerate(product([0, 1], repeat=3)):
        for j in range(3):
            coords[:, i, j] = coords0[:, j] + idx[j]
        iw[:, i] = dists[:, 0, idx[0]] * dists[:, 1, idx[1]] * dists[:, 2, idx[2]]
        res += iw[:, i] * idata[:, coords[:, i, 0], coords[:, i, 1], coords[:, i, 2]]
    res /= norms[np.newaxis, :]
    return *[res[i, :] for i in range(len(data))], (coords, iw / norms[:, np.newaxis])


def calc_los_image(image, common_args):
    # tic = time.time()
    columns, rows, ecef_to_local, timescale, top_alt, stepsize = common_args
    rot_sat_channel = R.from_quat(image['qprime'])  # Rotation between satellite pointing and channel pointing
    q = image['afsAttitudeState']  # Satellite pointing in ECI
    rot_sat_eci = R.from_quat(np.roll(q, -1))  # Rotation matrix for q (should go straight to ecef?)

    current_ts = timescale.from_datetime(image["EXPDate"])
    localR = np.linalg.norm(sfapi.wgs84.latlon(image["TPlat"], image["TPlon"], elevation_m=0).at(current_ts).position.m)
    eci_to_ecef = R.from_matrix(itrs.rotation_at(current_ts))
    satpos_eci = image['afsGnssStateJ2000'][0:3]
    satpos_ecef = eci_to_ecef.apply(satpos_eci)
    los = []
    for column in columns:
        los_col = []
        for row in rows:
            los_ecef = get_los_ecef(image, column, row, rot_sat_channel, rot_sat_eci, eci_to_ecef)
            posecef_i_sph, weights = get_steps_in_local_grid(image, dict_of(columns, rows, ecef_to_local, timescale,
                                                                            top_alt, stepsize),
                                                             satpos_ecef, los_ecef, localR=localR, do_abs=False)
            los_col.append(posecef_i_sph)
        los.append(los_col.copy())
    # toc = time.time()
    # logging.log(15, f"LOS calculated for image {image['num_image']} in {toc - tic:.3f} s.")
    return los


def test_grid(jb, processes):
    logging.info("Calculating LOS...")
    _ = multiprocess(calc_los_image, jb["data"], jb["image_vars"], processes,
                     [jb["columns"], jb["rows"], jb["ecef_to_local"], jb["timescale"], jb["top_alt"], jb["stepsize"]])
    logging.info("Geometry successfully verified!")


def calc_obs(jb, processes):
    res = multiprocess(calc_obs_image, jb["data"], jb["image_vars"], processes,
                       [jb["columns"], jb["rows"]], unzip=True)
    return np.stack(res[:2], axis=0), res[2]


def calc_obs_image(image, common_args):
    tic = time.time()
    columns, rows = common_args
    image_profiles = [[], []]
    image_tanalts = []
    for column in columns:
        s_profiles, s_tanalt = prepare_profiles(image, ("IR1c", "IR2c"), column, rows)
        for i in range(2):
            image_profiles[i].append(s_profiles[i])
        image_tanalts.append(s_tanalt)
    toc = time.time()
    logging.log(15, f"Observations processed for image {image['num_image']} in {toc - tic:.3f} s.")
    return image_profiles[0], image_profiles[1], image_tanalts


def calc_K(jb, rt_data, o2, atm, valid_obs, processes, fx_only=False):
    res = multiprocess(calc_K_image, jb["data"], jb["image_vars"], processes,
                       [jb["edges"], jb["is_regular_grid"], jb["columns"], jb["rows"], jb["ecef_to_local"],
                        jb["rad_grid"], jb["alongtrack_grid"], jb["acrosstrack_grid"], jb["stepsize"], jb["top_alt"],
                        jb["excludeK"], jb['timescale'], o2, atm, rt_data, valid_obs, fx_only], unzip=False)

    K, fx = [], []
    for i in range(2):
        fx.append(np.array(list(chain.from_iterable([profiles[i] for _, profiles in res]))))
        if not fx_only:
            K.append(sp.vstack([k_part[i] for k_part, _ in res]))
    return None if fx_only else sp.vstack(K), np.stack(fx, axis=0)


def calc_K_image(image, common_args):
    edges, is_regular_grid, columns, rows, ecef_to_local, rad_grid, alongtrack_grid, acrosstrack_grid, stepsize, \
        top_alt, excludeK, timescale, o2, atm, rt_data, valid_obs, fx_only = common_args

    image_calc_profiles = [[], []]
    image_K = []
    valid_obs_im = valid_obs[image["num_image"], :, :]

    image["los"] = calc_los_image(image, (columns, rows, ecef_to_local, timescale, top_alt, stepsize))

    if fx_only:
        pname = "Forward model"
    else:
        pname = "Jacobian"
        for i in range(2):
            image_K.append(sp.lil_array((len(rows) * len(columns),
                           2 * (len(rad_grid) - 1) * (len(alongtrack_grid) - 1) * (len(acrosstrack_grid) - 1))))

    image_k_row = 0
    tic = time.time()
    for i, column in enumerate(columns):
        for j, row in enumerate(rows):
            if valid_obs_im[i, j]:
                if fx_only:
                    ircalc = calc_rad_ng(image["los"][i][j], stepsize, o2, atm, rt_data, edges)
                else:
                    ircalc, VERgrad, TEMPgrad = calc_rad2(image["los"][i][j], stepsize, o2, atm, rt_data, edges)
                    if excludeK is not None:
                        VERgrad[:, excludeK] = 0.0
                        TEMPgrad[:, excludeK] = 0.0
                    for ch in range(2):
                        k_row_c = np.hstack([VERgrad[ch, ...].reshape(-1), TEMPgrad[ch, ...].reshape(-1)])
                        image_K[ch][image_k_row, :] = k_row_c.reshape(-1)
            else:
                ircalc = [0.0, 0.0]
            image_k_row += 1
            for ch in range(2):
                image_calc_profiles[ch].append(ircalc[ch])
        if i == 0:
            toc = time.time()
            logging.log(15, f"{pname}: Started image {image['num_image']}, ETA {(toc - tic) * len(columns):.3f} s.")
    toc = time.time()
    logging.log(15, f"{pname}: Image {image['num_image']} processed in {toc-tic:.1f} s.")
    return image_K, image_calc_profiles

# %%
import numpy as np
from mats_utils.geolocation.coordinates import col_heights  # , findheight
from mats_l1_processing.pointing import pix_deg
from scipy.spatial.transform import Rotation as R
# from scipy.interpolate import CubicSpline, interp1d
import scipy.sparse as sp
from skyfield import api as sfapi
from skyfield.framelib import itrs
from mats_l2_processing.grids import cart2sph, sph2cart, geoid_radius
from fast_histogram import histogramdd
from bisect import bisect_left
# import boost_histogram as bh
import mygrad as mg
import time
from itertools import chain, repeat
from multiprocessing import Pool
# import jax.numpy as jnp
# from jax import grad, jit, value_and_grad
# from bisect import bisect_left

# %%


def load_abstable():
    global abstable_tan_height
    global abstable_distance
    factors_path = "/home/olemar/Projects/Universitetet/MATS/MATS-L2-processing/data/splinedlogfactorsIR2.npy"
    # factors_path = "/home/lk/mats-analysis/MATS-L2-processing/data/splinedlogfactorsIR2.npy"
    abstable_tan_height, abstable_distance = np.load(factors_path, allow_pickle=True)
    return


def generate_timescale():
    global timescale
    timescale = sfapi.load.timescale()
    return


def generate_stepsize():
    global stepsize
    # stepsize = 100
    stepsize = 1000
    return


def generate_top_alt():
    global top_alt
    top_alt = 120e3
    return

# %%


def prepare_profile(ch, col=None, rows=None):
    """Extract data and tanalt for image

    Detailed description

    Args:
        single column
        range of rows

    Returns:
        profile, tanalts
    """

    image = np.stack(ch[vname])
    if col is None:
        col = int(ch['NCOL']/2)
    if rows.any() == None:
        rows = np.arrange(0, ch['NROW'])

    cs = col_heights(ch, col, 10, spline=True)
    tanalt = np.array(cs(rows))
    profile = np.array(image[rows, col])*1e15
    return profile, tanalt


def prepare_profiles(ch, vnames, col, rows):
    cs = col_heights(ch, col, 10, spline=True)
    tanalts = np.array(cs(rows))
    profiles = [np.array(np.stack(ch[vname])[rows, col]) * 1e13 for vname in vnames]
    return profiles, tanalts


def eci_to_ecef_transform(date):
    return R.from_matrix(itrs.rotation_at(timescale.from_datetime(date)))


def select_data(df, num_profiles, start_index = 0):
    if num_profiles + start_index > len(df):
        raise ValueError('Selected profiles out of bounds')
    df = df.loc[start_index:start_index+num_profiles-1]
    df = df.reset_index(drop=True)


def find_top_of_atmosphere(top_altitude,localR,satpos,losvec):
    sat_radial_pos = np.linalg.norm(satpos)
    los_zenith_angle = np.arccos(np.dot(satpos,losvec)/sat_radial_pos)
    # solving quadratic equation to find distance start and end of atmosphere 
    b = 2*sat_radial_pos*np.cos(los_zenith_angle)
    root=np.sqrt(b**2+4*((top_altitude+localR)**2 - sat_radial_pos**2))
    distance_top_1 =(-b-root)/2
    distance_top_2 =(-b+root)/2

    return [distance_top_1,distance_top_2]


def generate_steps(stepsize,top_altitude,localR,satpos,losvec):
    distance_top_of_atmosphere = find_top_of_atmosphere(top_altitude,localR,satpos,losvec)
    steps = np.arange(distance_top_of_atmosphere[0], distance_top_of_atmosphere[1], stepsize)
    return steps


def generate_local_transform(data):
    """Calculate the transform from ecef to the local coordinate system.

    Detailed descriptiongg

    Args:
        df (pandas dataframe): data.

    Returns:
        ecef_to_local
    """
    first = 0
    mid = int((data["size"] - 1) / 2)
    last = data["size"] - 1

    eci_to_ecef = eci_to_ecef_transform(data['EXPDate'][mid])

    posecef_first = eci_to_ecef.apply(data["afsTangentPointECI"][first, :]).astype('float32')
    posecef_mid = eci_to_ecef.apply(data["afsTangentPointECI"][mid, :]).astype('float32')
    posecef_last = eci_to_ecef.apply(data["afsTangentPointECI"][last, :]).astype('float32')

    observation_normal = np.cross(posecef_first, posecef_last)
    observation_normal = observation_normal / np.linalg.norm(observation_normal)  # normalize vector

    posecef_mid_unit = posecef_mid / np.linalg.norm(posecef_mid)  # unit vector for central position
    ecef_to_local = R.align_vectors([[1, 0, 0], [0, 1, 0]], [posecef_mid_unit, observation_normal])[0]

    return ecef_to_local


def get_los_ecef(image,icol,irow,rot_sat_channel,rot_sat_eci,eci_to_ecef):
    x, y = pix_deg(image, icol, irow) #Angles for single pixel
    rotation_channel_pix = R.from_euler('xyz', [0, y, x], degrees=True).apply([1, 0, 0]) #Rot matrix between pixel pointing and channels pointing
    los_eci = np.array(rot_sat_eci.apply(rot_sat_channel.apply(rotation_channel_pix)))
    los_ecef = eci_to_ecef.apply(los_eci)
    return los_ecef


def get_steps_in_local_grid(image,ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False):

    if localR is None:
        date = image['EXPDate']
        current_ts = timescale.from_datetime(date)
        localR = np.linalg.norm(sfapi.wgs84.latlon(image["TPlat"], image["TPlon"], elevation_m=0).at(current_ts).position.m)

    s_steps = generate_steps(stepsize,top_altitude=top_alt,localR=localR,satpos=satpos_ecef,losvec=los_ecef)

    posecef=(np.expand_dims(satpos_ecef, axis=0).T+s_steps*np.expand_dims(los_ecef, axis=0).T).astype('float32')
    poslocal = ecef_to_local.apply(posecef.T)  # convert to local (for middle alongtrack measurement)
    poslocal_sph = cart2sph(poslocal)
    poslocal_sph=np.array(poslocal_sph).T
    if do_abs:
        weights = get_weights(poslocal_sph,s_steps,localR)
    else:
        weights = np.ones((poslocal_sph.shape[0]))

    return poslocal_sph,weights


def get_weights(posecef_sph,s_steps,localR):
    #Calculate the weights 
    minarg=posecef_sph[:,0].argmin() #find tangent point
    distances=(s_steps-s_steps[minarg])/1000 #distance to tangent point (km)
    target_tanalt = (posecef_sph[minarg,0]-localR)/1000 #value of tangent altitude
    lowertan=bisect_left(abstable_tan_height,target_tanalt)-1 #index in table of value below tangent altitude
    uppertan=lowertan+1 #index in table of value above tangent altitude
    absfactor_distance_below=abstable_distance[lowertan](distances) #extract values below
    absfactor_distance_above=abstable_distance[uppertan](distances) #extract values above
    #make interpolater that generates optical depth for each distance as a function of tanalt
    interpolated_abs_factor=interp1d([abstable_tan_height[lowertan],abstable_tan_height[uppertan]],np.array([absfactor_distance_below,absfactor_distance_above]).T)
    #get transmissivity for generated optical depth vector
    weights=np.exp(interpolated_abs_factor(target_tanalt))
    return weights


def generate_grid(data, columns, rows, ecef_to_local, grid_proto=None):
    lims = grid_limits(data, columns, rows, ecef_to_local)
    print(lims)
    if grid_proto is None:
        grid_proto = [lims[i][2] for i in range(3)]
    result = []
    for i in range(3):
        if (type(grid_proto[i]) is int) or (type(grid_proto[i]) is float):
            grid_len = int(grid_proto[i])
            assert grid_len > 0, "Malformed grid_spec parameter!"
            result.append(np.linspace(lims[i][0], lims[i][1], grid_len))
        elif isinstance(grid_proto[i], np.ndarray):
            assert len(grid_proto[i].shape) == 1
            # for j, spec in enumerate(grid_spec[i]):
            #    assert type(spec) is tuple, "Malformed grid_spec parameter!"
            #    assert len(spec) == 3, "Malformed grid_spec parameter!"
            #    if j == 0:
            #        axis_grid = np.linspace(*spec)
            #    else:
            #        axis_grid = np.concatenate((axis_grid, np.linspace(*spec)))
            result.append(grid_proto[i][np.logical_and(grid_proto[i] > lims[i][0], grid_proto[i] < lims[i][1])])
        else:
            raise ValueError(f"Malformed grid_spec parameter: {type(grid_proto[i])}")
    return result


def grid_limits(data, columns, rows, ecef_to_local):

    first = 0
    mid = int((data["size"] - 1) / 2)
    last = data["size"] - 1

    mid_date = data['EXPDate'][mid]
    current_ts = timescale.from_datetime(mid_date)
    localR = np.linalg.norm(sfapi.wgs84.latlon(data["TPlat"][mid], data["TPlon"][mid], elevation_m=0).at(current_ts).position.m)

    rot_sat_channel = R.from_quat(data['qprime'][mid, :]) # Rotation between satellite pointing and channel pointing

    q = data['afsAttitudeState'][mid, :]  # Satellite pointing in ECI
    rot_sat_eci = R.from_quat(np.roll(q, -1))  # Rotation matrix for q (should go straight to ecef?)

    eci_to_ecef=R.from_matrix(itrs.rotation_at(current_ts))
    satpos_eci = data['afsGnssStateJ2000'][mid, 0:3]
    satpos_ecef=eci_to_ecef.apply(satpos_eci)


    #change to do only used columns and rows
    if len(columns)==1:
        mid_col = columns[0]
    else:
        left_col = columns[0]
        mid_col = int(data["NCOL"][0] / 2) - 1
        right_col = data["NCOL"][0] - 1

    irow = rows[0]
    poslocal_sph = []

    get_los_vars = ['CCDSEL', 'NCSKIP', 'NCBINCCDColumns', 'NRSKIP', 'NRBIN', 'NCOL', 'EXPDate', 'TPlat', 'TPlon']
    # Which localR to use in get_step_local_grid?
    for idx in [first, mid, last]:
        image = get_image(data, idx, get_los_vars)
        los_ecef = get_los_ecef(image, mid_col, irow, rot_sat_channel,rot_sat_eci,eci_to_ecef)
        poslocal_sph.append(get_steps_in_local_grid(image, ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False)[0])


    if len(columns) > 1:
        for col in [left_col, right_col]:
            for idx in [first, mid, last]:
                image = get_image(data, idx, get_los_vars)
                los_ecef = get_los_ecef(image, col, irow, rot_sat_channel, rot_sat_eci,
                                        eci_to_ecef)
                poslocal_sph.append(get_steps_in_local_grid(image, ecef_to_local,
                                    satpos_ecef, los_ecef, localR=None, do_abs=False)[0])

    poslocal_sph = np.vstack(poslocal_sph)
    max_rad = poslocal_sph[:,0].max()
    min_rad = poslocal_sph[:,0].min()
    max_along = poslocal_sph[:,2].max()
    min_along = poslocal_sph[:,2].min()
    max_across = poslocal_sph[:,1].max()
    min_across = poslocal_sph[:,1].min()
    if len(columns) < 2:
        max_across = max_across + 0.2
        min_across = min_across - 0.2

    nalt = int(len(rows / 2)) + 2
    nacross = int(len(columns / 2)) + 2
    nalong = int(data["size"] / 2) + 2

    if (nacross-2)<2:
        nacross=2
    if (nalong-2)<2:
        nalong=2

    # altitude_grid = np.linspace(min_rad-10e3,localR+top_alt+10e3,nalt)
    # acrosstrack_grid = np.linspace(min_across-0.1,max_across+0.1,nacross)
    # alongtrack_grid = np.linspace(min_along-0.2,max_along+0.2,nalong)

    return (min_rad - 10e3, localR + top_alt + 10e3, nalt), (min_across - 0.1, max_across + 0.1, nacross), \
        (min_along - 0.6, max_along + 0.6, nalong)


def center_grid(grid):
    return (grid[:-1] + grid[1:]) / 2


def get_image(data, idx, var):
    res = {"num_image": idx}
    for v in var:
        if len(data[v].shape) > 1:
            res[v] = data[v][idx, ...]
        else:
            res[v] = data[v][idx]
    return res


def calc_jacobian(df, columns, rows, edges=None, grid_proto=None, processes=4):
    # -*- coding: utf-8 -*-
    """Calculate Jacobian.

    Detailed description

    Args:
        df (pandas dataframe): data.
        coluns (array): columns to use.
        rows: rows to use

    Returns:
        y,
        K,
        altitude_grid, (change to edges)
        alongtrack_grid,
        acrosstrack_grid,
        ecef_to_local

    """
    generate_timescale()  # generates a global timescale object
    generate_stepsize()
    generate_top_alt()
    load_abstable()
    ecef_to_local = generate_local_transform(df)

    if edges is None:
        altitude_grid, acrosstrack_grid, alongtrack_grid = generate_grid(df, columns, rows, ecef_to_local, grid_proto)
        edges = [altitude_grid, acrosstrack_grid, alongtrack_grid]

    is_regular_grid = True
    for axis_edges in edges:
        widths = np.diff(axis_edges)
        if not all(np.abs(widths - np.mean(widths)) < 0.001 * np.abs(np.mean(widths))):
            is_regular_grid = False
            break
    do_abs = (df['channel'][0] == "IR2")
    if do_abs:
        print("Warning: absorbtion disabled for selected channel (not implemented yet).")
    per_image_args = [(i, df.loc[i], df.iloc[i], df['EXPDate'][i]) for i in range(len(df))]
    common_args = (edges, is_regular_grid, do_abs, columns, rows,
                   ecef_to_local, altitude_grid, alongtrack_grid, acrosstrack_grid)
    time0 = time.time()
    with Pool(processes=processes) as pool:
        results = pool.starmap(image_jacobian, zip(per_image_args, repeat(common_args)))
    time1 = time.time()
    print("Assembling sparse Jacobian matrix...")
    K = sp.vstack([k_part for k_part, _, _ in results])
    y = np.array(list(chain.from_iterable([profiles for _, profiles, _ in results])))
    tan_alts = np.array(list(chain.from_iterable([profiles for _, _, profiles in results])))

    time2 = time.time()
    print(f"Jacobian contribution from {len(df)} images calculated in {time1 - time0:.1f} s.")
    print(f"Results assembled to a sparse matrix in {time2 - time1:.1f} s.")
    return y, K, altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local, tan_alts


def image_jacobian(per_image_arg, common_args):
    i, loc, iloc, expDate = per_image_arg
    edges, grid_is_regular, do_abs, columns, rows, \
        ecef_to_local, altitude_grid, alongtrack_grid, acrosstrack_grid = common_args

    print(f"Processing of image {i} started.")
    tic = time.time()

    image_profiles = []
    image_tanalts = []
    image_K = sp.lil_array((len(rows) * len(columns),
                           (len(altitude_grid) - 1) * (len(alongtrack_grid) - 1) * (len(acrosstrack_grid) - 1)))
    image_k_row = 0

    rot_sat_channel = R.from_quat(loc['qprime'])  # Rotation between satellite pointing and channel pointing
    q = loc['afsAttitudeState']  # Satellite pointing in ECI
    rot_sat_eci = R.from_quat(np.roll(q, -1))  # Rotation matrix for q (should go straight to ecef?)

    current_ts = timescale.from_datetime(expDate)
    localR = np.linalg.norm(sfapi.wgs84.latlon(loc.TPlat, loc.TPlon, elevation_m=0).at(current_ts).position.m)
    print(localR)
    eci_to_ecef = R.from_matrix(itrs.rotation_at(current_ts))
    satpos_eci = loc['afsGnssStateJ2000'][0:3]
    satpos_ecef = eci_to_ecef.apply(satpos_eci)
    for column in columns:
        s_profile, s_tanalt = prepare_profile(iloc, column, rows)
        image_profiles.append(s_profile)
        image_tanalts.append(s_tanalt)
        for irow in rows:
            los_ecef = get_los_ecef(loc, column, irow, rot_sat_channel, rot_sat_eci, eci_to_ecef)
            posecef_i_sph, weights = get_steps_in_local_grid(loc, ecef_to_local, satpos_ecef, los_ecef, localR,
                                                             do_abs=do_abs)
            if grid_is_regular:
                hist = histogramdd(posecef_i_sph[::1, :], weights=weights,
                    range=[[edges[0][0], edges[0][-1]], [edges[1][0], edges[1][-1]], [edges[2][0], edges[2][-1]]],
                    bins=[len(edges[0]) - 1, len(edges[1]) - 1, len(edges[2]) - 1])
            else:
                hist, _ = np.histogramdd(posecef_i_sph[::1, :], bins=edges, weights=weights)
            image_K[image_k_row, :] = hist.reshape(-1)
            image_k_row += 1

    toc = time.time()
    print(f"Image {i} processed in {toc-tic:.1f} s.")
    return image_K, image_profiles, image_tanalts


def ir1fun(pos, path_step, o2s, atm, rt_data, edges):
    VER, Temps = atm
    VER = np.array(VER)
    Temps = np.array(Temps)
    pos = np.array(pos)
    startT = np.linspace(100, 600, 501)
    pathtemps = interppos(pos, Temps, edges)
    # print(pathtemps)
    sigmas = interpT(pathtemps, startT, rt_data["sigma"])
    # print(sigmas.max())
    emissions = interpT(pathtemps, startT, rt_data["emission"])
    o2 = interppos(pos, o2s, edges)
    tau = (sigmas * o2).cumsum(axis=1) * path_step * 1e2  # m -> cm
    # print(tau)
    VERs = interppos(pos, VER) * path_step * 1e2  # m -> cm
    res = rt_data["filters"] @ (np.exp(-tau) * VERs * emissions)
    return res[0].sum() / 4 / np.pi * 1e4  # cm-2-> m-2


def ir2fun(pos, path_step, o2s, atm, rt_data, edges):
    # return total LOS intensity in the filter ph m-2 sr-1 s-1
    VER, Temps = atm
    startT = np.linspace(100, 600, 501)
    pathtemps = interppos(pos, Temps, edges)
    # print(pathtemps)
    sigmas = interpT(pathtemps, startT, rt_data["sigma"])
    # print(sigmas.max())
    emissions = interpT(pathtemps, startT, rt_data["emission"])
    o2 = interppos(pos, o2s, edges)
    tau = (sigmas * o2).cumsum(axis=1) * path_step * 1e2  # m -> cm
    # print(tau)
    VERs = interppos(pos, VER) * path_step * 1e2  # m -> cm
    res = rt_data["filters"] @ (np.exp(-tau) * VERs * emissions)
    return res[1].sum() / 4 / np.pi * 1e4  # cm-2-> m-2


def grad_path2grid(pathGrad, gridVar, posidx, cumulative=False):
    res = np.zeros((pathGrad.shape[0], *gridVar.shape))
    for idx in range(pathGrad.shape[1]):
        res[:, posidx[0][idx], posidx[1][idx], posidx[2][idx]] += pathGrad[:, idx]
    return res


def calc_rad(pos, path_step, o2s, atm, rt_data, edges):
    VER, Temps = [np.array(arr) for arr in atm]
    startT = np.linspace(100, 600, 501)
    pathtemps, posidxs = interppos(pos, Temps, edges)
    o2, _ = interppos(pos, o2s, edges)
    pathVER, _ = interppos(pos, VER, edges)
    sigmas, sigmas_pTgrad = interp_and_grad_T(pathtemps, startT, rt_data["sigma"], rt_data["sigma_grad"])
    emissions, emissions_pTgrad = interp_and_grad_T(pathtemps, startT, rt_data["emission"], rt_data["emission_grad"])
    exp_tau = np.exp(-(sigmas * o2).cumsum(axis=1)) * (path_step / 4 / np.pi * 1e6)
    del sigmas
    grad_Temps = grad_path2grid(rt_data["filters"] @ (pathVER * exp_tau * emissions_pTgrad), Temps, posidxs)
    del emissions_pTgrad
    path_tau_em = pathVER * exp_tau * emissions
    grad_Temps -= grad_path2grid(rt_data["filters"] @ (np.flip(np.cumsum(np.flip(path_tau_em, axis=1), axis=1), axis=1) * sigmas_pTgrad * o2), Temps, posidxs)
    del sigmas_pTgrad, o2
    res = np.sum(rt_data["filters"] @ path_tau_em, axis=1)
    del path_tau_em
    # rad = rt_data["filters"] @ (pathVER * tau_em)
    grad_VER = grad_path2grid(rt_data["filters"] @ (exp_tau * emissions), VER, posidxs)
    # grad_pTemps_emi = rt_data["filters"] @ (pathVER * exp_tau * emissions_pTgrad)
    # grad_Temps_tau = grad_path2grid(rt_data["filters"] @ (np.flip(np.cumsum(np.flip(pathVER * exp_tau * emissions, axis=1), axis=1), axis=1) * sigmas_pTgrad * o2), Temps, posidxs)

    return res, grad_VER, grad_Temps
    # del res, VER, Temps
    # print("calc_rad done!")
    # return ret


def calc_rad2(pos, path_step, o2s, atm, rt_data, edges):
    # times = [time.time()]
    # titles = []
    VER, Temps = [np.array(arr) for arr in atm]
    startT = np.linspace(100, 600, 501)
    pathtemps, posidxs = interppos(pos, Temps, edges)
    o2, _ = interppos(pos, o2s, edges)
    pathVER, _ = interppos(pos, VER, edges)
    # times.append(time.time())
    # titles.append("Path interpolation")
    sigmas, sigmas_pTgrad, emissions, emissions_pTgrad = interp_and_grad_T(pathtemps, startT,
        [rt_data[name] for name in ["sigma", "sigma_grad", "emission", "emission_grad"]])
    # emissions, emissions_pTgrad = interp_and_grad_T(pathtemps, startT, rt_data["emission"], rt_data["emission_grad"])
    # times.append(time.time())
    # titles.append("Temp interpolation")

    exp_tau = np.exp(-(sigmas * o2).cumsum(axis=1) * path_step * 1e2) * (path_step / 4 / np.pi * 1e6)
    del sigmas
    # times.append(time.time())
    # titles.append("exp_tau")
    #emissions, emissions_pTgrad = interp_and_grad_T(pathtemps, startT, rt_data["emission"], rt_data["emission_grad"])
    # times.append(time.time())
    # titles.append("Emission interpolation")
    grad_Temps = grad_path2grid(rt_data["filters"] @ (exp_tau * emissions_pTgrad) * pathVER, Temps, posidxs)
    # times.append(time.time())
    # titles.append("Temp gradiant (emissions)")
    del emissions_pTgrad
    path_tau_em = exp_tau * emissions
    grad_Temps -= grad_path2grid(rt_data["filters"] @ (np.flip(np.cumsum(np.flip(path_tau_em * pathVER, axis=1), axis=1), axis=1) * sigmas_pTgrad) * o2, Temps, posidxs)
    # times.append(time.time())
    # titles.append("Temp gradient (tau)")
    del sigmas_pTgrad, o2
    path_tau_em = rt_data["filters"] @ path_tau_em
    res = np.sum(path_tau_em * pathVER, axis=1)
    grad_VER = grad_path2grid(path_tau_em, VER, posidxs)
    # times.append(time.time())
    # titles.append("VER gradient")
    #print_times(times, titles)
    return res, grad_VER, grad_Temps


def print_times(times, titles):
    assert len(titles) + 1 == len(times)
    res = ""
    for i, title in enumerate(titles):
        res += f"{title}: {times[i + 1] - times[i]:.3f} s, "
    print(res)


def interp_and_grad_T(x, xs, ys):
    ix = np.array(np.floor(x - 100), dtype=int)
    # return ys[ix, :].T
    # return ((x - xs[ix - 1]) * ys[ix, :].T + (xs[ix] - x) * ys[ix - 1, :].T) / (xs[ix] - xs[ix - 1])
    #return ((x - xs[ix - 1]) * ys[ix, :].T + (xs[ix] - x) * ys[ix - 1, :].T)
    return [y[ix, :].T for y in ys]


def interp_and_grad_T2(x, xs, ys):
    ix = np.array(np.floor(x - 100), dtype=int)
    # return ys[ix, :].T
    # return ((x - xs[ix - 1]) * ys[ix, :].T + (xs[ix] - x) * ys[ix - 1, :].T) / (xs[ix] - xs[ix - 1])
    #return ((x - xs[ix - 1]) * ys[ix, :].T + (xs[ix] - x) * ys[ix - 1, :].T)
    return [y[ix, :] for y in ys]



def interppos(pos, inArray, edges):
    iz = np.array(np.floor((pos[:, 0] - edges[0][0]) / np.diff(edges[0]).mean()), dtype=int)
    iy = np.array(np.floor((pos[:, 1] - edges[1][0]) / np.diff(edges[1]).mean()), dtype=int)
    ix = np.array(np.floor((pos[:, 2] - edges[2][0]) / np.diff(edges[2]).mean()), dtype=int)
    # for c in range(3):
    #     print(c, edges[c].shape, edges[c][0], np.diff(edges[c]).mean(), edges[c][-1])
    #     print(c, pos[:, c])
    return inArray[iz, iy, ix], (iz, iy, ix)


def get_alt_lat_lon(radius_grid, acrosstrack_grid, alongtrack_grid, ecef_to_local):
    rr, acrr, alongg = np.meshgrid(radius_grid, acrosstrack_grid, alongtrack_grid, indexing="ij")
    lxx, lyy, lzz = sph2cart(rr, acrr, alongg)
    glgrid = ecef_to_local.inv().apply(np.dstack((lxx.flatten(), lyy.flatten(), lzz.flatten()))[0, :, :])
    latt = np.arcsin(glgrid[:, 2].reshape(rr.shape) / rr)
    altt = rr - geoid_radius(latt)
    latt = np.rad2deg(latt)[0, :, :]
    lonn = np.rad2deg(np.arctan2(glgrid[:, 1], glgrid[:, 0]).reshape(rr.shape))[0, :, :]
    return altt, latt, lonn


def initialize_T_jacobian(data, columns, rows, grid_proto=None):
    generate_timescale()  # generates a global timescale object
    generate_stepsize()
    generate_top_alt()

    # A "Jacobian base" dictionary to store atmosphere-independent jacobian elements
    jb = {"data": data, "columns": columns, "rows": rows}
    jb["ecef_to_local"] = generate_local_transform(data)
    jb["edges"] = generate_grid(data, columns, rows, jb["ecef_to_local"], grid_proto)
    jb["rad_grid"], jb["acrosstrack_grid"], jb["alongtrack_grid"] = jb["edges"]
    jb["alt"], jb["lat"], jb["lon"] = \
        get_alt_lat_lon(*[center_grid(jb[name]) for name in
                        ["rad_grid", "acrosstrack_grid", "alongtrack_grid"]], jb["ecef_to_local"])

    is_regular_grid = True
    for axis_edges in jb["edges"]:
        widths = np.diff(axis_edges)
        if not all(np.abs(widths - np.mean(widths)) < 0.001 * np.abs(np.mean(widths))):
            is_regular_grid = False
            break
    jb["is_regular_grid"] = is_regular_grid
    return jb


def calc_T_jacobian(jb, rt_data, o2, atm, processes=4):
    # -*- coding: utf-8 -*-
    """Calculate Jacobian.

    Detailed description

    Args:
        df (pandas dataframe): data.
        coluns (array): columns to use.
        rows: rows to use

    Returns:
        y,
        K,/localR
        altitude_grid, (change to edges)
        alongtrack_grid,
        acrosstrack_grid,
        ecef_to_local

    """
    # per_image_args = [(i, jb["df"].loc[i], jb["df"].iloc[i], jb["df"]['EXPDate'][i]) for i in range(len(jb["df"]))]
    image_vars = ['CCDSEL', 'NCSKIP', 'NCBINCCDColumns', 'NRSKIP', 'NRBIN', 'NCOL', 'NROW',
                  'EXPDate', 'qprime', 'afsGnssStateJ2000', 'afsAttitudeState',
                  "IR1c", "IR2c", "TPlon", "TPlat", "TEXPMS"]
    per_image_args = [get_image(jb["data"], i, image_vars) for i in range(jb["data"]["size"])]
    common_args = (jb["edges"], jb["is_regular_grid"], jb["columns"], jb["rows"], jb["ecef_to_local"], jb["rad_grid"],
                   jb["alongtrack_grid"], jb["acrosstrack_grid"], o2, atm, rt_data)
    time0 = time.time()
    assert processes >= 1
    if processes == 1:
        results = [[], [], [], []]
        for image in per_image_args:
            res_image = image_T_jacobian(image, common_args)
            for el in range(4):
                results[el].append(res_image[el])
    else:
        with Pool(processes=processes) as pool:
            results = pool.starmap(image_T_jacobian, zip(per_image_args, repeat(common_args)))
    time1 = time.time()
    print("Assembling sparse Jacobian matrix...")
    K, y, fx = [], [], []
    for i in range(2):
        K.append(sp.vstack([k_part[i] for k_part, _, _, _ in results]))
        y.append(np.array(list(chain.from_iterable([profiles[i] for _, profiles, _, _ in results]))))
        fx.append(np.array(list(chain.from_iterable([profiles[i] for _, _, profiles, _ in results]))))
    tan_alts = np.array(list(chain.from_iterable([profiles for _, _, _, profiles in results])))

    time2 = time.time()
    print(f"Jacobian contribution from {jb['data']['size']} images calculated in {time1 - time0:.3f} s.")
    print(f"Results assembled to a sparse matrix in {time2 - time1:.3f} s.")
    return np.stack(y, axis=0), sp.vstack(K), np.stack(fx, axis=0), tan_alts


def image_T_jacobian(image, common_args):
    # i, loc, iloc, expDate = per_image_arg
    edges, grid_is_regular, columns, rows, ecef_to_local, altitude_grid, \
        alongtrack_grid, acrosstrack_grid, o2, atm, rt_data = common_args

    print(f"Processing of image {image['num_image']} started.")
    tic = time.time()

    image_profiles = [[], []]
    image_calc_profiles = [[], []]
    image_tanalts = []
    image_K = []
    for i in range(2):
        image_K.append(sp.lil_array((len(rows) * len(columns),
                       2 * (len(altitude_grid) - 1) * (len(alongtrack_grid) - 1) * (len(acrosstrack_grid) - 1))))
    image_k_row = 0

    rot_sat_channel = R.from_quat(image['qprime'])  # Rotation between satellite pointing and channel pointing
    q = image['afsAttitudeState']  # Satellite pointing in ECI
    rot_sat_eci = R.from_quat(np.roll(q, -1))  # Rotation matrix for q (should go straight to ecef?)

    current_ts = timescale.from_datetime(image["EXPDate"])
    localR = np.linalg.norm(sfapi.wgs84.latlon(image["TPlat"], image["TPlon"], elevation_m=0).at(current_ts).position.m)
    eci_to_ecef = R.from_matrix(itrs.rotation_at(current_ts))
    satpos_eci = image['afsGnssStateJ2000'][0:3]
    satpos_ecef = eci_to_ecef.apply(satpos_eci)

    for column in columns:
        # column_start = time.time()
        s_profiles, s_tanalt = prepare_profiles(image, ("IR1c", "IR2c"), column, rows)
        for i in range(2):
            image_profiles[i].append(s_profiles[i])
        image_tanalts.append(s_tanalt)
        # profiles_done = time.time()
        toc = time.time()
        print(f"Image {image['num_image']}, starting on column {column} of {len(columns)}, {toc - tic:.3f} s so far.")
        for irow in rows:
            # print(f"Starting row {irow} ...")
            # row_start = time.time()
            los_ecef = get_los_ecef(image, column, irow, rot_sat_channel, rot_sat_eci, eci_to_ecef)
            posecef_i_sph, weights = get_steps_in_local_grid(image, ecef_to_local, satpos_ecef, los_ecef, localR,
                                                             do_abs=False)
            # geometry_done = time.time()
            ircalc, VERgrad, TEMPgrad = calc_rad2(posecef_i_sph, stepsize, o2, atm, rt_data, edges)
            # rt_done = time.time()
            for ch in range(2):
                k_row_c = np.hstack([VERgrad[ch, ...].reshape(-1), TEMPgrad[ch, ...].reshape(-1)])
                image_K[ch][image_k_row, :] = k_row_c.reshape(-1)
                image_calc_profiles[ch].append(ircalc[ch])
            # K_amended = time.time()
            # print(f"Row {irow} done:")
            # print(f"Geometry: {geometry_done-row_start:.3f} s, RT: {rt_done - geometry_done:.3f} s," +
            #      f"matrix write:  {K_amended - rt_done:.3f} s")
        image_k_row += 1

    toc = time.time()
    print(f"Image {image['num_image']} processed in {toc-tic:.1f} s.")
    return image_K, image_profiles, image_calc_profiles, image_tanalts


# %%

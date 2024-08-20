import numpy as np
from mats_l1_processing.pointing import pix_deg
from mats_l2_processing.util import get_image
from scipy.spatial.transform import Rotation as R
from skyfield import api as sfapi
from skyfield.framelib import itrs
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
    # DEGREE = np.pi / 180.0
    EQRAD = 6378.14 * 1000
    FLAT = 1.0 / 298.257
    Rmax = EQRAD
    Rmin = Rmax * (1.0 - FLAT)
    Re = np.sqrt(1. / (np.cos(latitude)**2 / Rmax**2 + np.sin(latitude)**2 / Rmin**2))
    return Re


def sph2cart(r, phi, theta):
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def cart2sph(pos):
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    longitude = np.arctan2(y, x)
    latitude = np.arcsin(z / radius)

    return radius, longitude, latitude


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


def make_grid_proto(proto, offset=0.0, scaling=1.0):
    if (type(proto) is int) or (type(proto) is float):
        return int(proto)
    assert type(proto) is list, "Malformed grid specification!"
    if type(proto[0]) is tuple:
        for i, interval in enumerate(proto):
            assert type(interval) is tuple, "Malformed grid specification!"
            assert len(interval) == 3, "Malformed grid specification!"
            if i == 0:
                res = np.arange(*interval)
            else:
                res = np.concatenate((res, np.arange(*interval)))
    else:
        res = np.array(proto)
    return res * scaling + offset


def generate_grid(data, grid_def, grid_proto=None):
    # columns, rows, ecef_to_local, top_alt, stepsize, timescale
    lims = grid_limits(data, grid_def)
    # print(lims)
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
            result.append(grid_proto[i][np.logical_and(grid_proto[i] > lims[i][0], grid_proto[i] < lims[i][1])])
        else:
            raise ValueError(f"Malformed grid_spec parameter: {type(grid_proto[i])}")
    return result


def get_los_ecef(image, icol, irow, rot_sat_channel, rot_sat_eci, eci_to_ecef):
    # Angles for single pixel
    x, y = pix_deg(image, icol, irow)
    # Rot matrix between pixel pointing and channels pointing
    rotation_channel_pix = R.from_euler('xyz', [0, y, x], degrees=True).apply([1, 0, 0])
    los_eci = np.array(rot_sat_eci.apply(rot_sat_channel.apply(rotation_channel_pix)))
    los_ecef = eci_to_ecef.apply(los_eci)
    return los_ecef


def get_steps_in_local_grid(image, grid_def, satpos_ecef, los_ecef, localR=None, do_abs=False):

    if localR is None:
        date = image['EXPDate']
        current_ts = grid_def["timescale"].from_datetime(date)
        localR = np.linalg.norm(sfapi.wgs84.latlon(image["TPlat"], image["TPlon"],
                                                   elevation_m=0).at(current_ts).position.m)

    s_steps = generate_steps(grid_def["stepsize"], grid_def["top_alt"], localR, satpos_ecef, los_ecef)

    posecef = (np.expand_dims(satpos_ecef, axis=0).T + s_steps * np.expand_dims(los_ecef, axis=0).T).astype('float32')
    poslocal = grid_def["ecef_to_local"].apply(posecef.T)  # convert to local (for middle alongtrack measurement)
    poslocal_sph = cart2sph(poslocal)
    poslocal_sph = np.array(poslocal_sph).T
    if do_abs:
        raise NotImplementedError("This method of absorbtion calculation is no longer in use")
        # weights = get_weights(poslocal_sph, s_steps, localR)
    else:
        weights = np.ones((poslocal_sph.shape[0]))

    return poslocal_sph, weights


def grid_limits(data, grid_def):

    first = 0
    mid = int((data["size"] - 1) / 2)
    last = data["size"] - 1

    mid_date = data['EXPDate'][mid]
    current_ts = grid_def["timescale"].from_datetime(mid_date)
    localR = np.linalg.norm(sfapi.wgs84.latlon(data["TPlat"][mid], data["TPlon"][mid],
                                               elevation_m=0).at(current_ts).position.m)

    rot_sat_channel = R.from_quat(data['qprime'][mid, :])  # Rotation between satellite pointing and channel pointing

    q = data['afsAttitudeState'][mid, :]  # Satellite pointing in ECI
    rot_sat_eci = R.from_quat(np.roll(q, -1))  # Rotation matrix for q (should go straight to ecef?)

    eci_to_ecef = R.from_matrix(itrs.rotation_at(current_ts))
    satpos_eci = data['afsGnssStateJ2000'][mid, 0:3]
    satpos_ecef = eci_to_ecef.apply(satpos_eci)

    # change to do only used columns and rows
    if len(grid_def["columns"]) == 1:
        mid_col = grid_def["columns"][0]
    else:
        left_col = grid_def["columns"][0]
        right_col = grid_def["columns"][-1]
        mid_col = int((left_col + right_col) / 2)
        # mid_col = int(data["NCOL"][0] / 2) - 1
        # right_col = data["NCOL"][0] - 1

    irow = grid_def["rows"][0]
    poslocal_sph = []

    get_los_vars = ['CCDSEL', 'NCSKIP', 'NCBINCCDColumns', 'NRSKIP', 'NRBIN', 'NCOL', 'EXPDate', 'TPlat', 'TPlon']
    # Which localR to use in get_step_local_grid?
    for idx in [first, mid, last]:
        image = get_image(data, idx, get_los_vars)
        los_ecef = get_los_ecef(image, mid_col, irow, rot_sat_channel, rot_sat_eci, eci_to_ecef)
        poslocal_sph.append(get_steps_in_local_grid(image, grid_def, satpos_ecef, los_ecef, localR=None)[0])

    if len(grid_def["columns"]) > 1:
        for col in [left_col, right_col]:
            for idx in [first, mid, last]:
                image = get_image(data, idx, get_los_vars)
                los_ecef = get_los_ecef(image, col, irow, rot_sat_channel, rot_sat_eci,
                                        eci_to_ecef)
                poslocal_sph.append(get_steps_in_local_grid(image, grid_def, satpos_ecef, los_ecef, localR=None)[0])

    poslocal_sph = np.vstack(poslocal_sph)
    # max_rad = poslocal_sph[:, 0].max()
    min_rad = poslocal_sph[:, 0].min()
    max_along = poslocal_sph[:, 2].max()
    min_along = poslocal_sph[:, 2].min()
    max_across = poslocal_sph[:, 1].max()
    min_across = poslocal_sph[:, 1].min()
    if len(grid_def["columns"]) < 2:
        max_across = max_across + 0.2
        min_across = min_across - 0.2

    nalt = int(len(grid_def["rows"] / 2)) + 2
    nacross = int(len(grid_def["columns"] / 2)) + 2
    nalong = int(data["size"] / 2) + 2

    if (nacross - 2) < 2:
        nacross = 2
    if (nalong - 2) < 2:
        nalong = 2

    return (min_rad - 10e3, localR + grid_def["top_alt"] + 10e3, nalt), (min_across - 0.1, max_across + 0.1, nacross), \
        (min_along - 0.6, max_along + 0.6, nalong)


def generate_steps(stepsize, top_altitude, localR, satpos, losvec):
    distance_top_of_atmosphere = find_top_of_atmosphere(top_altitude, localR, satpos, losvec)
    steps = np.arange(distance_top_of_atmosphere[0], distance_top_of_atmosphere[1], stepsize)
    return steps


def find_top_of_atmosphere(top_altitude, localR, satpos, losvec):
    sat_radial_pos = np.linalg.norm(satpos)
    los_zenith_angle = np.arccos(np.dot(satpos, losvec) / sat_radial_pos)
    # solving quadratic equation to find distance start and end of atmosphere
    b = 2 * sat_radial_pos * np.cos(los_zenith_angle)
    root = np.sqrt(b**2 + 4 * ((top_altitude + localR)**2 - sat_radial_pos**2))
    distance_top_1 = (-b - root) / 2
    distance_top_2 = (-b + root) / 2

    return [distance_top_1, distance_top_2]


# def eci_to_ecef_transform(date):
#     return R.from_matrix(itrs.rotation_at(timescale.from_datetime(date)))


def generate_local_transform(data, timescale):
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

    # eci_to_ecef = eci_to_ecef_transform(data['EXPDate'][mid])
    eci_to_ecef = R.from_matrix(itrs.rotation_at(timescale.from_datetime(data['EXPDate'][mid])))

    posecef_first = eci_to_ecef.apply(data["afsTangentPointECI"][first, :]).astype('float32')
    posecef_mid = eci_to_ecef.apply(data["afsTangentPointECI"][mid, :]).astype('float32')
    posecef_last = eci_to_ecef.apply(data["afsTangentPointECI"][last, :]).astype('float32')

    observation_normal = np.cross(posecef_first, posecef_last)
    observation_normal = observation_normal / np.linalg.norm(observation_normal)  # normalize vector

    posecef_mid_unit = posecef_mid / np.linalg.norm(posecef_mid)  # unit vector for central position
    ecef_to_local = R.align_vectors([[1, 0, 0], [0, 1, 0]], [posecef_mid_unit, observation_normal])[0]

    return ecef_to_local


def initialize_geometry(data, columns, rows, conf, grid_proto=None, processes=1):
    logging.info("Initializing 3-D grid...")
    grid_def = {"columns": columns, "rows": rows, "stepsize": conf.STEP_SIZE, "top_alt": conf.TOP_ALT}
    grid_def["timescale"] = sfapi.load.timescale()
    grid_def["ecef_to_local"] = generate_local_transform(data, grid_def["timescale"])

    # A "Jacobian base" dictionary to store atmosphere-independent jacobian elements
    geo = {"data": data}
    geo["edges"] = generate_grid(data, grid_def, grid_proto=grid_proto)
    geo.update(grid_def)

    # jb["rad_grid"], jb["acrosstrack_grid"], jb["alongtrack_grid"] = jb["edges"]
    cnames = ["rad_grid", "acrosstrack_grid", "alongtrack_grid"]
    for i, name in enumerate(cnames):
        geo[name] = geo["edges"][i]
        geo[f"{name}_c"] = center_grid(geo["edges"][i])
    geo["alt"], geo["lat"], geo["lon"] = get_alt_lat_lon(geo["rad_grid_c"], geo["acrosstrack_grid_c"],
                                                         geo["alongtrack_grid_c"], geo["ecef_to_local"])

    geo["is_regular_grid"] = True
    for axis_edges in geo["edges"]:
        widths = np.diff(axis_edges)
        if not all(np.abs(widths - np.mean(widths)) < 0.001 * np.abs(np.mean(widths))):
            geo["is_regular_grid"] = False
            break

    k_alt_range = conf.RET_ALT_RANGE
    if k_alt_range is None:
        geo["excludeK"] = None
        geo["k_alt_range"] = (0, 150)
    else:
        geo["excludeK"] = np.logical_or(geo["alt"] < k_alt_range[0] * 1e3, geo["alt"] > k_alt_range[1] * 1e3)
        geo["k_alt_range"] = k_alt_range

    geo["image_vars"] = ['CCDSEL', 'NCSKIP', 'NCBINCCDColumns', 'NRSKIP', 'NRBIN', 'NCOL', 'NROW',
                         'EXPDate', 'qprime', 'afsGnssStateJ2000', 'afsAttitudeState',
                         "IR1c", "IR2c", "TPlon", "TPlat", "TEXPMS"]
    return geo

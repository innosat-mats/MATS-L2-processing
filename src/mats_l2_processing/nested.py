import numpy as np
import scipy.sparse as sp
from scipy.interpolate import interp1d

from mats_l2_processing.regularisation import Sa_inv_multivariate
from mats_l2_processing.util import geoid_radius, sph2cart, cart2sph, array_shifts
from mats_l2_processing.atm import gridded_data
from mats_l2_processing.forward_model import Forward_model_temp_abs
from mats_l2_processing.grid_1d import Alt_1D_stacked_grid
from mats_l2_processing.solver import Linear_solver
from mats_l2_processing.parameters import get_updated_conf


def nested_VER_1D(conf, const, metadata, main_data, rt_data, prefix, processes=1):
    # Create an updated configuration for 1D retrieval
    vars_1D = {"RET_QTY": ["VER"],
               "SCALES": [1e4],
               "BOUNDS": [(0, 2e8)],
               "AUX_QTY": ["O2", "T"]}
    conf_1d = get_updated_conf(conf, vars_1D)

    # Setup 1D grid
    column = int(np.floor(metadata["NCOL"][0] / 2))
    grid = Alt_1D_stacked_grid(metadata, conf_1d, const, column, processes=processes, verify=False)

    # Setup a priori and aux data
    # idx = ["VER_apr" in item[0] for item in conf.GRIDDED_PRE].index(True)
    # gridded_spec = conf.GRIDDED_PRE_1D[:idx]
    gridded = gridded_data(conf.GRIDDED_PRE_1D, from_grid=grid).data
    atm_apr = [np.zeros_like(gridded["T_apr"])]
    aux = [gridded["O2"], gridded["T_apr"]]

    # Setup forward model
    fwdm = Forward_model_temp_abs(conf_1d, const, grid, metadata, aux, rt_data, combine_images=False)
    obs = fwdm.prepare_obs(conf_1d, main_data, {"IR1": "IR1c", "IR2": "IR2c"})

    # Setup inverse model
    Sa_inv, terms = Sa_inv_multivariate((grid.centers), conf.SA_WEIGHTS_1D_APR, volume_factors=True,
                                        store_terms=False, var_scales=None)
    num_im_obs = len(grid.rows) * len(conf_1d.CHANNELS)
    Se_inv = sp.diags(np.ones(num_im_obs), 0).astype('float32') / (conf.RAD_SCALE ** 2 * num_im_obs)
    solver = Linear_solver(fwdm, obs, conf_1d, Sa_inv, Se_inv, atm_apr, fname=f"{prefix}_ver_1D_apr")

    # Solve and write the result to the nc file
    atm_res = solver.solve(processes)
    return atm_res[0], grid


def hor_shifts_1D(alt_grid, tp_lat, sat_pos, ref_alt=80e3):
    # Calculates horizontal TP shifts of 1D retrieval profile in the orbital plane w.r.t. reference altitude
    assert len(tp_lat) == sat_pos.shape[0]
    sat_radial_coord = np.sqrt(sat_pos[:, 0] ** 2 + sat_pos[:, 1] ** 2 + sat_pos[:, 2] ** 2)
    alt, sat_R = np.meshgrid(alt_grid, sat_radial_coord, indexing='ij')
    _, geoid_r = np.meshgrid(alt_grid, geoid_radius(np.deg2rad(tp_lat)), indexing='ij')
    tp_ang = np.arccos((geoid_r + alt) / sat_R)
    ref_idx = np.argmin(np.abs(alt_grid - ref_alt))
    tp_ang -= tp_ang[ref_idx, :][np.newaxis, :]
    return tp_ang


def get_alongtrack_coord(ecef_to_local, tp_lat, tp_lon):
    assert len(tp_lon) == len(tp_lat)
    tp_ecef_sph = np.stack([np.ones(len(tp_lon)), np.deg2rad(tp_lon), np.deg2rad(tp_lat)], axis=0)
    tp_ecef_cart = np.array(sph2cart(tp_ecef_sph[0, :], tp_ecef_sph[1, :], tp_ecef_sph[2, :])).T
    tp_local_sph = cart2sph(ecef_to_local.inv().apply(tp_ecef_cart))
    mid_image_idx = int(np.floor(len(tp_lat) / 2))
    along = tp_local_sph[1] - tp_local_sph[1][mid_image_idx]
    across = tp_local_sph[2]
    return along, across


def init_3D_from_1D(grid_3d, grid_1d, ver_1d, metadata):
    assert all(grid_3d.centers[0] == grid_1d.centers[0])
    assert ver_1d.shape[0] == grid_1d.atm_shape[1]
    alt_grid = grid_3d.centers[0]

    along_coord, _ = get_alongtrack_coord(grid_3d.ecef_to_local, metadata["TPlat"], metadata["TPlon"])
    shifts = hor_shifts_1D(grid_1d.centers[0], metadata["TPlat"], metadata["afsGnssStateJ2000"],
                           ref_alt=grid_1d.ref_alt)
    alongg = along_coord[np.newaxis, :] + shifts

    # talongg, tshifts, talong_coord = [x * 6450 for x in [alongg, shifts, along_coord]]
    # breakpoint()
    res = np.nan * np.ones((len(alt_grid), len(grid_3d.centers[2])))
    for i, alt in enumerate(alt_grid):
        interp = interp1d(alongg[i, :], ver_1d[:, i], fill_value=(ver_1d[0, i], ver_1d[-1, i]),
                          bounds_error=False)
        res[i, :] = interp(grid_3d.centers[2])

    return np.broadcast_to(res[:, np.newaxis, :], tuple([len(x) for x in grid_3d.centers]))


def compensate_curvature(grid, metadata, qtys):
    along, across = get_alongtrack_coord(grid.ecef_to_local, metadata["tp_lat"], metadata["tp_lon"])
    interp = interp1d(along, across, fillvalue=(across[0], across[-1]))
    ang_cor = - interp(grid.centers[2])
    shifts = np.broadcast_tp(ang_cor[np.newaxis, :], (len(grid.centers[0]), len(grid.centers[2])))
    return [array_shifts[qty, shifts, 1] for qty in qtys]

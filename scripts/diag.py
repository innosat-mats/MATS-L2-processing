import logging
import numpy as np
import datetime as DT
import argparse
import scipy.sparse as sp

from mats_l2_processing import parameters, io
from mats_l2_processing.obs import obs_data_from_L2_ncdf
# from mats_l2_processing.util import geoid_radius
from mats_l2_processing.grid_3d import Alt_along_3D_grid
from mats_l2_processing.regularisation import Sa_inv_multivariate
from mats_l2_processing.diagnostics import calc_diagnostic, write_diag_base_info


logging.addLevelName(15, "PROG")
logging.basicConfig(level=15, format='%(levelname)-5s %(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M:%S', filename='diag.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(15)
console.setFormatter(logging.Formatter('%(levelname)-5s %(message)s'))
logging.getLogger('').addHandler(console)


def get_args():
    parser = argparse.ArgumentParser(description="MATS 3-D retireval diagnostics",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prefix", type=str, default="tomo", help="Tomography results and jacobian file prefix")
    parser.add_argument("--obs_data", type=str, default="IRc.nc",
                        help="Observations from netCDF files (comma separated list), in the order of CHANNELS")
    parser.add_argument("--conf", type=str, default="conf.py", help="Read configuration from file.")
    parser.add_argument("--processes", type=int, default=6, help="Number of threads for jacobian calculation.")
    return parser.parse_args()


def main():
    # Parse arguments and configuration files
    args = get_args()
    conf, const = parameters.make_conf("diag", args.conf, args)

    num_stats, num_dpoints, num_dqty = len(conf.DIAG_STATS), len(conf.DIAG_POINTS), len(conf.DIAG_QTY)
    if num_stats * num_dpoints * num_dqty == 0:
        raise ValueError("At least one each of diagnostics statistic, grid point and quantity must be set!")

    logging.info("Loading input data...")
    # Construct grid object using L1 data
    start_time, stop_time = [DT.datetime(*time, tzinfo=DT.timezone.utc) for time in [conf.START_TIME, conf.STOP_TIME]]

    obs_files = args.obs_data.split(",")
    metadata = [io.read_L1_ncdf(file, start_time=start_time, stop_time=stop_time, var=const.NEEDED_DATA,
                                center_times=True) for file in obs_files]
    grid = Alt_along_3D_grid(metadata, conf, const, processes=args.processes, verify=False)

    # Read in L2 data
    L2_vars = conf.DIAG_QTY + [f"{v}_apr" for v in conf.DIAG_QTY] + conf.CHANNELS + [f"{v}_sim" for v in conf.CHANNELS]
    diag_data = {}
    L2_data = io.read_ncdf(f"{args.prefix}_L2.nc", L2_vars) 
    diag_data["obs_data"] = obs_data_from_L2_ncdf(L2_data, conf.CHANNELS)
    diag_data["obs_data_sim"] = obs_data_from_L2_ncdf(L2_data, conf.CHANNELS, obs_suffix="_sim")
    diag_data["atm_sol"] = grid.atm_from_L2_ncdf(L2_data)
    diag_data["atm_apr"] = grid.atm_from_L2_ncdf(L2_data, atm_suffix="_apr")

    # Read Jacobian
    diag_data["K"] = sp.load_npz(f"{args.prefix}_last_K.npz")

    logging.info("Building regularisation matrices...")
    earth_radius = const.MEAN_EARTH_RADIUS + 8e4
    scaled_grid = [grid.points[0], grid.points[1] * earth_radius, grid.points[2] * earth_radius]
    diag_data["Sa_inv"], terms = Sa_inv_multivariate(scaled_grid, conf.SA_WEIGHTS, volume_factors=True,
                                                     store_terms=True, var_scales=None, aspect_ratio=conf.ASPECT_RATIO,
                                                     exp_alt_axis=conf.EXP_ALT_AXIS)

    obs_size = len(diag_data["obs_data"].flatten())
    diag = np.ones((obs_size))
    diag_data["Se_inv"] = sp.diags(diag, 0).astype('float32') / (conf.RAD_SCALE ** 2 * len(conf.CHANNELS))

    # Identify indices of elements for which diagnostics will be calculated
    scaled_grid = [x * 1e-3 for x in scaled_grid]
    grid_idxs = [np.array([np.argmin(np.abs(scaled_grid[d] - p[d])) for d in range(len(grid.points))])
                 for p in conf.DIAG_POINTS]
    diag_data["num_points"] = len(grid_idxs)
    idxs, qty_idxs = [], []
    for q, qty in enumerate(conf.DIAG_QTY):
        for i, idx in enumerate(grid_idxs):
            idxs.append(grid.vec_idx(q, idx))
            qty_idxs.append(q)
    diag_data["idxs"] = np.array(idxs)
    diag_data["qty_idxs"] = np.array(qty_idxs)

    logging.info("Running diagnostics...")
    write_diag_base_info(grid, grid_idxs, f"{args.prefix}_L2.nc")
    for dia in conf.DIAG_STATS:
        calc_diagnostic(diag_data, grid, dia, f"{args.prefix}_L2.nc", conf, const)


if __name__ == "__main__":
    main()

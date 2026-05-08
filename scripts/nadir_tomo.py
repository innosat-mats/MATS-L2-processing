import logging
import argparse
import datetime as DT
import numpy as np
import scipy.sparse as sp

from mats_l2_processing import parameters
from mats_l2_processing.io import read_nadir_gl_zarr
from mats_l2_processing.solver import Linear_solver
from mats_l2_processing.nadir import Nadir_grid_lonlat, Nadir_forward_model
from mats_l2_processing.regularisation import Sa_inv_multivariate
from mats_l2_processing.obs_preprocessing import denoise


logging.addLevelName(15, "PROG")
logging.basicConfig(level=15, format='%(levelname)-5s %(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M:%S', filename='LM.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(15)
console.setFormatter(logging.Formatter('%(levelname)-5s %(message)s'))
logging.getLogger('').addHandler(console)


def get_args():
    parser = argparse.ArgumentParser(description="MATS nadir tomography test",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("obs_data", type=str,
                        help="Observations from netCDF files (comma separated list), in the order of CHANNELS")
    parser.add_argument("--conf", type=str, default="conf.py", help="MATS L2 configuration file.")
    parser.add_argument("--mask", type=str, help="npy file with bad pixel mask for the nadir channel.")
    parser.add_argument("--mask_var", type=str, default="bias_mask", help="The variable to read from mask file.")
    parser.add_argument("--start_time", dest="START_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("--stop_time", dest="STOP_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("--processes", type=int, default=1, help="Number of threads for jacobian calculation.")
    return parser.parse_args()


def main():
    # Parse arguments and configuration files
    args = get_args()
    conf, const = parameters.make_conf("nadir_tomo", args.conf, args)

    entries = [st.split(":") for st in args.obs_data.split(",")]
    files = {float(entry[0]): entry[1] for entry in entries}

    if args.START_TIME:
        time_range = [DT.datetime(*time, tzinfo=DT.timezone.utc) for time in [args.START_TIME, args.STOP_TIME]]
    else:
        time_range = None

    logging.info("Loading and processing input data...")
    nadir_data = read_nadir_gl_zarr(files, time_range=time_range,
                                    perc=(None if conf.PERC_FILTER < 0 else conf.PERC_FILTER))
    if args.mask:
        mask = np.flip(np.swapaxes(np.load(args.mask)[args.mask_var], 0, 1), axis=0)
    else:
        mask = None
    if conf.PERC_FILTER > 0:
        nadir_data["img"] -= nadir_data["perc"][np.newaxis, :, :]
        hot_pix = np.swapaxes(nadir_data["perc"], 0, 1)
    else:
        hot_pix = None
    if conf.NADIR_DENOISE:
        nadir_data["img"] = denoise(nadir_data["img"], conf.NADIR_DENOISE_HW, conf.NADIR_DENOISE_THR)

    logging.info("Initializing geometry...")
    grid = Nadir_grid_lonlat(nadir_data, conf, const, mask=mask)
    fwdm = Nadir_forward_model(grid)

    logging.info("Initializing inverse model...")
    reg_points = (grid.points[0], grid.points[1] * np.cos(np.deg2rad(np.mean(grid.lat))), grid.points[2])
    Sa_inv, terms = Sa_inv_multivariate(reg_points, conf.SA_WEIGHTS, volume_factors=True, store_terms=True,
                                        aspect_ratio=1, var_scales=None, exp_alt_axis=-1)
    Se_inv = sp.diags(np.ones((grid.num_obs)), 0).astype('float32') / (conf.RAD_SCALE ** 2 * len(fwdm.channels))

    logging.info("Calculating jacobian...")
    jac, valid_points = grid.calc_jacobian(args.processes)

    logging.info("Solving...")
    obs_data = fwdm.prepare_obs(nadir_data["img"])
    # obs_data_test = np.full_like(obs_data, 1.0)
    # obs_data_test[0, :, 28, 7] = 1.0
    # breakpoint()
    solver = Linear_solver(fwdm, obs_data, conf, Sa_inv, Se_inv, np.zeros(grid.atm_shape), Sa_terms=terms)
    sol = solver.solve(args.processes, jac=jac, fx=np.zeros_like(obs_data))
    grid.write_nadir_L2_ncdf(obs_data, np.where(valid_points, sol, np.nan), "L2_nadir.nc", mask=True, hot_pix=hot_pix)


if __name__ == "__main__":
    main()

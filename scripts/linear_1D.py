import logging
import numpy as np
import datetime as DT
import argparse
from os.path import expanduser
import scipy.sparse as sp

from mats_l2_processing.io import read_L1_ncdf
from mats_l2_processing.parameters import make_conf
from mats_l2_processing.atm import gridded_data
from mats_l2_processing.grid_1d import Alt_1D_stacked_grid
from mats_l2_processing.solver import Linear_solver
from mats_l2_processing.forward_model import Forward_model_temp_abs
from mats_l2_processing.regularisation import Sa_inv_multivariate
from mats_l2_processing.util import dict_contents, filter_dict


logging.addLevelName(15, "PROG")
logging.basicConfig(level=15, format='%(levelname)-5s %(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M:%S', filename='LM.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(15)
console.setFormatter(logging.Formatter('%(levelname)-5s %(message)s'))
logging.getLogger('').addHandler(console)


def get_args():
    parser = argparse.ArgumentParser(description="1D MATS retrieval script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--obs_data", type=str, default="IRc.nc", help="Observational data from netCDF file.")
    parser.add_argument("--start_time", dest="START_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("--stop_time", dest="STOP_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("--conf", type=str, default="conf.py", help="Read configuration from file.")
    parser.add_argument("--prefix", type=str, required=True, help="File name prefixes for output.")
    parser.add_argument("--processes", type=int, default=6, help="Number of threads to use.")
    parser.add_argument("--no_reg_analysis", action="store_true",
                        help="Do not calculate regularisation diagnostics.")
    parser.add_argument("--verify_grid", action="store_true",
                        help="Compute all LOS before doing anything else, to ensure they fit into grid.")
    parser.add_argument("--debug_nan", action="store_true",
                        help="Do not abort immediately if Nan's encountered, warn and work around instead. " +
                        "Intended for debugging. DO NOT USE FOR PRODUCTION!")
    return parser.parse_args()


def main():
    # Parse arguments and configuration files
    args = get_args()
    conf, const = make_conf("linear_1D", args.conf, args)

    # Parse times
    start_time, stop_time = [DT.datetime(*time, tzinfo=DT.timezone.utc) for time in [conf.START_TIME, conf.STOP_TIME]]

    logging.info("Loading input data...")
    metadata = read_L1_ncdf(args.obs_data, start_time=start_time, stop_time=stop_time,
                            var=const.NEEDED_DATA + ["valid_img"])
    main_data = read_L1_ncdf(args.obs_data, start_time=start_time, stop_time=stop_time, var=["IR1c", "IR2c"])
    main_data, metadata = [filter_dict(arr, metadata["valid_img"]) for arr in [main_data, metadata]]
    logging.info(f"Found {metadata['size']} valid images.")

    rt_data = np.load(expanduser(conf.RT_DATA_FILE))
    rt_data = {name: rt_data[name].copy() for name in ["filters", "sigma", "emission", "sigma_grad", "emission_grad"]}

    logging.info("Initializing geometry...")
    column = conf.COL_RANGE[0]
    grid = Alt_1D_stacked_grid(metadata, conf, const, column, processes=args.processes, verify=False)

    logging.info("Initializing atmospheric quantities...")
    gridded = gridded_data(conf.GRIDDED_PRE, from_grid=grid).data
    atm_apr = [gridded[f"{name}_apr"] for name in grid.ret_qty]
    if len(grid.aux_qty) > 0:
        aux = [gridded[name] for name in grid.aux_qty]

    logging.info("Initializing forward model...")
    fwdm = Forward_model_temp_abs(conf, const, grid, metadata, aux, rt_data, combine_images=False)
    obs = fwdm.prepare_obs(conf, main_data, {"IR1": "IR1c", "IR2": "IR2c"})

    logging.info("Initializing inverse model...")
    Sa_inv, terms = Sa_inv_multivariate((grid.centers), conf.SA_WEIGHTS, volume_factors=True,
                                        store_terms=False, var_scales=None)
    num_im_obs = len(grid.rows) * len(conf.CHANNELS)
    Se_inv = sp.diags(np.ones(num_im_obs), 0).astype('float32') / (conf.RAD_SCALE ** 2 * num_im_obs)

    solver = Linear_solver(fwdm, obs, conf, Sa_inv, Se_inv, atm_apr, fname=args.prefix)

    logging.info("Solving...")
    solver.solve(args.processes)


if __name__ == "__main__":
    main()

import logging
import numpy as np
import datetime as DT
import argparse
from os.path import expanduser
import sys
import scipy.sparse as sp

from mats_l2_processing import parameters, io
from mats_l2_processing.atm import gridded_data
from mats_l2_processing.util import geoid_radius
from mats_l2_processing.grid_3d import Alt_along_3D_grid
from mats_l2_processing.solver import Lavenberg_marquardt_solver
from mats_l2_processing.forward_model import Forward_model_temp_abs
from mats_l2_processing.regularisation import Sa_inv_multivariate
from mats_l2_processing.nested import nested_VER_1D, init_3D_from_1D

from mats_l2_processing.plotting import plot_2d

def get_args():
    parser = argparse.ArgumentParser(description="Iterative solver for temperature from MATS data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--obs_data", type=str, default="IRc.nc", help="Observational data from netCDF file.")
    parser.add_argument("--start_time", dest="START_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("--stop_time", dest="STOP_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("--conf", type=str, default="conf.py", help="Read configuration from file.")
    parser.add_argument("--prefix", type=str, required=True, help="File name prefixes for numpy output.")
    parser.add_argument("--processes", type=int, default=6, help="Number of threads for jacobian calculation.")
    parser.add_argument("--no_reg_analysis", action="store_true",
                        help="Do not calculate regularisation diagnostics.")
    parser.add_argument("--verify_grid", action="store_true",
                        help="Compute all LOS before doing anything else, to ensure they fit into grid.")
    parser.add_argument("--save_jacobian", action="store_true",
                        help="Save first itration jacobian as .npy archive. This may need 2 GB of disk space or more!")
    parser.add_argument("--load_jacobian", action="store_true",
                        help="Load first iteration jacobian from a file. Overrides --save_jacobian.")
    parser.add_argument("--debug_nan", action="store_true",
                        help="Do not abort immediately if Nan's encountered, warn and work around instead. " +
                        "Intended for debugging. DO NOT USE FOR PRODUCTION!")
    return parser.parse_args()


def main():
    # Parse arguments and configuration files
    args = get_args()
    conf, const = parameters.make_conf("post", args.conf, args)

    preload_vars = conf.AUX_QTY + conf.RET_QTY + [f"{var}_apr" for var in conf.RET_QTY]
    gridded = gridded_data(conf.GRIDDED_POST, from_L2_ncdf=args.L2_ncdf, ncdf_preload=preload_vars)
    for name, dat in gridded_data.items()



if __name__ == "__main__":
    main()

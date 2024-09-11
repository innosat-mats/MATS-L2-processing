import logging
import numpy as np
import datetime as DT
import argparse
from os.path import expanduser
import xarray as xr
import scipy.sparse as sp

from mats_l2_processing import grids, parameters, io, atm
from mats_l2_processing import forward_model as fwdm
from mats_l2_processing import inverse_model as invm

logging.addLevelName(15, "PROG")
logging.basicConfig(level=15, format='%(levelname)-5s %(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M:%S', filename='LM.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(15)
console.setFormatter(logging.Formatter('%(levelname)-5s %(message)s'))
logging.getLogger('').addHandler(console)


def get_args():
    parser = argparse.ArgumentParser(description="Iterative solver for temperature from MATS data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--obs_data", type=str, required=True, help="Observational data from netCDF file.")
    parser.add_argument("--rt_data", type=str, default=expanduser("~/L2_data/rt_data_c.npz"),
                        help="Radiative transfer data from npz file.")
    parser.add_argument("--clim_data", type=str, default=expanduser("~/L2_data/msis_cmam_climatology_200.nc"),
                        help="Climatological data from netCDF file.")
    parser.add_argument("--ver_apr", type=str, default=expanduser("~/L2_data/IR2_clim_1.nc"),
                        help="VER climatology from netCDF file.")
    parser.add_argument("--t_apr", type=str, default=expanduser("~/L2_data/T_clim_saber.nc"),
                        help="T climatology from netCDF file.")
    parser.add_argument("--start_time", dest="START_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("--stop_time", dest="STOP_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("--conf", type=str, help="Read configuration from file.")
    parser.add_argument("--ver_fact", type=float, help="Offset for ver apriori.")
    parser.add_argument("--o2_fact", type=float, help="Factor to multiply o2 apriori by.")
    parser.add_argument("--t_offset", type=float, help="Offset for temperature apriori, K.")
    parser.add_argument("--col_range", dest="COL_RANGE", type=int, nargs=2,
                        help="Range of image columns to use")
    parser.add_argument("--row_range", type=int, nargs=2, help="Range of image rows to use")
    parser.add_argument("--prefix", type=str, required=True, help="File name prefixes for numpy output.")
    parser.add_argument("--processes", type=int, default=6, help="Number of threads for jacobian calculation.")
    # parser.add_argument("--nlc_width", type=float, help="NLC dimension along track.")
    parser.add_argument("--nimages", type=int, nargs=2, default=[40, 250],
                        help="Make sure the number of images is in the given ranges.")
    parser.add_argument("--reg_analysis", action="store_true",
                        help="Calculate additional regularisation diagnostics.")
    parser.add_argument("--verify_grid", action="store_true",
                        help="Compute all LOS before doing anything else, to ensure they fit into grid.")
    parser.add_argument("--save_jacobian", action="store_true",
                        help="Save first itration jacobian as .npy archive. This may need 2 GB of disk space or more!")
    parser.add_argument("--load_jacobian", action="store_true",
                        help="Load first iteration jacobian from a file. Overrides --save_jacobian.")
    parser.add_argument("--debug_nan", action="store_true",
                        help="Add additional checks to catch nan's. Slow and requires extra memory!")
    return parser.parse_args()


def main():
    # Parse arguments and configuration files
    args = get_args()
    conf, const = parameters.make_conf("iter_T", args.conf, args)

    # Parse times
    start_time, stop_time = [DT.datetime(*time, tzinfo=DT.timezone.utc) for time in [conf.START_TIME, conf.STOP_TIME]]
    mean_time = DT.datetime.fromtimestamp(sum(map(DT.datetime.timestamp, [start_time, stop_time])) / 2)

    logging.info("Loading input data...")
    data = io.read_L1_ncdf(args.obs_data, start_time=start_time, stop_time=stop_time, var=const.NEEDED_DATA)
    rt_data = np.load(expanduser(args.rt_data))
    rt_data = {name: rt_data[name].copy() for name in ["filters", "sigma", "emission", "sigma_grad", "emission_grad"]}
    clim_data = xr.load_dataset(expanduser(args.clim_data))
    ver_apr = io.read_ncdf(args.ver_apr, ["altitude", "latitude", "VER"])

    logging.info("Initializing geometry...")
    row_range = args.row_range if args.row_range else (0, data["NROW"][0])
    columns, rows = [np.arange(r[0], r[1], 1) for r in [conf.COL_RANGE, row_range]]
    local_earth_radius = grids.geoid_radius(np.deg2rad(np.mean(data['TPlat'])))
    grid_proto = [grids.make_grid_proto(conf.ALT_GRID, scaling=1e3, offset=local_earth_radius),
                  grids.make_grid_proto(conf.ACROSS_GRID, scaling=1e3 / local_earth_radius),
                  grids.make_grid_proto(conf.ALONG_GRID, scaling=1e3 / local_earth_radius)]
    jb = grids.initialize_geometry(data, columns, rows, conf, grid_proto=grid_proto, processes=args.processes)

    if args.verify_grid:
        fwdm.test_grid(jb, args.processes)
    # Interpolate background atmosphere on retrieval grid
    logging.info("Initializing atmospheric data...")
    o2, atm_apr = atm.get_background(jb, mean_time, clim_data, ver_apr)
    if args.o2_fact:
        o2 *= args.o2_fact
    if args.ver_fact:
        atm_apr = [atm_apr[0] * args.ver_fact, atm_apr[1]]
    if args.t_offset:
        atm_apr = [atm_apr[0], atm_apr[1] + args.t_offset]

    logging.info("Processing observation data...")
    y, tan_alts = fwdm.calc_obs(jb, args.processes)

    logging.info("Generating covariance matrices...")
    Sa_inv, terms = invm.Sa_inv_multivariate((jb["rad_grid_c"], jb["acrosstrack_grid_c"] * local_earth_radius,
                                              jb["alongtrack_grid_c"] * local_earth_radius),
                                             [conf.SA_WEIGHTS_VER, conf.SA_WEIGHTS_T], store_terms=args.reg_analysis,
                                             volume_factors=True, aspect_ratio=conf.ASPECT_RATIO)
    Se_inv = sp.diags(np.ones([y.flatten().shape[0]]), 0).astype('float32') / (conf.RAD_SCALE ** 2 * len(y))

    logging.info("Starting LM iteration...")
    invm.lm_solve(atm_apr, y, tan_alts, Se_inv, Sa_inv.tocsr(), terms, conf, jb, rt_data, o2, args.processes,
                  args.prefix, save_K=args.save_jacobian, load_K=args.load_jacobian, verify=args.debug_nan)


if __name__ == "__main__":
    main()

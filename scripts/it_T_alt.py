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
    conf, const = parameters.make_conf("iter_T", args.conf, args)

    # Parse times
    start_time, stop_time = [DT.datetime(*time, tzinfo=DT.timezone.utc) for time in [conf.START_TIME, conf.STOP_TIME]]

    logging.info("Loading input data...")
    metadata = io.read_L1_ncdf(args.obs_data, start_time=start_time, stop_time=stop_time, var=const.NEEDED_DATA)
    metadata["EXPDate_s"] = metadata["EXPDate_s"] + metadata["TEXPMS"] * 5e-4
    metadata["EXPDate"] = np.array([date + exposure * 5e-4 * DT.timedelta(0, 1)
                                    for date, exposure in zip(metadata["EXPDate"], metadata["TEXPMS"])])
    main_data = io.read_L1_ncdf(args.obs_data, start_time=start_time, stop_time=stop_time, var=["IR1c", "IR2c"])
    rt_data = np.load(expanduser(conf.RT_DATA_FILE))
    rt_data = {name: rt_data[name].copy() for name in ["filters", "sigma", "emission", "sigma_grad", "emission_grad"]}

    logging.info("Initializing geometry...")
    grid = Alt_along_3D_grid(metadata, conf, const, processes=args.processes, verify=args.verify_grid)

    logging.info("Performing 1D linear VER retrieval to set VER a priori...")
    ver_1d, grid_1d = nested_VER_1D(conf, const, metadata, main_data, rt_data, args.prefix, processes=args.processes)
    ver_from_1d = init_3D_from_1D(grid, grid_1d, ver_1d, metadata)

    logging.info("Initializing atmospheric quantities...")
    gridded = gridded_data(conf.GRIDDED_PRE, extend_previous={"VER_from_1D": ver_from_1d}, from_grid=grid).data
    atm_apr = [gridded["VER_apr"], gridded["T_apr"]]

    # breakpoint()
    eff_rad_km = geoid_radius(np.mean(metadata["TPlat"])) * 1e-3 + 80
    cid = int(np.floor(len(grid.centers[1]) / 2))
    plot_2d(grid_1d.img_time - grid_1d.img_time[0], grid_1d.centers[0] * 1e-3, ver_1d, fname="1d_ver_raw",
            xlabel="Exposure time, s", ylabel="Altitude, km")
    plot_2d(grid.centers[2] * eff_rad_km, grid.centers[0] * 1e-3, ver_from_1d[:, cid, :].T, fname="1d_ver_interp",
            xlabel="Exposure time, s", ylabel="Altitude, km")
    plot_2d(grid.centers[2] * eff_rad_km, grid.centers[0] * 1e-3, gridded["VER_apr"][:, cid, :].T, fname="1d_ver_apr",
            xlabel="Exposure time, s", ylabel="Altitude, km")
    plot_2d(grid.centers[2] * eff_rad_km, grid.centers[0] * 1e-3, gridded["T_apr"][:, cid, :].T, fname="1d_t_apr",
            xlabel="Exposure time, s", ylabel="Altitude, km")

    out_fname = f"{args.prefix}_L2.nc"
    grid.write_grid_ncdf(out_fname)
    grid.write_atm_ncdf(out_fname, atm_apr, atm_suffix="_apr", atm_suffix_long=", a priori")
    io.append_gen_ncdf(out_fname, {"O2": ["O2 number density", "cm-3", gridded["O2"][np.newaxis, ...],
                                          ("time", "alt_coord", "acrosstrack_coord", "alongtrack_coord")]})
    # sys.exit(0)

    logging.info("Initializing forward model...")
    fwdm = Forward_model_temp_abs(conf, const, grid, metadata, [gridded["O2"]], rt_data, combine_images=True)

    logging.info("Initializing inverse model...")
    obs = fwdm.prepare_obs(conf, main_data, {"IR1": "IR1c", "IR2": "IR2c"})
    del main_data

    local_earth_radius = geoid_radius(np.deg2rad(np.mean(metadata['TPlat'])))
    Sa_inv, terms = Sa_inv_multivariate((grid.centers[0], grid.centers[1] * local_earth_radius,
                                         grid.centers[2] * local_earth_radius), conf.SA_WEIGHTS, volume_factors=True,
                                        store_terms=(not args.no_reg_analysis), aspect_ratio=conf.ASPECT_RATIO,
                                        var_scales=None)

    obs_size = len(obs.flatten())
    Se_inv = sp.diags(np.ones((obs_size)), 0).astype('float32') / (conf.RAD_SCALE ** 2 * len(fwdm.channels))

    invm = Lavenberg_marquardt_solver(fwdm, obs, conf, Sa_inv, Se_inv, atm_apr, args.prefix, Sa_terms=terms,
                                      save_jac=args.save_jacobian, load_jac=args.load_jacobian)

    logging.info("Starting Levemberg-Marquardt iteration...")
    invm.solve(args.processes)


if __name__ == "__main__":
    main()

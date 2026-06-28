import numpy as np
import datetime as DT
import argparse


from mats_l2_processing.grid_3d import Alt_along_3D_grid
from mats_l2_processing.util import geoid_radius
from mats_l2_processing import parameters, io


def acrosstrack_width(metadata, ref_alt):
    res = np.empty_like(metadata["time"])
    for i in range(len(res)):
        idxs = [np.argmin(np.abs(ref_alt - metadata["TPheightPixel"][i, :, col])) for col in [0, -1]]
        res[i] = np.sqrt(np.sum([(metadata[name][i, idxs[0], 0] - metadata[name][i, idxs[1], -1]) ** 2
                                for name in ["TPECEFx", "TPECEFy", "TPECEFz"]]))
    return res


def get_args():
    parser = argparse.ArgumentParser(description="Iterative solver for temperature from MATS data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--obs_data", type=str, default="IR1c.nc",
                        help="Observations from netCDF files (comma separated list), in the order of CHANNELS")
    parser.add_argument("--start_time", dest="START_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("--stop_time", dest="STOP_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("--conf", type=str, default="conf.py", help="Read configuration from file.")
    parser.add_argument("--prefix", type=str, required=True, help="File name prefixes for numpy output.")
    parser.add_argument("--processes", type=int, default=6, help="Number of threads for jacobian calculation.")
    parser.add_argument("--verify_grid", action="store_true",
                        help="Compute all LOS before doing anything else, to ensure they fit into grid.")
    return parser.parse_args()


def main():
    # Parse arguments and configuration files
    args = get_args()
    conf, const = parameters.make_conf("grid", args.conf, args)

    # Parse times
    start_time, stop_time = [DT.datetime(*time, tzinfo=DT.timezone.utc) for time in [conf.START_TIME, conf.STOP_TIME]]
    metadata = io.read_L1_ncdf(args.obs_data, start_time=start_time, stop_time=stop_time, var=const.NEEDED_DATA,
                               center_times=True)
    grid = Alt_along_3D_grid([metadata], conf, const, processes=args.processes, verify=args.verify_grid)
    grid.write_grid_ncdf_exp(f"{args.prefix}_L2.nc", track_width=acrosstrack_width(metadata, 8e4))


if __name__ == "__main__":
    main()

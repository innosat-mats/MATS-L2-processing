import numpy as np
import datetime as DT
import argparse


from mats_l2_processing.grid_3d import Alt_along_3D_grid
from mats_l2_processing.util import DT2seconds
from mats_l2_processing import parameters, io


def acrosstrack_width(metadata, ref_alt):
    res = np.empty_like(metadata["time"])
    for i in range(len(res)):
        idxs = [np.argmin(np.abs(ref_alt - metadata["TPheightPixel"][i, :, col])) for col in [0, -1]]
        res[i] = np.sqrt(np.sum([(metadata[name][i, idxs[0], 0] - metadata[name][i, idxs[1], -1]) ** 2
                                for name in ["TPECEFx", "TPECEFy", "TPECEFz"]]))
    return res


def gen_chunk_times(start_date, stop_date):
    start_time, stop_time = [DT.datetime(*(time + [0, 0, 0]), tzinfo=DT.timezone.utc)
                             for time in [start_date, stop_date]]
    assert stop_time > start_time
    ndays = int(np.round((stop_time - start_time) / DT.timedelta(days=1)))
    days_dt = [start_time + DT.timedelta(days=d) for d in range(ndays)]
    minutes_dt = np.array([DT.timedelta(minutes=int(m)) for m in np.arange(5, 1436, 10)])

    return np.concatenate([ddt + minutes_dt for ddt in days_dt])


def write_grids_ncdf(grid_sample, chunk_times, conf, fname, sza=False):
    eff_radius = grid_sample.ref_rad
    sec_unit = "seconds since 2000-01-01"
    max_img_per_chunk = 102 + 2 * conf.TOMO_PADDING_IMG
    dim_pars = {"alt": ("Altitude", "meter", grid_sample.points[0]),
                "acrosstrack_coord": ("Horizontal coordinate in the direction perpendicular to the orbital plane",
                                      "meter", grid_sample.points[1] * eff_radius),
                "alongtrack_coord": ("Horizontal coordinate in the direction of the satellite track", "meter",
                                     grid_sample.points[2] * eff_radius),
                "img_time": ("Acquisition time of individual MATS images", sec_unit, np.zeros(max_img_per_chunk)),
                "chunk_time": ("Center time of data chunk aquisition", sec_unit, chunk_times)
    }
    full_shape = ((len(chunk_times), len(grid_sample.points[1]), len(grid_sample.points[2])))
    full_dims = ("chunk_time", "acrosstrack_coord", "alongtrack_coord")
    along_shape = (full_shape[0], full_shape[2])
    along_dims = (full_dims[0], full_dims[2])

    ncvars = {"longitude": ("Longitude", "degree_east", np.full(full_shape, np.nan), full_dims),
              "latitude": ("Latitude", "degree_north", np.full(full_shape, np.nan), full_dims),
              "valid_time_alongtrack": ("Valid time for each alongtrack coordinate", sec_unit,
                                        np.full(along_shape, np.nan), along_dims),
              "valid_data_alongtrack": ("Valid data flag for alongtrack coordinate", None,
                                        np.zeros(along_shape), along_dims),
              "chunk_img_time": ("Acquisition time of individual MATS images in data chunk", sec_unit,
                                 np.full((len(chunk_times), max_img_per_chunk), np.nan),
                                 ("chunk_time", "img_time")),
              "valid_chunk": ("Flag indicating chunk has meaningful data", None, np.zeros(len(chunk_times)),
                              ("chunk_time")),
    }

    if sza:
        ncvars["sza"] = ("Solar zenith angle for alongtrack coordinate", "degree",
                         np.full(along_shape, np.nan), along_dims)

    io.write_gen_ncdf(fname, dim_pars, ncvars, {})


def get_args():
    parser = argparse.ArgumentParser(description="Iterative solver for temperature from MATS data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--obs_data", type=str, default="IR1.nc",
                        help="Observations from netCDF files (comma separated list), in the order of CHANNELS")
    parser.add_argument("--start_date", dest="START_TIME", type=int, nargs=3, help="Start time for data set.")
    parser.add_argument("--stop_date", dest="STOP_TIME", type=int, nargs=3, help="Start time for data set.")
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
    chunk_times = gen_chunk_times(args.START_TIME, args.STOP_TIME)
    chunk_hw = DT.timedelta(minutes=5) + DT.timedelta(seconds=conf.TOMO_PADDING_IMG * const.NOMINAL_IMG_STEP)

    fname = f"{args.prefix}_L2.nc"
    init_grid = False
    var = const.NEEDED_DATA + ["TPsza"]
    for idx in range(len(chunk_times)):
        start_time, stop_time = [chunk_times[idx] + sgn * chunk_hw for sgn in [-1, 1]]
        metadata = io.read_L1_ncdf(args.obs_data, start_time=start_time, stop_time=stop_time, var=var,
                                   center_times=True)
        if len(metadata["time"]) < 2 * conf.TOMO_PADDING_IMG + 10:
            print(f"INFO: insufficient data for chunk {idx}.")
            continue

        try:
            grid = Alt_along_3D_grid([metadata], conf, const, processes=args.processes, verify=args.verify_grid)
        except Exception:
            print(f"WARN: chunk {idx} grid initialization failed!")
            continue

        if not init_grid:
            write_grids_ncdf(grid, [DT2seconds(t) for t in chunk_times], conf, fname, sza=True)
            init_grid = True

        grid.write_to_grids(idx, fname, sza=metadata["TPsza"])
        print(f"INFO: chunk {idx} suceeded.")


if __name__ == "__main__":
    main()

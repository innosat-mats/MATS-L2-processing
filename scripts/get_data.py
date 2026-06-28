from mats_utils.rawdata.read_data import read_MATS_data, read_MATS_PM_data, load_multi_parquet
from mats_l2_processing.parameters import make_conf
from mats_l2_processing.pointing import get_deg_map, faster_heights
from mats_l2_processing.io import write_ncdf_L1, write_ncdf_L1b_zarr_format, read_ncdf
from mats_l2_processing.util import get_filter, multiprocess
from mats_utils.rawdata.read_data import store_as_parquet
# from mats_utils.rawdata.release import write_ncdf_L1b_release
import argparse
import numpy as np
import datetime as DT


def get_args():
    parser = argparse.ArgumentParser(description="Get MATS data (single channel)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--start_time", dest="START_TIME", type=int, nargs=6,
                        help="Start time for data set.")
    parser.add_argument("--stop_time", dest="STOP_TIME", type=int, nargs=6,
                        help="Start time for data set.")
    parser.add_argument("--version", dest="VERSION", type=str,
                        help="Data version.")
    parser.add_argument("--channel", dest="CHANNEL", type=str, help="Channel to retrieve",
                        choices=["IR1", "IR2", "IR3", "IR4", "UV1", "UV2", "all", "PM", "NADIR"])
    parser.add_argument("--filter", dest="FILTER", type=str,
                        help="Data filters. Syntax: field_name1:min:max,field_name2...")
    parser.add_argument("--from_parquet_dir", type=str,
                        help="Get data from local parquet files from specifief dir., not AWS.")
    parser.add_argument("--conf", type=str, help="Read configuration from file.")
    parser.add_argument("--ncdf_out", type=str, help="Output file name (ncdf)")
    parser.add_argument("--level", type=str, choices=["1a", "1b"], default="1b", help="Data level.")
    parser.add_argument("--ncdf_format", type=str, choices=["zarr_like", "all_vars"], default="zarr_like",
                        help="ncdf format for writing leNcdf format for writing level 1b data.")
    parser.add_argument("--release", type=str,
                        help="Make a release-type ncdf with the specified release version.")
    parser.add_argument("--parquet_out", type=str, help="Output file name (parquet)")

    parser.add_argument("--meta", action="store_true", help="Do not store images themselves, just metadata.")
    parser.add_argument("-t", action="store_true", help="Calculate tangent point coordinates for every pixel.")
    parser.add_argument("--dist_data", type=str, default="/home/lk/L2_data/distortion_splines_tck.nc",
                        help="A ncdf file with distortion data")
    parser.add_argument("--processes", type=int, default=15, help="Number of processes for TP calculations.")
    return parser.parse_args()


def tp_calc_wrapper(image, common_args):
    image["time"] = image["EXPDate"].to_pydatetime()
    return faster_heights(image, None, full_coords=True, deg_map=common_args[0])


def main():
    args = get_args()
    conf, _ = make_conf("get_data", args.conf, args)

    if args.from_parquet_dir:
        dftop = load_multi_parquet(args.from_parquet_dir, DT.datetime(*conf.START_TIME),
                                   stop=DT.datetime(*conf.STOP_TIME), filt=get_filter(conf.CHANNEL))
    else:
        if args.CHANNEL == "PM":
            dftop = read_MATS_PM_data(DT.datetime(*conf.START_TIME), DT.datetime(*conf.STOP_TIME),
                                      {}, level="1a", version=conf.VERSION)
        else:
            filt = get_filter(conf.CHANNEL)
            if args.FILTER is not None:
                for f in args.FILTER.split(","):
                    items = f.split(':')
                    assert len(items) == 3, "Malformed filter parameter!"
                    filt[items[0]] = [float(items[1]), float(items[2])]

            dftop = read_MATS_data(DT.datetime(*conf.START_TIME), DT.datetime(*conf.STOP_TIME),
                                   filt, level=args.level, version=conf.VERSION)
    # breakpoint()

    if args.parquet_out:
        store_as_parquet(dftop, f"{args.parquet_out}.parquet.gzip")
        # dftop.to_parquet(f"{args.parquet_out}.parquet.gzip", compression='gzip', engine='fastparquet')
    else:
        # if args.release:
        #     write_ncdf_L1b_release(dftop, args.ncdf_out, conf.CHANNEL, conf.VERSION, args.release)
        # else:
        #     write_ncdf_L1(dftop, args.ncdf_out, conf.CHANNEL, conf.VERSION, image=not args.meta, level=args.level)
        if args.ncdf_out is None:
            ofile = f"L1b_{conf.CHANNEL}"
            if args.START_TIME:
                dates = ["_".join([str(x) for x in array]) for array in [args.START_TIME, args.STOP_TIME]]
                ofile += f"-{'-'.join(dates)}"
            ofile = f"{ofile}.nc"
        else:
            ofile = args.ncdf_out

        if args.ncdf_format == "all_vars" or args.level == "1a":
            write_ncdf_L1(dftop, ofile, conf.CHANNEL, args.VERSION, image=not args.meta, level=args.level)
        else:
            if args.t:
                variables = ["EXPDate", "afsAttitudeState", "afsGnssStateJ2000", "NCSKIP", "NRSKIP", "NCOL", "NROW",
                             "NRBIN", "NCBINCCDColumns", "channel", "TEXPMS", "qprime"]
                dist_data = read_ncdf(args.dist_data, None, get_units=False)
                deg_map = get_deg_map(dftop.iloc[0], dist_data)
                tp_data = multiprocess(tp_calc_wrapper, dftop, variables, args.processes, [deg_map])
                num_img = len(tp_data)
                tp_vars = {name: np.stack([tp_data[k][name] for k in range(num_img)], axis=0)
                           for name in ["TPECEFx", "TPECEFy", "TPECEFz", "TPheightPixel"]}
                del tp_data
            else:
                tp_vars = None

            write_ncdf_L1b_zarr_format(dftop, ofile, conf.CHANNEL, args.VERSION, extra_vars=tp_vars)


if __name__ == "__main__":
    main()

from mats_utils.rawdata.read_data import read_MATS_data, read_MATS_PM_data, load_multi_parquet
from mats_l2_processing.parameters import make_conf
from mats_l2_processing.io import write_ncdf_L1b
from mats_l2_processing.util import get_filter
from mats_utils.rawdata.read_data import store_as_parquet
from mats_utils.rawdata.release import write_ncdf_L1b_release
import argparse
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
                        choices=["IR1", "IR2", "IR3", "IR4", "UV1", "UV2", "all", "PM"])
    parser.add_argument("--from_parquet_dir", type=str,
                        help="Get data from local parquet files from specifief dir., not AWS.")
    parser.add_argument("--conf", type=str, help="Read configuration from file.")
    parser.add_argument("--ncdf_out", type=str, help="Output file name (ncdf)")
    parser.add_argument("--release", type=str,
                        help="Make a release-type ncdf with the specified release eersion.")
    parser.add_argument("--parquet_out", type=str, help="Output file name (parquet)")
    parser.add_argument("--meta", action="store_true", help="Do not store images themselves, just metadata.")
    return parser.parse_args()


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
            dftop = read_MATS_data(DT.datetime(*conf.START_TIME), DT.datetime(*conf.STOP_TIME),
                                   get_filter(conf.CHANNEL), level="1b", version=conf.VERSION)
    # breakpoint()
    if args.ncdf_out:
        if args.release:
            write_ncdf_L1b_release(dftop, args.ncdf_out, conf.CHANNEL, conf.VERSION, args.release)
        else:
            write_ncdf_L1b(dftop, args.ncdf_out, conf.CHANNEL, conf.VERSION, im_calibrated=not args.meta)
    if args.parquet_out:
        store_as_parquet(dftop, f"{args.parquet_out}.parquet.gzip")
        # dftop.to_parquet(f"{args.parquet_out}.parquet.gzip", compression='gzip', engine='fastparquet')
    elif not args.ncdf_out:
        raise RuntimeError("Please specify output file name for at least one format (ncdf and/or parquet).")


if __name__ == "__main__":
    main()

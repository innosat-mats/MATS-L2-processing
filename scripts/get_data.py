from mats_utils.rawdata.read_data import read_MATS_data
from mats_l2_processing.parameters import make_conf
from mats_l2_processing.io import write_ncdf_L1b
from mats_l2_processing.util import get_filter
import argparse
import datetime as DT
import pickle


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
                        choices=["IR1", "IR2", "IR3", "IR4", "UV1", "UV2"])
    parser.add_argument("--conf", type=str, help="Read configuration from file.")
    parser.add_argument("--ncdf_out", type=str, help="Output file name (ncdf)")
    parser.add_argument("--pickle_out", type=str, help="Output file name (pickle)")
    return parser.parse_args()


def main():
    args = get_args()
    conf, _ = make_conf("get_data", args.conf, args)

    dftop = read_MATS_data(DT.datetime(*conf.START_TIME), DT.datetime(*conf.STOP_TIME),
                           get_filter(conf.CHANNEL), level="1b", version=conf.VERSION)
    if args.ncdf_out:
        write_ncdf_L1b(dftop, args.ncdf_out, conf.CHANNEL, conf.VERSION)
    if args.pickle_out:
        with open(args.pickle_out, 'wb') as handle:
            pickle.dump(dftop, handle)
    elif not args.ncdf_out:
        raise RuntimeError("Please specify output file name for at least one format (ncdf and/or pickle).")


if __name__ == "__main__":
    main()

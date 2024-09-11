from mats_utils.rawdata.read_data import all_hours, hours_filename
import numpy as np
import argparse
import datetime as DT
from os.path import isfile


def get_args():
    parser = argparse.ArgumentParser(description="Save MATS data as hourly parquet files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("START_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("STOP_TIME", type=int, nargs=6, help="Stop time for data set.")
    parser.add_argument("--channel", dest="CHANNEL", type=str, default="all", help="Channel to retrieve",
                        choices=["IR1", "IR2", "IR3", "IR4", "UV1", "UV2", "all"])
    parser.add_argument("--schedule", dest="SCHEDULE", type=str, help="Schedule name.")
    return parser.parse_args()


def main():
    args = get_args()

    hours = all_hours(DT.datetime(*args.START_TIME), DT.datetime(*args.STOP_TIME))
    fnames = [hours_filename(h) for h in hours]
    present = np.array([isfile(fname) for fname in fnames])
    next_present = np.roll(present, -1)
    next_present[-1] = 0
    previous_present = np.roll(present, 1)
    previous_present[0] = 0
    last_missing = np.logical_and(np.logical_not(present), next_present)
    first_missing = np.logical_and(np.logical_not(present), previous_present)
    intervals = zip(np.where(first_missing)[0], np.where(last_missing)[0])
    for intv in intervals:
        if intv[0] == intv[1]:
            print(f"{fnames[intv[1]]} missing!")
        else:
            print(f"{fnames[intv[0]]} - {fnames[intv[1]]} missing!")


if __name__ == "__main__":
    main()

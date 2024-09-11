from mats_utils.rawdata.read_data import read_MATS_data, store_as_parquet, all_hours, hours_filename
from mats_l2_processing.util import get_filter
import argparse
import datetime as DT


def get_args():
    parser = argparse.ArgumentParser(description="Save MATS data as hourly parquet files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("START_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("STOP_TIME", type=int, nargs=6, help="Stop time for data set.")
    parser.add_argument("VERSION", type=str, help="Data version.")
    parser.add_argument("--channel", dest="CHANNEL", type=str, default="all", help="Channel to retrieve",
                        choices=["IR1", "IR2", "IR3", "IR4", "UV1", "UV2", "all"])
    parser.add_argument("--schedule", dest="SCHEDULE", type=str, help="Schedule name.")
    return parser.parse_args()


def main():
    args = get_args()

    hours = all_hours(DT.datetime(*args.START_TIME), DT.datetime(*args.STOP_TIME))
    filt = get_filter(args.CHANNEL)
    if args.SCHEDULE:
        filt.update({"schedule_name": [args.SCHEDULE, args.SCHEDULE]})
    for h in hours:
        fname = hours_filename(h)
        print(f"Downloading {fname}...")
        try:
            df = read_MATS_data(h, h + DT.timedelta(hours=1), filt, level="1b", version=args.VERSION)
            store_as_parquet(df, fname)
        except Exception:
            print(f"No data for this version for {fname}!")


if __name__ == "__main__":
    main()

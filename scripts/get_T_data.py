import datetime as DT
import numpy as np
import argparse

from mats_l2_processing.obs import time_offsets, cross_maps, reinterpolate, remove_background
from mats_l2_processing.io import read_chn_from_pandas, add_ncdf_vars, write_ncdf_L1b
from mats_l2_processing.parameters import make_conf
from mats_l2_processing.util import get_filter
from mats_utils.rawdata.read_data import load_multi_parquet, read_MATS_data

CHNAMES = ["IR1", "IR2", "IR3", "IR4"]


def get_args():
    parser = argparse.ArgumentParser(description="Prepare data for temperature retrieval",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--start_time", dest="START_TIME", type=int, nargs=6, help="Start time for data set.")
    parser.add_argument("--stop_time", dest="STOP_TIME", type=int, nargs=6, help="Stop time for data set.")
    parser.add_argument("--version", dest="VERSION", type=str, help="Data version.")
    parser.add_argument("--aws", action="store_true",
                        help="Download data from AWS, rather than reading local files.")
    parser.add_argument("--parquet_dir", type=str, default='/mnt/data0/MATS_data/CROPD_v0.6',
                        help="Directory where to look for parquet files.")
    parser.add_argument("--conf", type=str, help="Read configuration from file.")
    parser.add_argument("--ncdf_out", type=str, default="IRc.nc", help="Output file name (ncdf).")
    return parser.parse_args()


def main():
    args = get_args()
    conf, const = make_conf("superpose", args.conf, args)

    # Read in data
    if args.aws:
        df = read_MATS_data(DT.datetime(*conf.START_TIME), DT.datetime(*conf.STOP_TIME),
                            get_filter(conf.CHANNEL), level="1b", version=conf.VERSION)
    else:
        df = load_multi_parquet(args.parquet_dir, DT.datetime(*conf.START_TIME), DT.datetime(*conf.STOP_TIME))
        with open(f"{args.parquet_dir}/version.txt", "r") as version_file:
            version = version_file.readlines()[0].replace("\n", "")
            if version != conf.VERSION:
                raise ValueError(f"Configuration specifies data version {conf.VERSION}, " +
                                 f"but loaded data of version {version}! Abort!")

    # Align images on different channels
    times = [df[df["channel"] == ch]["EXPDate"].to_numpy() for ch in CHNAMES]
    offsets, numimg = time_offsets(times)
    data = [read_chn_from_pandas(df, chn, offsets[i], numimg, const.ALL_VARS) for i, chn in enumerate(CHNAMES)]

    # Calculate the positions of IR1 pixels w.r.t. the image centres of other channels
    deg_maps, cmaps = cross_maps(data, const.CCD_VARS)
    # Interpolate all IR channels on IR1 grid
    reint = reinterpolate(data, deg_maps, cmaps)
    ir2, ir3, ir4 = [reint[i, :, :, :] for i in range(3)]
    ir1 = data[0]["ImageCalibrated"][:ir2.shape[0], :, :]

    if float(conf.VERSION) < 0.55:
        ir1, ir2, ir3, ir4 = [imag * 1000 / np.mean(data[i]["TEXPMS"]) for i, imag in enumerate([ir1, ir2, ir3, ir4])]
    else:
        ir1, ir2, ir3, ir4 = [x * 0.1 for x in [ir1, ir2, ir3, ir4]]

    # IR1 and IR2 on the same (IR1) grid with backgrounds removed
    ir1c, ir2c = remove_background(ir1, ir2, ir3, ir4, recal=conf.RECAL_FAC_IR)

    # Write a netCDF file with IR1 data
    df1 = df[df["channel"] == "IR1"]
    df1.reset_index(drop=True, inplace=True)
    df1 = df1[offsets[0]:(offsets[0] + numimg)]
    df1.reset_index(drop=True, inplace=True)
    write_ncdf_L1b(df1, args.ncdf_out, "IR1", conf.VERSION, im_calibrated=False)
    add_ncdf_vars(args.ncdf_out, "CalibrationErrors",
                  [("IR1", "IR1 calibrated image", ir1),
                   ("IR2", "IR2 calibrated image reinterpolated on IR1 grid", ir2),
                   ("IR3", "IR3 calibrated image reinterpolated on IR1 grid", ir3),
                   ("IR4", "IR4 calibrated image reinterpolated on IR1 grid", ir4),
                   ("IR1c", "IR1 calibrated image with background removed", ir1c),
                   ("IR2c", "IR2 calibrated image with background removed", ir2c)])


if __name__ == "__main__":
    main()

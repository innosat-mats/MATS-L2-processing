import numpy as np
import netCDF4 as nc
import argparse

from mats_l2_processing.io import read_L1_ncdf, add_ncdf_vars
from mats_l2_processing.obs_preprocessing import jpg_fail_proxy


def get_args():
    """ Parses command line arguments using argparse.

    Returns:
        argparse object with arguments.
    """

    parser = argparse.ArgumentParser(description="Calculate jpg failure proxy and add it to L1b netCDF file",
                                     formatter_class=argparse.MetavarTypeHelpFormatter)
    # Data source options
    parser.add_argument("file", type=str, help="Netcdf file to append. ")
    parser.add_argument("--img_var", type=str, default="ImageCalibrated",
                        help="Name of the variable with image data.")
    parser.add_argument("--out_var", type=str, nargs=3, default=["jpgf_proxy", "jpg_fail"],
                        help="Output var. names: for proxy (float for each img,), flag (bool for each img.)." +
                        " Pass empty string to not write that variable.")
    parser.add_argument("--TPh_var", type=str, default="TPheightPixel",
                        help="Name of the variable with TP height data, in meters.")
    parser.add_argument("--alt_range", type=float, nargs=2, default=[95, 105],
                        help="Altitude range to analyse, in km."),
    parser.add_argument("--threshold", type=float, default=2,
                        help="Minimum value of proxy to set the flag.")
    parser.add_argument("--col_range", type=float, nargs=2, default=(6, 42),
                        help="Range of columns to use for analysis")
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = get_args()
    write_proxy, write_flag = [len(x) > 0 for x in args.out_var]

    mdata = read_L1_ncdf(args.file, var=[args.img_var, args.TPh_var])
    jpgf = jpg_fail_proxy(mdata[args.img_var][:, :, args.col_range[0]:args.col_range[1]])

    if write_proxy:
        add_ncdf_vars(args.file, "TPlon",
                      [(args.out_var[0], "JPG compression failure proxy for each image.", jpgf)],
                      units=[(args.out_var[0], "")])

    if write_flag:
        # flag = pad_flag(iap > args.threshold, mdata["time_s"], args.time_pad)
        add_ncdf_vars(args.file, "TPlon",
                      [(args.out_var[1], "JPG compression failure flag for each image", jpgf > args.threshold)],
                      units=[(args.out_var[1], "")])


if __name__ == "__main__":
    main()

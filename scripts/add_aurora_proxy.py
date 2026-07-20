import numpy as np
import netCDF4 as nc
import argparse

from mats_l2_processing.io import read_L1_ncdf, add_ncdf_vars
from mats_l2_processing.obs_preprocessing import aurora_proxy


def get_args():
    """ Parses command line arguments using argparse.

    Returns:
        argparse object with arguments.
    """

    parser = argparse.ArgumentParser(description="Calculate aurora proxy and add it to L1b netCDF file",
                                     formatter_class=argparse.MetavarTypeHelpFormatter)
    # Data source options
    parser.add_argument("file", type=str, help="Netcdf file to append. ")
    parser.add_argument("--IR1_var", type=str, default="ImageFinal",
                        help="Name of the variable with IR1 data.")
    parser.add_argument("--out_var", type=str, nargs=3, default=["aurora_proxy", "aurora_id", "aurora"],
                        help="Output var. names: proxy \(per image\), full proxy(per pix.), flag (bool)." +
                        " Pass empty string to not write that variable.")
    parser.add_argument("--IR2_var", type=str, default="IR2c",
                        help="Name of the variable with IR2 data.")
    parser.add_argument("--TPh_var", type=str, default="TPheightPixel",
                        help="Name of the variable with TP height data, in meters.")
    parser.add_argument("--alt_range", type=float, nargs=2, default=[95, 105],
                        help="Altitude range to analyse, in km."),
    parser.add_argument("--denoise_iter", type=int, default=1,
                        help="Number of iterations of noise removal (binary erosion)")
    parser.add_argument("--shift", type=int, default=0,
                        help="Use sensor shift to supress noise")
    parser.add_argument("--threshold", type=int, default=4,
                        help="Minimum number of pixels with non-zero proxy value to set the flag.")
    parser.add_argument("--time_pad", type=float, default=90,
                        help="Temporal distance (in s) from detected aurora to be also flagged as aurora.")
    parser.add_argument("--col_range", type=float, nargs=2, default=(6, 42),
                        help="Range of columns to use for analysis")
    parser.add_argument("--aurora_rad_thr", type=float, default=8e13,
                        help="Images with min(IR1, IR2) above this value at the top are considered aurora.")

    parser.add_argument("--processes", type=int, default=1,
                        help="Number of CPUs to use for the calculation.")
    return parser.parse_args()


def pad_flag(detections, time, pad):
    max_idx_d = int(np.ceil(pad / np.percentile(np.diff(time), 1)))
    detected_idx = np.arange(len(detections))[detections]
    flags = detections.copy()
    for idx in detected_idx:
        start, stop = (idx - max_idx_d, idx + max_idx_d)
        flags[start:stop] = np.logical_or(np.abs(time[start:stop] - time[idx]) < pad, flags[start:stop])
    return flags


def main():
    # Parse command line arguments
    args = get_args()
    write_img, write_full, write_flag = [len(x) > 0 for x in args.out_var]

    mdata = read_L1_ncdf(args.file, var=[args.IR1_var, args.IR2_var, args.TPh_var])
    ap = aurora_proxy(mdata, vrange=args.alt_range, denoise_iter=args.denoise_iter, roll=args.shift, ir1v=args.IR1_var,
                      ir2v=args.IR2_var, tpv=args.TPh_var, nproc=args.processes, col_range=args.col_range,
                      aurora_thr=args.aurora_rad_thr)
    if write_full:
        add_ncdf_vars(args.file, args.IR1_var, [(f"{args.out_var[1]}_IR{i + 1}",
                                                "Aurora proxy for each pixel. Non-zero value indicates aurora.",
                                                ap[:, i, :, :]) for i in range(2)])
    if write_img or write_flag:
        iap = np.sum(np.minimum(ap[:, 0, :, args.col_range[0]:args.col_range[1]],
                                ap[:, 1, :, args.col_range[0]:args.col_range[1]]) > 0, axis=(1, 2))

    if write_img:
        add_ncdf_vars(args.file, "TPlon",
                      [(args.out_var[0], "Aurora proxy for each image. Non-zero value indicates aurora.", iap)],
                      units=[("aurora_proxy", "")])

    if write_flag:
        flag = pad_flag(iap > args.threshold, mdata["time_s"], args.time_pad)
        add_ncdf_vars(args.file, "TPlon",
                      [(args.out_var[2], "Aurora flag for each image. 1: aurora, 0: no aurora.", flag)],
                      units=[("aurora_proxy", "")])


if __name__ == "__main__":
    main()

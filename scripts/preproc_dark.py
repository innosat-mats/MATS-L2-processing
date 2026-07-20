import numpy as np
import netCDF4 as nc
import argparse

from mats_l2_processing import parameters
from mats_l2_processing.io import read_L1_ncdf, add_ncdf_vars
from mats_l2_processing.obs_preprocessing import jpg_fail_proxy, subtract_top_median, hot_pix_trans_masks, fill_invalid


def get_args():
    parser = argparse.ArgumentParser(description="Preprocessing for L1b nightglow",
                                     formatter_class=argparse.MetavarTypeHelpFormatter)
    # Data source options
    parser.add_argument("file", type=str, help="Netcdf file with L1b data to process")
    parser.add_argument("--hpix_file", type=str, help=".npz file with hot pixel maps")
    parser.add_argument("--conf", type=str, default="conf.py", help="Configuration file")

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = get_args()
    conf, const = parameters.make_conf("L1b_preproc_dark", args.conf, args)

    mdata = read_L1_ncdf(args.file, var=const.PREPROC_VARS)
    img = mdata["ImageCalibrated"] * conf.PREPROC_OBS_FACTOR

    if args.hpix_file:
        of = conf.OUTLIER_FILTER_SIGMAS if conf.OUTLIER_FILTER_SIGMAS > 0 else None
        masks, maskIdx = hot_pix_trans_masks(args.hpix_file, mdata["channel"][0], conf.HPIX_TRANS_THRESHOLD,
                                             mdata["time_s"])
        img, filled = fill_invalid(img, np.stack([masks[i, ...] for i in maskIdx], axis=0), outlier_filter=of)
        add_ncdf_vars(args.file, "ImageCalibrated", [("filled", "Pixels corrected in L1c processing", filled)],
                      units=[("filled", "")])

    jpgf = jpg_fail_proxy(img[:, :, conf.COL_RANGE[0]:conf.COL_RANGE[1]])
    add_ncdf_vars(args.file, "TPlon", [("jpgf_proxy", "JPG compression failure proxy for each image.", jpgf)],
                  units=[("jpgf_proxy", "")])
    add_ncdf_vars(args.file, "TPlon", [("jpg_fail", "JPG compression failure flag for each image",
                                        jpgf > conf.JPG_FAIL_THRESHOLD)], units=[("jpgf_proxy", "")])

    if conf.SUB_TOP_CONF["strategy"] is not None:
        img, medians = subtract_top_median(img, mdata["TPheightPixel"], conf.SUB_TOP_ALT_RANGE, conf=conf.SUB_TOP_CONF)
        add_ncdf_vars(args.file, "TPlon", [("sub_top_value", "Subtracted radiance (based on top of img.)", medians)],
                      units=[("sub_top_value", "photon nanometer-1 meter-2 steradian-1 second-1")])
    with nc.Dataset(args.file, 'r+') as nf:
        nf["ImageCalibrated"][:] = img


if __name__ == "__main__":
    main()

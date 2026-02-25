from mats_l2_processing.pointing import Pointing, TP_data
from mats_l2_processing.io import read_L1_ncdf, append_gen_ncdf
from mats_l2_processing import parameters
from mats_l2_processing.util import multiprocess
import argparse
from skyfield.api import load


def get_args():
    parser = argparse.ArgumentParser(description="Calculate geometric parameters for MATS L1 data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("l1", type=str,
                        help="Filename for L1 data (tomography result, ncdf or npz file).")
    parser.add_argument("--nproc", type=int, default=10, help="Numper of processes to run in parallel.")
    parser.add_argument("--lonlat", action="store_true",
                        help="Calculate longtitude and latitude for tangent points.")
    parser.add_argument("--sza", action="store_true",
                        help="Calculate solar zenith angle for tangent points.")
    return parser.parse_args()


def TP_data_wrapper(image, common_args):
    interpolated = TP_data(image, common_args[0], common_args[1], **common_args[2])
    return [interpolated[v] for v in common_args[1]]


def main():
    args = get_args()
    conf, const = parameters.make_conf("heights", "conf.py", {})
    metadata = read_L1_ncdf(args.l1, var=const.POINTING_DATA)
    pointing = Pointing(metadata, conf, const)
    kwargs = {"ny": 10, "planets_file": conf.PLANET_FILE}
    var = ["height"]
    if args.lonlat:
        var += ["lon", "lat"]
    if args.sza:
        var += ["sza"]

    mout = multiprocess(TP_data_wrapper, metadata, const.POINTING_DATA, args.nproc,
                        [pointing, var, kwargs], unzip=True, stack=False)

    dims = ("EXPDate", "im_row", "im_col")
    shape = [metadata["size"], -1, mout[0].shape[-1]]
    res = {v: mout[i].reshape(shape) for i, v in enumerate(var)}
    del mout

    out_spec = {"TPheight_dist": ["Tangent point heights, with distortion compensation", "meter", res["height"], dims]}
    if args.lonlat:
        out_spec["TPlat_dist"] = ["Tangent point latitude, with distortion compensation",
                                  "degree_north", res["lat"], dims]
        out_spec["TPlon_dist"] = ["Tangent point longitude, with distortion compensation",
                                  "degree_east", res["lon"], dims]
    if args.sza:
        out_spec["sza_dist"] = ["Solar zenith angle, with distortion compensation",
                                "degree_east", res["sza"], dims]

    append_gen_ncdf(args.l1, out_spec, overwrite=True)


if __name__ == "__main__":
    main()

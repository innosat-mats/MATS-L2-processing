import numpy as np
import argparse

from mats_l2_processing.io import add_ncdf_vars
from mats_l2_processing.atm import gridded_data
from mats_l2_processing import parameters


def get_args():
    parser = argparse.ArgumentParser(description="Perform preconfigured postprocessing on MATS L2 data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("L2_ncdf", type=str,
                        help="NetCDF file with L2 data for post processing. Results will be appended to it.")
    parser.add_argument("--conf", type=str, default="conf.py", help="Configuration file.")
    parser.add_argument("--extra_vars", type=str,
                        help="Extra input vars to read (comma separated list). Defaults are all results and a priori.")
    parser.add_argument("--ref_var", type=str, default="longitude",
                        help="Output variable dimensions will be as for this variable. Default works for most grids.")
    return parser.parse_args()


def main():
    # Parse arguments and configuration files
    args = get_args()
    conf, const = parameters.make_conf("post", args.conf, args)

    # Read in data and do the post processing
    extra_vars = args.extra_vars.split(",") if args.extra_vars else []
    preload_vars = conf.AUX_QTY + conf.RET_QTY + [f"{var}_apr" for var in conf.RET_QTY] + extra_vars
    gridded = gridded_data(conf.GRIDDED_POST, from_L2_ncdf=args.L2_ncdf, ncdf_preload=preload_vars)

    # Read the long names and units for the result variables from the configuration
    long_names, units, write = {}, {}, {}
    for entry in conf.GRIDDED_POST:
        num_vars = len(entry[0])
        spec_vars = [("long_name", long_names, ""), ("units", units, ""), ("write", write, True)]
        for vtype, dic, default in spec_vars:
            if vtype in entry[2].keys():
                assert len(entry[2][vtype]) == num_vars, \
                    f"GRIDDED_POST entry has {num_vars} vars, but {entry[2][vtype]} long names / units!"
                dic.update({entry[0][i]: entry[2][vtype][i] for i in range(num_vars)})
            else:
                dic.update({entry[0][i]: default for i in range(num_vars)})

    # Write results to file
    new_vars = [name for name in gridded.data.keys() if name not in preload_vars]
    new_data = [(name, long_names[name], gridded.data[name]) for name in new_vars if write[name]]
    units = [(name, units[name]) for name in new_vars if write[name]]
    add_ncdf_vars(args.L2_ncdf, args.ref_var, new_data, units=units)


if __name__ == "__main__":
    main()

import numpy as np
import netCDF4 as nc
import argparse
from matplotlib import pyplot as plt
from mats_l2_processing.util import seconds2DT


def get_args():
    parser = argparse.ArgumentParser(description="Plot stacked 1D retrievals",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("l2", type=str, default="L2_tomo.nc",
                        help="Filename for L2 data (tomography result, ncdf or npz file).")
    parser.add_argument("--var", type=str, default="VER", help="Pick data variable to plot.")
    parser.add_argument("--value_range", type=float, nargs=2, help="Range of values to plot.")
    parser.add_argument("--percentile_range", type=float, nargs=2, default=[1, 99],
                        help="Range of value percentiles to plot.")
    parser.add_argument("--title", type=str, help="Use custom plot title.")
    parser.add_argument("--divergent", action="store_true", help="Use symmetric divergent color scale.")
    parser.add_argument("--prefix", type=str, default="lin", help="Plotly filename prefix.")
    return parser.parse_args()


def read_ncdf(filename, var=None):
    res = {}
    with nc.Dataset(filename, 'r') as nf:
        if var is None:
            var = nf.variables()
        for name in var:
            res[name] = nf[name][:]
    return res


def main():
    args = get_args()
    L2 = read_ncdf(args.l2, var=["alt_coord", "img_time"] + [args.var])
    time = [seconds2DT(ts) for ts in L2["img_time"].data]
    alts = L2["alt_coord"] * 1e-3
    tt, aa = np.meshgrid(time, alts, indexing='ij')
    cmap = "coolwarm" if args.divergent else "inferno"

    if args.value_range:
        vmin, vmax = args.value_range
    elif args.percentile_range:
        vmin, vmax = [np.percentile(L2[args.var], p) for p in args.percentile_range]

    if args.divergent:
        vmax = np.maximum(np.abs(vmin), np.abs(vmax))
        vmin = - vmax

    plt.figure()
    plt.pcolor(tt, aa, L2[args.var], cmap=cmap, vmin=vmin, vmax=vmax)
    if args.title:
        plt.title(args.title)
    plt.xlabel("Observation time")
    plt.ylabel("Altitude, km")
    plt.colorbar()
    outfile = f"{args.prefix}_{args.var}"
    plt.savefig(f"{outfile}.png", dpi=400)


if __name__ == "__main__":
    main()

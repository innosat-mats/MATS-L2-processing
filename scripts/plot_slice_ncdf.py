import numpy as np
import argparse
from mats_l2_processing.io import read_ncdf
from mats_l2_processing.plotting import plot_slice


def get_args():
    parser = argparse.ArgumentParser(description="Plot tomography slice",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("l2", type=str, default="L2_tomo.nc",
                        help="Filename for L2 data (tomography result, ncdf or npz file).")
    parser.add_argument("var", type=str,
                        help="Pick data variable to plot (comma separated list)")
    parser.add_argument("--slice_axis", type=str, choices=["alt", "across", "along"], default="across",
                        help="Pick a direction across which to slice.")
    parser.add_argument("--slice_mean", type=float, nargs=2,
                        help="Show mean along slice_axis instead of a slice. Overrides slice_pos.")
    parser.add_argument("--slice_pos", type=float, default=0,
                        help="Pick a position at slice_axis at which to slice.")
    parser.add_argument("--prefix", type=str, default="slice", help="Slice plot filename prefix.")
    parser.add_argument("--cmap", type=str, help="Colormap to use")
    parser.add_argument("--pdf", action="store_true", help="Output pdf instead of png.")
    parser.add_argument("--title", type=str, help="Use custom plot title (comma separated list for each var)")
    parser.add_argument("--divergent", action="store_true", help="Use symmetric divergent color scale.")
    parser.add_argument("--val_range", type=float, nargs=2, help="Min and max for colorscale.")
    parser.add_argument("--percentile_range", type=float, nargs=2, default=[1, 99],
                        help="Min and max for colorscale.")
    parser.add_argument("--xrange", type=float, nargs=2, help="Range for x axis of 2-D plot")
    parser.add_argument("--yrange", type=float, nargs=2, help="Range for y axis of 2-D plot.")
    parser.add_argument("--log_scale", action="store_true", help="Use log scale for colormap")
    return parser.parse_args()


def main():
    args = get_args()
    argsd = args.__dict__.copy()

    coord_names = ["alt_coord", "acrosstrack_coord", "alongtrack_coord"]
    plot_vars = argsd.pop("var").split(",")

    slice_axis = argsd.pop("slice_axis")
    pos = [argsd.pop("slice_pos")]
    slice_mean = argsd.pop("slice_mean")
    pos = pos if slice_mean is None else slice_mean
    prefix = argsd.pop("prefix")
    extension = ".pdf" if argsd.pop("pdf") else ".png"

    titles = argsd.pop("title")
    if titles is None:
        titles = [None for v in plot_vars]
    elif len(titles).split() == 1:
        titles = [titles for v in plot_vars]
    else:
        titles = titles.split(",")
    if len(titles) != len(plot_vars):
        raise ValueError("If titles are a comma separated list, there must be as many titles as plot variables!")

    data, units = read_ncdf(argsd.pop("l2"), coord_names + plot_vars, get_units=True)
    coords = [data[name] for name in coord_names]
    for var, title in zip(plot_vars, titles):
        fname = None if prefix is None else f"{prefix}_{var}_{slice_axis}_{'-'.join([str(p) for p in pos])}{extension}"
        plot_slice(coords, data[var][0, ...], slice_axis, pos, fname=fname, title=title, cb_label=f"{var}, {units[var]}",
                   **argsd)


if __name__ == "__main__":
    main()

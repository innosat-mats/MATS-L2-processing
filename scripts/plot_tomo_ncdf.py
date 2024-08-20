import numpy as np
import netCDF4 as nc
import argparse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
import plotly.offline as poff
from plotly.subplots import make_subplots
# import plotly.io as pio
from plotly.express.colors import sample_colorscale
from mats_l2_processing.grids import sph2cart, geoid_radius
from scipy.signal import savgol_filter

EARTH_RADIUS = 6371000  # Used to convert horiz. distances to km, geoid used for proper alts


def get_args():
    parser = argparse.ArgumentParser(description="Plot tomography",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("l2", type=str, default="L2_tomo.nc",
                        help="Filename for L2 data (tomography result, ncdf or npz file).")
    parser.add_argument("--var", type=str, default="VER",
                        help="Pick data variable to plot.")
    parser.add_argument("--slice_axis", type=str, choices=["alt", "across", "along"],
                        help="Pick a direction across which to slice.")
    parser.add_argument("--slice_mean", type=float, nargs=2,
                        help="Show mean along slice_axis instead of a slice.")
    parser.add_argument("--slice_pos", type=float, default=0,
                        help="Pick a position at slice_axis at which to slice.")
    parser.add_argument("--slice_save", type=str, default="slice", help="Slice plot filename prefix.")
    parser.add_argument("--title", type=str, help="Use custom plot title.")

    parser.add_argument("--divergent", action="store_true", help="Use symmetric divergent color scale.")
    parser.add_argument("--plotly", action="store_true", help="Plot in 3-D with plotly.")
    parser.add_argument("--isos", type=float, nargs="+", default=[1e11, 3e11, 6e11, 1e12],
                        help="Isosurface values to plot")
    parser.add_argument("--cs_range", type=float, nargs=2, help="Min and max for colorscale.")
    parser.add_argument("--x_range", type=float, nargs=2, help="Range for x axis of 2-D/3-D plot")
    parser.add_argument("--y_range", type=float, nargs=2, help="Range for y axis of 2-D/3-D plot.")
    parser.add_argument("--z_range", type=float, nargs=2, help="Range for z axis of 3-D plot.")
    parser.add_argument("--opacity", type=float, default=0.5, help="Isosurface opacity.")
    parser.add_argument("--font_size", type=float, default=12, help="Plotly font size.")
    parser.add_argument("--log_scale", action="store_true", help="Use log scale for colormap")
    parser.add_argument("--subtract_alt_mean", action="store_true", help="Use log scale for colormap")
    parser.add_argument("--detrend_axis", type=str, choices=["alt", "across", "along"],
                        help="Detrend data with Savitzky-Golay filter in y direction.")
    parser.add_argument("--plotly_save", type=str, default="tomo", help="Plotly filename prefix.")
    return parser.parse_args()


def slice_id(grid, axis_idx, slice_pos):
    return np.argmin(np.abs(np.mean(grid[axis_idx[0]], axis=axis_idx[1:]) - slice_pos))


def plot_grid_slice(data, km_grid, args, vtitle):
    tdata = data.copy()
    axes_names = ("altitude", "across-track distance", "along track distance")

    if args.slice_axis == "alt":
        axis_idx = (0, 2, 1)
    elif args.slice_axis == "across":
        axis_idx = (1, 2, 0)
    else:
        axis_idx = (2, 1, 0)

    if args.slice_mean:
        # smin, smax = [slice_id(km_grid, axis_idx, pos) for pos in args.slice_mean]
        where = np.logical_and(km_grid[axis_idx[0]] > args.slice_mean[0], km_grid[axis_idx[0]] < args.slice_mean[1])
        plot_data = np.mean(tdata, axis=axis_idx[0], where=where)
        # plot_data = np.mean(tdata, axis_idx[0])
        title = f"Mean along {axes_names[axis_idx[0]]}"
        position = "mean"
    else:
        slice_idx = slice_id(km_grid, axis_idx, args.slice_pos)
        # np.argmin(np.abs(np.mean(km_grid[axis_idx[0]], axis=axis_idx[1:]) - args.slice_pos))
        plot_data = tdata.take(axis=axis_idx[0], indices=slice_idx)
        position = f"{args.slice_pos:.1f}"
        title = f"Slice across {axes_names[axis_idx[0]]} axis at {position} km"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xx, yy = [np.mean(km_grid[axis_idx[n]], axis=axis_idx[0]).T for n in [1, 2]]
    cs_range = args.cs_range if args.cs_range else (np.min(plot_data), np.max(plot_data))
    if args.log_scale:
        norm = LogNorm()
        vmin, vmax = None, None
    else:
        norm = None
        vmin, vmax = cs_range
    if args.divergent:
        cmap = "coolwarm"
        mabs = np.maximum(np.abs(vmin), np.abs(vmax))
        vmin, vmax = -mabs, mabs
    else:
        cmap = "inferno"
    plot = ax.pcolor(xx, yy, plot_data.T, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.set_xlabel(f"{axes_names[axis_idx[1]]}, km")
    ax.set_ylabel(f"{axes_names[axis_idx[2]]}, km")
    if args.x_range:
        ax.set_xlim(*args.x_range)
    if args.y_range:
        ax.set_ylim(*args.y_range)
    if args.slice_axis == "alt":
        ax.set_aspect('equal')
        cb_orientation = "horizontal"
        fig.set_figwidth(12)
    else:
        cb_orientation = "vertical"

    ftitle = title if args.title is None else args.title
    ax.set_title(ftitle)
    plt.colorbar(plot, orientation=cb_orientation)

    fname = f"{args.slice_save}_{vtitle}_{args.slice_axis}_{position}.png"
    plt.savefig(fname, dpi=400)


def plotly_3d(data, km_grid, args):
    plot_data = np.transpose(data, (2, 1, 0))
    zz, yy, xx = [np.transpose(km_grid[i], (1, 2, 0)) for i in range(3)]
    zz = np.broadcast_to(zz[int(zz.shape[0] / 2), int(zz.shape[1] / 2), :], zz.shape)
    # xxt, yyt, zzt = np.meshgrid(np.linspace(-200, 200, 41), np.linspace(-2000, 2000, 401), np.linspace(50, 120, 71))
    bounds = [(arr.min(), arr.max()) for arr in km_grid]
    for i, b in enumerate([args.z_range, args.y_range, args.x_range]):
        if b is not None:
            bounds[i] = tuple(b)
    # test_data = ((zz - 90) / 3) ** 2 + (yy / 100) ** 2 + (xx / 20) ** 2

    fig = make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]])
    colors = [sample_colorscale("Blues", (iso - args.cs_range[0]) / (args.cs_range[1] - args.cs_range[0]))
              for iso in args.isos]

    for i, iso in enumerate(args.isos):
        col = colors[i][0] if iso > 0 else 'red'
        print(iso)
        fig.add_trace(go.Isosurface(x=xx.flatten(), y=yy.flatten(), z=zz.flatten(), value=plot_data.flatten(),
                                    isomin=iso, isomax=iso, opacity=args.opacity, surface_count=1, showscale=False,
                                    colorscale=[[0, col], [1, col]], lighting=dict(specular=0.2),
                                    caps=dict(x_show=False, y_show=False, z_show=False, x_fill=0), visible=True))

    fig.update_layout(scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=5, z=3),
                      xaxis=dict(range=bounds[2]), yaxis=dict(range=bounds[1]), zaxis=dict(range=bounds[0]),
                      xaxis_title=dict(text='across-track distance, km', font=dict(size=args.font_size * 1.5)),
                      yaxis_title=dict(text='along-track distance, km', font=dict(size=args.font_size * 1.5)),
                      zaxis_title=dict(text='altitude, km', font=dict(size=args.font_size * 1.5))
                      ))
    poff.plot(fig, filename=f"{args.plotly_save}.html")


def get_local_cartesians(radius_grid, acrosstrack_grid, alongtrack_grid, ecef_to_local):
    # shape = (len(radius_grid),len(acrosstrack_grid),len(alongtrack_grid))
    # Non-uniform grid
    # nu_grid = {key: np.zeros(shape) for key in ["x", "y", "z", "lat", "alt"]}

    rr, acrr, alongg = np.meshgrid(radius_grid, acrosstrack_grid, alongtrack_grid, indexing="ij")
    lxx, lyy, lzz = sph2cart(rr, acrr, alongg)
    glgrid = ecef_to_local.inv().apply(np.dstack((lxx.flatten(), lyy.flatten(), lzz.flatten()))[0, :, :])
    altt = rr - geoid_radius(np.arcsin(glgrid[:, 2].reshape(rr.shape) / rr))
    return [np.transpose(arr, axes=(0, 2, 1)) for arr in
            [altt, alongg * EARTH_RADIUS, acrr * EARTH_RADIUS]]


def mean_alt_profile(data, alts):
    assert data.shape == alts.shape
    edges = np.arange(int(alts.min()) - 0.5, int(alts.max()) + 0.5, 1)
    counts, _ = np.histogram(alts, bins=edges)
    sums, _ = np.histogram(alts, bins=edges, weights=data)
    mean_prof = np.where(counts > 0, sums / counts, 0.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    res = np.zeros_like(data)
    amean = np.mean(mean_prof)
    reint = np.interp(alts, centers, mean_prof)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            res[:, i, j] = reint[:, i, j] * (np.mean(data[:, i, j]) / amean)
    # res = np.where(valid, np.interp(alts, centers, mean_prof) * factor, 0.0)
    return res


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

    dvars = []
    for item in args.var.split(":"):
        dvars += item.split(",")

    if args.l2.endswith(".nc"):
        L2 = read_ncdf(args.l2, var=["altitude", "acrosstrack_coord", "alongtrack_coord"] + dvars)
    elif args.l2.endswith(".npz"):
        for dvar in dvars:
            assert dvar in ["VER", "T"], f"{dvar} is not stored in jacobian/simulated data!"
        L2f = np.load(args.l2)
        L2 = {}
        for f in L2f.files:
            L2[f] = L2f[f]
    else:
        raise ValueError("Invalid L2 file! Please provide netCDF or numpy npz file!")
    km_grid = [.001 * np.broadcast_to(x, L2["altitude"].shape) for x in
               [L2["altitude"], L2["acrosstrack_coord"][np.newaxis, :, np.newaxis],
                L2["alongtrack_coord"][np.newaxis, np.newaxis, :]]]

    for v, item in enumerate(args.var.split(":")):
        arg = item.split(",")
        if len(arg) == 2:
            data = L2[arg[0]] - L2[arg[1]]
            title = f"{arg[0]}_difference"
        elif len(arg) == 1:
            data = L2[arg[0]]
            title = arg[0]
        else:
            raise ValueError("Invalid plotting variables specified!")
        if args.detrend_axis:
            numaxis = {"alt": 0, "across": 1, "along": 2}
            data = data - savgol_filter(data, 51, 3, axis=numaxis[args.detrend_axis])
            title = f"{arg[0]}_residual"
        if len(data.shape) == 4 and data.shape[0] == 1:
            data = data[0, ...]
            if v == 0:
                km_grid = [arr[0, ...] for arr in km_grid]
        if args.subtract_alt_mean:
            data -= mean_alt_profile(data, km_grid[0])
        if args.plotly:
            if len(args.var.split(";")) > 1:
                raise ValueError("Please only specify one variable for 3D plots!")
            plotly_3d(data, km_grid, args)
        if args.slice_axis is not None:
            plot_grid_slice(data, km_grid, args, title)


if __name__ == "__main__":
    main()

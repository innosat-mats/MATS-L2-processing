import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


def plot_2d(x_coord, y_coord, data, fname=None, val_range=None, percentile_range=[1, 99],
            divergent=False, xlabel=None, ylabel=None, cb_label=None, title=None, xrange=None,
            yrange=None, log_scale=False, dpi=400, cmap=None):
    assert len(x_coord.shape) == 1, f"{x_coord.shape}"
    assert len(y_coord.shape) == 1, f"{y_coord.shape}"
    assert len(data.shape) == 2, f"{data.shape}"
    assert x_coord.shape[0] == data.shape[0], f"{x_coord.shape[0]}, {data.shape[0]}"
    assert y_coord.shape[0] == data.shape[1], f"{y_coord.shape[0]}, {data.shape[1]}"

    xx, yy = np.meshgrid(x_coord, y_coord, indexing='ij')
    if cmap is None:
        cmap = "coolwarm" if divergent else "inferno"

    if val_range is not None:
        vmin, vmax = val_range
    else:
        vmin, vmax = [np.percentile(data, p) for p in percentile_range]

    if divergent:
        vmax = np.maximum(np.abs(vmin), np.abs(vmax))
        vmin = - vmax

    if log_scale:
        vmin, vmax = None, None
        norm = LogNorm()
    else:
        norm = None

    plt.figure()
    plt.pcolor(xx, yy, data, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    cb = plt.colorbar()
    if cb_label is not None:
        cb.set_label(cb_label)
    if xrange is not None:
        plt.xlim(xrange)
    if yrange is not None:
        plt.ylim(yrange)

    if fname is None:
        plt.show()
    else:
        dpi_val = None if fname.endswith(".pdf") else dpi
        plt.savefig(fname, dpi=dpi_val)


def grid_slice(coords, data, ax_idx, pos, stat="mean"):
    if len(pos) == 2:
        pos_idx = [np.argmin(np.abs(coords[ax_idx] - p)) for p in pos]
    elif len(pos) == 1:
        pos_idx = [np.argmin(np.abs(coords[ax_idx] - pos))]
    else:
        raise ValueError("Slice position must be either interval or single value (list/tuple of length 1 or 2).")

    if len(pos_idx) == 2 and pos_idx[0] != pos_idx[1]:
        pos_idx = pos_idx.sort()
        res = np.take(data, list(range(pos_idx[0]), pos_idx[1]), axis=ax_idx)
        if stat == "mean":
            res = np.mean(res, axis=ax_idx)
        elif stat == "median":
            res = np.median(res, axis=ax_idx)
        else:
            raise ValueError(f"Invalid slice statistic {stat}!")
    else:
        res = np.take(data, pos_idx[0], axis=ax_idx)
    return res


def plot_slice(coords, data, slice_axis, pos, **kwargs):
    axes_short_names = ['alt', 'across', 'along']
    axes_labels = ("altitude, km", "across-track distance, km", "along track distance, km")
    dcoords = [c.copy() * 1e-3 for c in coords]

    if slice_axis in axes_short_names:
        ax_idx = axes_short_names.index(slice_axis)
    else:
        raise ValueError("Invalid axis specified!")

    pdata = grid_slice(dcoords, data, ax_idx, pos, stat=kwargs.pop("mean", None))

    plot_coord_idx = list(range(3))
    plot_coord_idx.remove(ax_idx)
    plot_coord_idx.reverse()
    for idx, label in zip(plot_coord_idx, ["xlabel", "ylabel"]):
        if label not in kwargs.keys():
            kwargs[label] = axes_labels[idx]

    plot_2d(dcoords[plot_coord_idx[0]], dcoords[plot_coord_idx[1]], pdata.T, **kwargs)

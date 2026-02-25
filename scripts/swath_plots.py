import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

from mats_l2_processing.util import ecef2wgs84, seconds2DT
import sys

from mats_l2_processing.io import read_ncdf


def select_period(time, data, index, step=86400):
    if index is None:
        return data
    else:
        start, stop = [np.argmin(np.abs(time - (time[0] + step * idx))) for idx in [index, index + 1] ]
        return data[start:stop, ...]


def normalize(vec):
    return vec / np.sqrt(vec[..., 0] ** 2 + vec[..., 1] ** 2 + vec[..., 2] ** 2)


def vlen(vec):
    return np.sqrt(vec[..., 0] ** 2 + vec[..., 1] ** 2 + vec[..., 2] ** 2)


def dot(v1, v2):
    return v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1] + v1[..., 2] * v2[..., 2]


def cross(v1, v2):
    res = np.zeros_like(v1)
    res[..., 0] = v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1]
    res[..., 1] = v1[..., 0] * v2[..., 2] - v1[..., 2] * v2[..., 0]
    res[..., 2] = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    return res


def hor_dist(p1, p2):
    vec = p2 - p1
    return np.sqrt(np.dot(vec, vec) - np.dot(vec, normalize(0.5 * (p1 + p2))) ** 2)


def im_swath(TPpos, TPheight, req_hrange):
    lims = [0, 1]
    signs = [-1, 1]
    for j, b in enumerate([0, 2]):
        vrange = [(TPheight[b, i] - req_hrange[j]) * signs[j] for i in [0, 2]]
        node = vrange[0] / (vrange[0] - vrange[1])
        if vrange[1] - vrange[0] > 0:
            lims[0] = node
        else:
            lims[1] = node
    lims = [np.maximum(lims[0], 0), np.minimum(lims[1], 1)]
    rel_center = 0.5 * (lims[0] + lims[1])
    rel_swath = np.maximum(lims[1] - lims[0], 0)

    swath = hor_dist(TPpos[1, 0, :], TPpos[1, 2, :]) * rel_swath
    center = TPpos[0, 1, :] * (1 - rel_center) + rel_center * TPpos[2, 1, :]
    return swath, center


def off_swath(sat, pos0, pos1):
    normal = normalize(cross(pos0 - sat, pos0))
    return np.abs(dot(normal, pos1 - pos0))


def off_swath2(sat, pos0, pos1):
    normal = normalize(cross(pos0, sat))
    return np.abs(np.dot(pos1 - pos0, normal))


def off_swath3(sat0, sat1, pos0, pos1):
    normal = normalize(cross(pos0 - sat0, pos0 - sat1))
    return np.abs(np.dot(pos1 - pos0, normal))


def off_swath4(sat0, sat1, pos0, pos1):
    v1 = normalize(cross(sat0 - pos0, pos0))
    v2 = normalize(cross(sat1 - pos0, pos0))

    res = 2 * vlen(pos1 - pos0) * vlen(v1 - v2)
    #res = vlen(pos1 - pos0) * dot(v2 - v1, normalize(pos0))
    # breakpoint()
    return res

def pair_swath(s1, s2, d):
    factor = np.maximum(1 - 2 * d / (s1 + s2), 0)
    return np.minimum(s1, s2) * factor


def remove_lon_jumps(lon):
    diff = np.zeros_like(lon)
    diff[:-1] = np.diff(lon)
    return np.where(diff > 180, np.nan, lon)


def combined_swaths(satpos, im_centers, im_swaths, times, duration=120.0, step=6.0):
    num_img = len(im_swaths)
    res = np.zeros(num_img)
    edges = np.zeros((num_img, 2), dtype=int)

    img_per_tomo = int(duration / step) - 1
    for i in range(num_img):
        edges[i, :] = [np.argmin(np.abs(times - times[i] - sgn * duration / 2))
                       for sgn in [-1, 1]]
        if edges[i, 1] - edges[i, 0] < img_per_tomo:
            edges[i, :] = -1

    of_swaths = np.zeros(num_img)

    for im in range(num_img):
        if edges[im, 0] < 0:
            res[im] = np.nan
        elif np.minimum(im_swaths[edges[im, 0]], im_swaths[edges[im, 1]]) < 1:
            res[im] = 0
        else:
            #of_swath = off_swath2(satpos[edges[im, 0], :], im_centers[edges[im, 0], :], im_centers[edges[im, 1], :])
            of_swath = off_swath4(satpos[edges[im, 0], :], satpos[edges[im, 1], :], im_centers[im, :], im_centers[edges[im, 1], :])
            of_swaths[im] = of_swath
            res[im] = pair_swath(im_swaths[edges[im, 0]], im_swaths[edges[im, 1]], of_swath)
    return res


def plot_curves(curves, times, swath, fname=None, period=None):
    dttimes = select_period(times, np.array([seconds2DT(t) for t in times]), period)

    plt.figure(figsize=(18, 12))
    for name, data, col in curves:
        plt.plot(dttimes, select_period(times, data, period),
                 color=col, label=name, linewidth=0.5)
    plt.legend()
    plt.xlabel("Observation time, s")
    plt.xticks(rotation='vertical')
    plt.ylabel("km")
    plt.ylim(-10, 270)
    plt.grid()

    pswath = select_period(times, swath, period)
    for lims, color in [((100, 200), "gold"), ((200, 300), 'g')]:
        plt.plot(dttimes, np.where(np.logical_and(pswath > lims[0], pswath < lims[1]), -5, np.nan), c=color, linewidth=3)

    #plt.plot(dttimes, np.where(np.logical_and(swath > 100, swath < 200), -5, np.nan), c='yellow', linewidth=3)
    #plt.plot(dttimes, np.where(swath > 200, -5, np.nan), c='g', linewidth=3)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)


def plot_map(lines, times, TPlon, TPlat, satlon, satlat, fname=None, period=None):
    pTPlon, pTPlat, psatlon, psatlat = [select_period(times, np.rad2deg(arr), period)
                                        for arr in [TPlon, TPlat, satlon, satlat]]
    pTPlon, psatlon = [remove_lon_jumps(lon) for lon in [pTPlon, psatlon]]

    plt.figure(figsize=(20, 12))
    for p, transf in enumerate([ccrs.PlateCarree(), ccrs.NorthPolarStereo(central_longitude=0)]):
        ax = plt.subplot(1, 2, p + 1, projection=transf)
        cl = ax.coastlines(color='black')
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')

        for name, stat, lims, color in lines:  # [(0, (100, 200), "yellow", '-'), ((200, 300), 'g')]:
            pstat = select_period(times, stat, period)
            selection = np.logical_and(pstat > lims[0], pstat < lims[1])
            ax.plot(np.where(selection, pTPlon, np.nan), pTPlat, c=color, linestyle='-',
                       label=f"centr. TP, {name}", transform=ccrs.PlateCarree())
            ax.plot(np.where(selection, psatlon, np.nan), psatlat, c=color, linestyle='--',
                       label=f"sat. track, {name}", transform=ccrs.PlateCarree())
        if p == 0:
            ax.legend(bbox_to_anchor=(0.5, 1.1), loc='lower center')
        else:
            ax.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)


req_hrange = [70e3, 100e3]

data = read_ncdf(sys.argv[1], ["EXPDate", "satpos", "TPpos", "TPheight"])
data["TPpos"] = np.swapaxes(data["TPpos"], 1, 2)
data["TPheight"] = np.swapaxes(data["TPheight"], 1, 2)
num_img = len(data["EXPDate"])
im_swaths = np.zeros(num_img)
im_centers = np.zeros((num_img, 3))

for p in range(num_img):
    im_swaths[p], im_centers[p, :] = im_swath(data["TPpos"][p, ...],
                                              data["TPheight"][p, ...],
                                              req_hrange)

swaths = combined_swaths(data["satpos"], im_centers, im_swaths, data["EXPDate"])
TPlat, TPlon, TPh = [x.reshape(data["TPpos"].shape[:-1]) for x in
                     ecef2wgs84(data["TPpos"].reshape(-1, 3))]
satlat, satlon, sath = ecef2wgs84(data["satpos"])
plt.figure()
plt.plot(TPlon[:, 1, 1], TPlat[:, 1, 1], c='b')
plt.plot(satlon, satlat, c='r')
plt.show()

lines_plot = [("TP altitude", data["TPheight"][:, 1, 1] * 1e-3, 'r'),
              ("Single image swath width", im_swaths * 1e-3, 'g'),
              ("Tomography swath width", swaths * 1e-3, 'b')]
lines_map = [(" tomo. swath > 200 km", swaths * 1e-3, (200, 300), 'g'),
             (" 100 km < tomo. swath < 200 km", swaths * 1e-3, (100, 200), 'gold'),
             (" tomo. swath < 100 km, im. swath > 100 km", np.where(swaths < 1e5, im_swaths, 0) * 1e-3,
              (100, 300), 'darkorange'),]
             #("im. swath < 100 km", swaths * 1e-3, (-1, 100), 'r')]


for pl in range(int(np.ceil((data["EXPDate"][-1] - data["EXPDate"][0]) / 86400))):
    plot_curves(lines_plot, data["EXPDate"], swaths * 1e-3, f'{sys.argv[2]}_{pl + 1:01d}_swath.pdf', period=pl)
    plot_map(lines_map, data["EXPDate"], TPlon[:, 1, 1], TPlat[:, 1, 1], satlon, satlat,
             fname=f'{sys.argv[2]}_{pl + 1:01d}_map.pdf', period=pl)

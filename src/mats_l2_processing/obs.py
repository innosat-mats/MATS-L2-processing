import numpy as np
import netCDF4 as nc
import logging
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RectBivariateSpline

from mats_l1_processing.pointing import pix_deg
from mats_l2_processing.util import get_image


def get_deg_map(image):
    dmap = np.zeros((image["NCOL"] + 1, image["NROW"], 2))
    for i in range(dmap.shape[0]):
        for j in range(dmap.shape[1]):
            dmap[i, j, :] = pix_deg(image, i, j)
    return dmap


def time_offsets(times):
    # Check temporal alignement, remove images from the beginning if necessary to align
    numtimes = len(times)
    stimes = []
    for chn, time in enumerate(times):
        stimes.append(time)
        if chn == 0:
            dt = np.median(np.diff(time))
            numimg = len(stimes[-1])
        else:
            numimg = np.minimum(numimg, len(stimes[-1]))

    offsets = np.zeros(numtimes, dtype=int)
    for i in range(1, numtimes):
        deltas = stimes[0][:numimg] - stimes[i][:numimg]
        offsets[i] = np.rint(np.median(deltas) / dt)

    min_offset = np.min(offsets)
    if min_offset < 0:
        offsets -= min_offset
    numimg = np.min([len(stimes[i]) - offsets[i] for i in range(numtimes)])

    for i in range(numtimes):
        rtimes = stimes[i][offsets[i]:(offsets[i] + numimg)]
        if i == 0:
            rtimes0 = rtimes
        if np.max(np.abs(rtimes - rtimes0)) > dt / 2:
            logging.warning(f"Misaligned channel #{i}")
            for st in [f"{x[0].minute}:{x[0].second}.{x[0].microsecond:2d} - " +
                       f'{x[1].minute}:{x[1].second}.{x[1].microsecond:2d} ' for x in zip(rtimes, rtimes0)]:
                logging.warning(st)
            raise ValueError("Could not align channels, maybe there are gaps in data?")
    return offsets, numimg


def cross_maps(data, image_vars):
    deg_maps = [get_deg_map(get_image(data[chn], 0, image_vars)) for chn in range(len(data))]
    rel_deg_map = np.zeros((len(deg_maps), deg_maps[0].shape[0], deg_maps[0].shape[1], 2))
    for chn in range(len(data)):
        # rel_qprime = R.inv(R.from_quat(data[chn]['qprime'][0, ...])) * R.inv(R.from_quat(data[chn]["afsAttitudeState"]
        # [0, ...])) * R.from_quat(data[0]["afsAttitudeState"][0, ...]) * R.from_quat(data[0]['qprime'][0, ...])
        rel_qprime = R.inv(R.from_quat(data[chn]['qprime'][0, ...])) * R.from_quat(data[0]['qprime'][0, ...])
        for i in range(rel_deg_map.shape[1]):
            for j in range(rel_deg_map.shape[2]):
                r = rel_qprime * R.from_euler('xyz', [0, deg_maps[0][i, j, 1], deg_maps[0][i, j, 0]], degrees=True)
                _, rel_deg_map[chn, i, j, 1], rel_deg_map[chn, i, j, 0] = r.as_euler('xyz', degrees=True)
    return deg_maps, rel_deg_map


def reinterpolate(data, deg_maps, cm):
    assert len(deg_maps) == cm.shape[0]
    numimgs = [len(data[i]["EXPDate"]) for i in range(len(data))]
    numimg = min(numimgs)
    numchn = len(numimgs)
    res = np.zeros((len(deg_maps) - 1, numimg, deg_maps[0].shape[1],
                    deg_maps[0].shape[0]))
    for chn in range(1, numchn):
        x = deg_maps[chn][:, int(deg_maps[chn].shape[1] / 2), 0]
        y = deg_maps[chn][int(deg_maps[chn].shape[0] / 2), :, 1]
        print(f"Interpolating channel {chn}/{len(deg_maps)}, {numimg} images")
        for im in range(numimg):
            interp = RectBivariateSpline(y, x, data[chn]["ImageCalibrated"][im, ...])
            res[chn - 1, im, :, :] = interp(cm[chn, :, :, 1], cm[chn, :, :, 0], grid=False).T
    return res


def remove_background(ir1, ir2, ir3, ir4, recal=None):
    ir1c, ir2c = ir1.copy(), ir2.copy()
    if recal is not None:
        ir1c, ir2c, ir3, ir4 = [recal[i] * arr for i, arr in enumerate([ir1c, ir2c, ir3, ir4])]
    ir3_off, ir4_off = [np.mean(arr[:, -11:-7, :], axis=1)[:, np.newaxis, :] / 1.05 for arr in [ir3, ir4]]
    bgr = (ir3 - ir3_off + ir4 - ir4_off) / 2
    ir1c -= bgr
    ir2c -= bgr
    return ir1c * 3.57, ir2c * 8.16


def remove_background_sep(ir1, ir2, ir3, ir4, recal=None):
    ir1c, ir2c = ir1.copy(), ir2.copy()
    if recal is not None:
        ir1c, ir2c, ir3, ir4 = [recal[i] * arr for i, arr in enumerate([ir1c, ir2c, ir3, ir4])]
    ir3_off, ir4_off = [np.mean(arr[:, -11:-7, :], axis=1)[:, np.newaxis, :] / 1.05 for arr in [ir3, ir4]]
    # bgr = (ir3 - ir3_off + ir4 - ir4_off) / 2
    ir1c -= ir4 - ir4_off
    ir2c -= ir3 - ir3_off
    return ir1c * 3.57, ir2c * 8.16

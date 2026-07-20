import numpy as np
import netCDF4 as nc
import datetime as DT
from bisect import bisect
import logging
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RectBivariateSpline, interpn, LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import deconvolve
from scipy.optimize import curve_fit
from scipy.ndimage import binary_erosion

# from mats_l1_processing.pointing import pix_deg
from mats_l2_processing.pointing import Pointing
from mats_l2_processing.util import get_image, multiprocess, DT2seconds


#def get_deg_map(image):
#    dmap = np.zeros((image["NCOL"] + 1, image["NROW"], 2))
#    for i in range(dmap.shape[0]):
#        for j in range(dmap.shape[1]):
#            dmap[i, j, :] = pix_deg(image, i, j)
#    return dmap


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


def cross_maps(data, image_vars, deg_maps):
    # deg_maps = [Pointing(data[chn], conf, const).chn_map(data[chn]["channel"]) for chn in range(len(data))]
    rel_deg_map = np.zeros((len(deg_maps), deg_maps[0].shape[0], deg_maps[0].shape[1], 2))
    for chn in range(len(data)):
        # rel_qprime = R.inv(R.from_quat(data[chn]['qprime'][0, ...])) * R.inv(R.from_quat(data[chn]["afsAttitudeState"]
        # [0, ...])) * R.from_quat(data[0]["afsAttitudeState"][0, ...]) * R.from_quat(data[0]['qprime'][0, ...])
        rel_qprime = R.inv(R.from_quat(data[chn]['qprime'][0, ...])) * R.from_quat(data[0]['qprime'][0, ...])
        for i in range(rel_deg_map.shape[1]):
            for j in range(rel_deg_map.shape[2]):
                r = rel_qprime * R.from_euler('xyz', [0, deg_maps[0][i, j, 1], deg_maps[0][i, j, 0]], degrees=True)
                _, rel_deg_map[chn, i, j, 1], rel_deg_map[chn, i, j, 0] = r.as_euler('xyz', degrees=True)
    return rel_deg_map


def reinterpolate(data, deg_maps, cm):
    assert len(deg_maps) == cm.shape[0]
    numimgs = [len(data[i]["time"]) for i in range(len(data))]
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


def reinterpolate2(data, deg_maps, cm, destray=[]):
    assert len(deg_maps) == cm.shape[0]
    numimgs = [len(data[i]["time"]) for i in range(len(data))]
    numimg = min(numimgs)
    numchn = len(numimgs)
    if len(destray) == 0:
        destray = [True for i in range(numchn - 1)]
    else:
        assert len(destray) == numchn - 1

    res = np.zeros((len(deg_maps) - 1, numimg, deg_maps[0].shape[1],
                    deg_maps[0].shape[0]))
    for chn in range(1, numchn):
        image_var = "ImageDestrayed" if destray[chn - 1] else "ImageCalibrated"
        x = deg_maps[chn][:, int(deg_maps[chn].shape[1] / 2), 0]
        y = deg_maps[chn][int(deg_maps[chn].shape[0] / 2), :, 1]
        print(f"Interpolating channel {chn}/{len(deg_maps)}, {numimg} images")
        cmf = np.flip(cm, axis=-1)
        for im in range(numimg):
            # breakpoint()
            res[chn - 1, im, :, :] = interpn([y, x], data[chn][image_var][im, ...],
                                             np.swapaxes(cmf[chn, :, :, :], 0, 1),
                                             method='linear', bounds_error=False, fill_value=0)
    return res


def reinterpolate3(data, deg_maps, cm, destray=[]):
    assert len(deg_maps) == cm.shape[0]
    numimgs = [len(data[i]["time"]) for i in range(len(data))]
    numimg = min(numimgs)
    numchn = len(numimgs)
    if len(destray) == 0:
        destray = [True for i in range(numchn - 1)]
    else:
        assert len(destray) == numchn - 1

    res = np.zeros((len(deg_maps) - 1, numimg, deg_maps[0].shape[0],
                    deg_maps[0].shape[1]))
    for chn in range(1, numchn):
        image_var = "ImageDestrayed" if destray[chn - 1] else "ImageCalibrated"
        # x, y = [deg_maps[chn][:, :, i].flatten() for i in range(2)]
        print(f"Interpolating channel {chn}/{len(deg_maps)}, {numimg} images")
        #cmf = np.flip(cm, axis=-1)
        # cms = np.swapaxes(cmf[chn, :, :, :], 0, 1)
        for im in range(numimg):
            # print(f"Image {im + 1}/{numimg}")
            # breakpoint()
            # res[chn - 1, im, :, :] = interpn([y, x], data[chn][image_var][im, ...],
            # #                                 np.swapaxes(cmf[chn, :, :, :], 0, 1),
            #                                 method='linear', bounds_error=False, fill_value=0)
            ip = CloughTocher2DInterpolator(deg_maps[chn].reshape(-1, 2), data[chn][image_var][im, ...].flatten(),
                                      fill_value=np.nan)
            res[chn - 1, im, :, :] = ip(cm[chn, :, :, 0].flatten(), cm[chn, :, :, 1].flatten()).reshape(deg_maps[0].shape[:-1])
            # ip = RBFInterpolator(deg_maps[chn].reshape(-1, 2), data[chn][image_var][im, ...].flatten(), neighbors=50, kernel="cubic")
            # res[chn - 1, im, :, :] = ip(cm[chn, ...].reshape(-1, 2)).reshape(deg_maps[0].shape[:-1])
    return edge_fix(res)


def reinterpolate_irreg(data, idxs, deg_map, cmap):
    numimages = np.sum(idxs > -1)
    res = np.zeros((numimages, cmap.shape[0], cmap.shape[1]))
    print(f"Interpolating {len(idxs)} images...")
    i = 0
    for idx in idxs:
        if idx < 0:
            continue
        ip = CloughTocher2DInterpolator(deg_map.reshape(-1, 2), data[idx, ...].flatten(),
                                        fill_value=np.nan)
        res[i, :, :] = ip(cmap[:, :, 0].flatten(),
                          cmap[:, :, 1].flatten()).reshape(cmap.shape[:-1])
        i += 1
    return edge_fix(res)


def edge_fix(data, fill=0):
    res = data.copy()

    cols = np.arange(data.shape[2], dtype=int)
    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            test = np.where(np.isnan(res[j, k, :]), np.nan, cols)
            if np.isnan(test).all():
                res[j, k, :] = fill
                continue
            amin, amax = int(np.nanmin(test)) + 1, int(np.nanmax(test)) - 1
            res[j, k, :amin] = res[j, k, amin]
            res[j, k, amax:] = res[j, k, amax]
    return res


def edge_fix_old(data, fill=0):
    res = data.copy()

    cols = np.arange(data.shape[3], dtype=int)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                test = np.where(np.isnan(res[i, j, k, :]), np.nan, cols)
                if np.isnan(test).all():
                    res[i, j, k, :] = fill
                    continue
                amin, amax = int(np.nanmin(test)) + 1, int(np.nanmax(test)) - 1
                res[i, j, k, :amin] = res[i, j, k, amin]
                res[i, j, k, amax:] = res[i, j, k, amax]
    return res


def remove_background_ds(ir1, ir2, ir3, ir4, ch_widths, rayleigh_scales, recal=None):
    ir1c, ir2c = ir1.copy(), ir2.copy()
    if recal is not None:
        ir1c, ir2c, ir3, ir4 = [recal[i] * arr for i, arr in enumerate([ir1c, ir2c, ir3, ir4])]
    ir1c -= ir4 * rayleigh_scales[0] / rayleigh_scales[3]
    ir2c -= ir4 * rayleigh_scales[1] / rayleigh_scales[3]
    return ir1c * ch_widths[0], ir2c * ch_widths[1]


def remove_background_ds_sep(ag, ir3, ir4, ch_widths, rayleigh_scales, agch, recal=None):
    chid = {"IR1": 0, "IR2": 1}
    ch = chid[agch]
    agc = ag.copy()
    if recal is not None:
        ids = [ch, 2, 3]
        ag, ir3, ir4 = [recal[ids[i]] * arr for i, arr in enumerate([ag, ir3, ir4])]
    agc -= ir4 * rayleigh_scales[ch] / rayleigh_scales[3]
    return agc * ch_widths[ch]


def remove_background_1ch(ag, bg, ch_width, rayleigh_scales, recal=None):
    print(ch_width, rayleigh_scales)
    agc, bgc = ag.copy(), bg.copy()
    if recal is not None:
        agc, bgc = [recal[i] * arr for i, arr in enumerate([agc, bgc])]
    agc -= bgc * rayleigh_scales[0] / rayleigh_scales[1]
    return agc * ch_width


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


def ndshift(data, offset, axis):
    shape = list(data.shape)
    size = shape[axis]
    res = data.copy()
    if offset == 0:
        return res
    pshape = shape.copy()
    if offset > 0:
        pshape[axis] = offset
        res = np.concatenate([np.take(data, indices=np.arange(offset, size), axis=axis), np.full(pshape, np.nan)],
                             axis=axis)
    else:
        pshape[axis] = - offset
        res = np.concatenate([np.full(pshape, np.nan), np.take(data, indices=np.arange(size + offset), axis=axis)],
                             axis=axis)
    return res


def ndshift_sym(data, offset, axis):
    shape = list(data.shape)
    size = shape[axis]
    res = data.copy()
    if offset == 0:
        return res
    elif offset > 0:
        idx = np.arange(2 * offset, size)
    else:
        idx = np.arange(size + 2 * offset)

    pshape = shape.copy()
    pshape[axis] = np.abs(offset)

    return np.concatenate([np.full(pshape, np.nan), np.take(data, indices=idx, axis=axis), np.full(pshape, np.nan)],
                          axis=axis)


def gen_neighborhood(data, ranges, force_sym=False):
    shiftf = ndshift_sym if force_sym else ndshift
    ndims = len(data.shape)
    assert len(ranges) == ndims, "Must have one interval in ranges for each data dimension!"
    idxs = [arr.flatten() for arr in np.meshgrid(*ranges, indexing='ij')]
    
    numn = len(idxs[0])
    for n in range(numn):
        if all([idxs[d][n] == 0 for d in range(ndims)]):
            numn -= 1
            for d in range(ndims):
                idxs[d] = np.delete(idxs[d], n)
            break

    res = np.empty(data.shape + (numn,))
    for n in range(numn):
        copy = data.copy()
        for d in range(ndims):
            copy = shiftf(copy, idxs[d][n], d)
        res[..., n] = copy

    return res


def denoise(data, hws, thr):
    assert len(data.shape) == len(hws), "Need one half-width per data dimension"
    ranges = [np.arange(-hw, hw + 1) for hw in hws]
    neighb = gen_neighborhood(data, ranges)
    median = np.nanmedian(neighb, axis=-1)
    pass_1 = np.where(np.abs(data - median) > thr * np.nanstd(neighb, axis=-1), median, data)
    neighb = gen_neighborhood(pass_1, ranges)
    return np.where(np.abs(data - median) > thr * np.nanstd(neighb, axis=-1), median, data)


def separate_scattered_stray(im, ref_rows, bot_row, scale_height, valid=None,
                             residual_treatment='smooth', sigma=5):
    if valid is None:
        valid = np.ones(im.shape[0])
    else:
        assert len(valid.shape) == 1 and valid.shape[0] == im.shape[0], \
            "'valid' must be a 1D ndarray, one value per image, or None"

    frows = np.arange(im.shape[1])
    top_rows = [int(0.5 * (ref_rows[j][0] + ref_rows[j][1] - 1)) for j in range(im.shape[2])]

    top = np.stack([np.median(im[:, ref_rows[j][0]:ref_rows[j][1], j], axis=1) for j in range(im.shape[2])], axis=1)
    im2 = im / top[:, np.newaxis, :] - 1
    fits = np.zeros((2, im.shape[0], im.shape[2]))
    rayleigh_fit, scat_fit = [np.full_like(im, np.nan) for _ in range(2)]
    for j in range(im.shape[2]):
        top_row = top_rows[j]
        rows = np.arange(bot_row, top_row)

        def exp_lin(x, a, b, Lp=scale_height, top_row=top_row, bot_row=bot_row):
            return a * (top_row - x) + b * np.exp(-(x - bot_row) / Lp)

        for i in range(im.shape[0]):
            if not valid[i]:
                continue
            fits[:, i, j] = curve_fit(exp_lin, rows, im2[i, bot_row:top_row, j], p0=[0, im2[i, bot_row, j]])[0]
            rayleigh_fit[i, :, j] = top[i, j] * fits[1, i, j] * np.exp(-(frows - bot_row) / scale_height)
            scat_fit[i, :, j] = top[i, j] * (1 + fits[0, i, j] * (top_row - frows))

    if residual_treatment == 'discard':
        return scat_fit, rayleigh_fit

    residual = im - scat_fit - rayleigh_fit
    # scat_fit_og = scat_fit.copy()
    # rayleigh_fit_og = rayleigh_fit.copy()
    for j in range(im.shape[2]):
        residual[:, top_rows[j] + 1:, j] = 0

    if residual_treatment == 'smooth':
        residual = gaussian_filter1d(residual, axis=1, sigma=sigma, mode='nearest')
    elif residual_treatment != 'full':
        raise ValueError(f"residual_treatment can take values 'discard', 'smooth' or 'full', but got {residual}.")
    scat_share = scat_fit / (scat_fit + rayleigh_fit)
    scat_fit += residual * scat_share
    rayleigh_fit += residual * (1 - scat_share)
    return scat_fit, rayleigh_fit  # , scat_fit_og, rayleigh_fit_og


def find_images(ref_times, aux_times, thr=3.1, valid_ref=None, valid_aux=None):
    if valid_ref is None:
        valid = np.ones_like(ref_times)
    if valid_aux is None:
        valid_aux = [np.ones_like(arr) for arr in aux_times]
    valid = valid_ref.copy()
    res = []
    for i, times in enumerate(aux_times):
        idx = np.searchsorted(times, ref_times, side='left')
        idx = np.where(idx == 0, 1, idx)
        idx = np.where(idx == len(times), len(times) - 1, idx)
        idx = np.where(np.abs(times[idx - 1] - ref_times) < np.abs(times[idx] - ref_times), idx - 1, idx)
        idx = np.where(np.logical_and(np.abs(times[idx] - ref_times) < thr, valid_aux[i][idx]), idx, -1)
        valid = np.logical_and(valid, idx > -1)
        res.append(idx)

    for i in range(len(aux_times)):
        res[i] = np.where(valid, res[i], -1)

    return valid, *res


def deghost(imgs, ghost_strength, offset_pix=None, offset_angle=None, pix_pitch=None, deg_map=None, pad=None):
    if offset_pix is not None:
        offset = offset_pix
    elif (offset_angle is not None) and (pix_pitch is not None):
        offset = offset_angle / pix_pitch
    elif (offset_angle is not None) and (deg_map is not None):
        offset = offset_angle / np.mean(np.diff(deg_map[:, : , 1], axis=0))
    else:
        raise ValueError("Specify offset_pix OR offset_angle and pix_pitch OR offset_angle and deg map.")

    offset_int = int(np.floor(np.abs(offset)))
    offset_frac = np.abs(offset) - offset_int
    kernel = np.zeros(offset_int + 2)
    kernel[0] = 1
    kernel[-2:] = ghost_strength * np.array([1 - offset_frac, offset_frac])
    if offset < 0:
        kernel = np.flip(kernel)

    if pad is not None:
        pad_shape = (imgs.shape[0], len(kernel) - 1, imgs.shape[2])
        if type(pad) is np.ndarray:
            assert pad.shape == pad_shape
            extension = pad
        elif type(pad) is float:
            extension = np.full(pad_shape, pad)
        elif pad == 'nearest':
            nearest_idx = -1 if offset > 0 else 0
            extension = np.broadcast_to(imgs[:, nearest_idx, :][:, np.newaxis, :], pad_shape)
        stack = [extension, imgs.copy()] if offset < 0 else [imgs.copy(), extension]
        data = np.concatenate(stack, axis=1)
    else:
        data = imgs

    res = np.zeros_like(imgs)

    for i in range(imgs.shape[0]):
        for k in range(imgs.shape[2]):
            res[i, :, k] = deconvolve(data[i, :, k], kernel)[0]

    return res


def pvgrad(data, denoise_iter=0, roll=0):
    pad = np.zeros((1, data.shape[1]))
    diff = np.diff(data, axis=0, append=pad)
    diff = np.maximum(0, diff)
    if denoise_iter > 0:
        diff[binary_erosion(diff > 0, iterations=denoise_iter) == 0] = 0.0
    if roll > 0:
        diff = np.minimum(diff, np.roll(diff, roll, axis=1))
    # denoised = np.minimum(diff, np.roll(diff, roll, axis=1)) if roll > 0 else diff
    return diff


def aurora_proxy_img(image, args):
    # Detect high signal above high thresold
    high = np.logical_and(image[args["TPh"]] > args["vrange"][1] - 1e3, image[args["TPh"]] < args["vrange"][1])
    bright_high = np.where(high, np.minimum(image[args["IR1"]], image[args["IR2"]]), 0.0) > args["high_thr"]
    bright_high[binary_erosion(bright_high, iterations=args["denoise_iter"] * 2) == 0] = 0

    valid = np.logical_and(image[args["TPh"]] > args["vrange"][0], image[args["TPh"]] < args["vrange"][1])
    valid_rows = np.sum(valid, axis=1).astype(bool)
    grads = []
    for chn in ["IR1", "IR2"]:
        rmean = sum([image[args[chn]][np.roll(valid_rows, r), :] for r in [-2, -1, 0, 1, 2]])
        grads.append(np.where(valid[valid_rows, :],
                              pvgrad(rmean, denoise_iter=args["denoise_iter"], roll=args["roll"]),
                              0.0))
    res = []
    for i in range(2):
        full = np.zeros_like(valid)
        full[valid_rows, :] = np.maximum(grads[i], bright_high[valid_rows, :])
        res.append(full)
    return np.stack(res, axis=0)
    # return np.sum((np.minimum(grads[0], grads[1]) > 0)[:, args["col_range"][0]:args["col_range"][1]]) + bright_high


def aurora_proxy(data, vrange=[95e3, 104e3], denoise_iter=1, roll=1, ir1v="IR1", ir2v="IR2", tpv="TPh", nproc=1,
                 col_range=(0, 44), full=False, aurora_thr=6e13):
    common_args = {"vrange": [x * 1e3 for x in vrange], "denoise_iter": denoise_iter, "roll": roll, "full": full,
                   "IR1": ir1v, "IR2": ir2v, "TPh": tpv, "col_range": col_range, "high_thr": aurora_thr}
    return multiprocess(aurora_proxy_img, data, [ir1v, ir2v, tpv], nproc, common_args, stack=True)


def jpg_fail_proxy(data, comp_hw=5):
    sums = np.maximum(data[:, 1:, :] + data[:, :-1, :], 0.0)
    series = np.where(sums > 0, np.maximum(data[:, 1:, :] - data[:, :-1, :], 0.0), 0.0)
    series = np.sqrt(np.sum(series ** 2, axis=(1, 2)))
    res = np.array([2 * series[i] / (np.median(series[i - comp_hw:i]) + np.median(series[i + 1:i + comp_hw + 1]))
                   for i in range(len(series))])
    return res


def extract_bit(calib_errors, bit, flip=False):
    mod = np.power(2, bit - 1)
    res = calib_errors % (2 * mod) >= mod
    if flip:
        return np.flip(res, axis=1)
    else:
        return res


def fill_invalid(data, invalid, ranges=[(0,), (-2, -1, 0, 1, 2), (-2, -1, 0, 1, 2)], outlier_filter=None):
    neighbours = gen_neighborhood(data, ranges, force_sym=True)
    medians = np.nanmedian(neighbours, axis=-1)
    medians[:, 0, 0] = medians[:, 0, 1]
    medians[:, 0, -1] = medians[:, 0, -2]
    medians[:, -1, 0] = medians[:, -1, 1]
    medians[:, -1, -1] = medians[:, -1, -2]

    if outlier_filter is not None:
        outliers = np.nanstd(neighbours, axis=-1) * outlier_filter < np.abs(data - medians)
        invalid = np.logical_or(invalid, outliers)

    return np.where(invalid, medians, data), invalid


def layer_sim_factor(top, thickness, h):
    Re = 6371
    rh2 = (Re + h) ** 2
    res = np.sqrt((Re + top + thickness) ** 2 - rh2) - np.sqrt((Re + top) ** 2 - rh2)
    return np.minimum(res / np.sqrt(thickness * (2 * Re + 2 * top + thickness)), 1.0)


def subtract_top_median(data, tph, ref_alt_range, conf={"strategy": "constant"}):
    medians = np.nanmedian(np.where(np.logical_and(tph > ref_alt_range[0], tph < ref_alt_range[1]), data, np.nan),
                           axis=(1, 2))
    if conf["strategy"] == "constant":
        return data - medians[:, np.newaxis, np.newaxis], medians
    if conf["strategy"] == "layer_sim":
        return data - layer_sim_factor(ref_alt_range[1], conf["thickness"], tph) *\
            medians[:, np.newaxis, np.newaxis], medians


def hot_pix_trans_masks(hot_pix_file, channel, threshold, img_times):
    hpdata = np.load(hot_pix_file)

    # Determine which mask to use for which image
    dts = np.array([DT2seconds(DT.datetime.fromisoformat(t).replace(tzinfo=DT.timezone.utc))
                    for t in hpdata[f"{channel}_datetimes"]])
    idxs = np.array([bisect(dts, t) for t in img_times])
    if (idxs == 0).any() or (idxs == len(dts)).any():
        raise ValueError("Some images are outside the time range covered by the hot pixel map!")
    idxs = idxs - 1

    # Generate masks themselves
    hmaps = hpdata[channel].astype(float)
    masks = np.abs(hmaps[1:, :, :] - hmaps[:-1, :, :]) > threshold

    return masks, idxs




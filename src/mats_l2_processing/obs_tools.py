import numpy as np
import datetime as DT
from bisect import bisect
import logging
import time

from mats_l2_processing.util import DT2seconds


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


def calc_medians(data, hws, substep=500):
    ranges = [np.arange(-hw, hw + 1) for hw in hws]
    res = []
    size = data.shape[0]
    for start in range(0, size, substep):
        res.append(np.nanmedian(gen_neighborhood(data[start:start + substep, ...], ranges, force_sym=True),
                                axis=-1))
        print(f"{(start + substep)/ size * 100} %")
    res = np.concatenate(res, axis=0)

    # res = np.concatenate([np.nanmedian(gen_neighborhood(data[start:start + substep, ...], ranges, force_sym=True),
    #                                    axis=-1) for start in range(0, data.shape[0], substep)], axis=0)

    # The algorithm above leaves image corner pixels undefined. The fix is below.
    if hws[0] == 0:
        res[:, 0, 0] = res[:, 0, 1]
        res[:, 0, -1] = res[:, 0, -2]
        res[:, -1, 0] = res[:, -1, 1]
        res[:, -1, -1] = res[:, -1, -2]
    else:
        for j in [0, -1]:
            res[j, 0, 0] = res[j, 0, 1]
            res[j, 0, -1] = res[j, 0, -2]
            res[j, -1, 0] = res[j, -1, 1]
            res[j, -1, -1] = res[j, -1, -2]
    return res


def dual_pass_medians(data, hws, abs_threshold, substep=500, rel_threshold=2, axis=1):
    pass_1 = data - calc_medians(data, hws, substep=substep)
    pass_1_m = np.median(pass_1, axis=0)
    mask = np.abs(np.maximum(ndshift(pass_1_m, 1, axis=axis), ndshift(pass_1_m, -1, axis=axis))) *\
        rel_threshold < pass_1_m
    mask = np.logical_and(mask, pass_1_m > abs_threshold)
    return data - calc_medians(np.where(mask[np.newaxis, ...], np.nan, data), hws, substep=substep), mask


def abs_grad(data, min_factor=0.01):
    res = np.zeros_like(data)
    res[:-1, :] = np.abs(data[1:, :] - data[:-1, :])
    res[-1, :] = np.abs(data[-1, :] - data[-2, :])
    res[:, :-1] += np.abs(data[:, 1:] - data[:, :-1])
    res[:, -1] += np.abs(data[:, -1] - data[:, -2])
    return np.maximum(res, min_factor * np.abs(data))


def calc_ROAD(data, hws, K, substep=500):
    shape = data.shape
    k = shape[0] * K
    ranges = [np.arange(-hw, hw + 1) for hw in hws]
    nb = gen_neighborhood(data, ranges, force_sym=False)
    # ROAD_ref = np.nanmean(np.nanstd(np.diff(np.sort(nb, axis=-1), axis=-1), axis=-1),
    #                      axis=0)
    res = []
    nb_median = np.median(nb, axis=-1)
    for arr in [data, nb_median]:
        anomaly = np.abs(arr[..., np.newaxis] - nb).transpose(0, 3, 1, 2).reshape(-1, shape[1], shape[2])
        res.append(np.nanmean(np.partition(anomaly, k, axis=0)[:k, ...], axis=0))
    return *res, np.median(np.median(data[..., np.newaxis] - nb, axis=-1), axis=0)


def calc_T_factor(t, T0, E0):
    T = t + 273
    k_B_eV = 8.617333262e-5  # Bolzmann's constant in eV
    return (T / T0) ** 1.5 * np.exp(- E0 / k_B_eV * (1 / T - 1 / T0))


def weighted_nanmean_outliers(data, weights, axis=-1, k=None, res_threshold=None):
    assert data.shape == weights.shape, f"{data.shape}, {weights.shape}"
    weighted = data * weights
    mask = np.isnan(weighted)
    res = np.nansum(weighted, axis=axis) / np.nansum(np.where(mask, np.nan, weights), axis=axis)
    if k is None and res_threshold is None:
        return res

    if k is not None:
        std = np.nanstd(data, axis=axis)
        mask = np.logical_or(mask, np.abs(data - np.expand_dims(res, axis=axis)) > std * k)
    if res_threshold is not None:
        mask = np.logical_or(mask, res > res_threshold)
    res = np.nansum(np.where(mask, np.nan, weighted), axis=axis) /\
        np.nansum(np.where(mask, np.nan, weights), axis=axis)
    return res


def ROLD(diff):
    return 1 + np.where(diff == 0.0, -1, np.maximum(np.log2(np.abs(diff)), -5) / 5)


def gen_hpix_map(t0, img, times, T, hws=(0, 1, 1), K=4, T0=235, E0=0.63, T_model=calc_T_factor, hpix_threshold=7,
                 num_darkest=2000, col_margins=(1, 1), row_margins=(1, 3), margin_weight_fact=2):
    shape = img.shape
    assert len(shape) == 3
    assert len(times.shape) == 1
    assert len(T.shape) == 1
    assert times.shape[0] == shape[0]
    assert T.shape[0] == shape[0]
    assert not np.isnan(T).any()
    assert not np.isnan(times).any()

    k = shape[0] * K
    ranges = [np.arange(-hw, hw + 1) for hw in hws]

    nb = gen_neighborhood(img, ranges, force_sym=False)
    darkest = np.argpartition(np.percentile(img, 90, axis=(1, 2)), num_darkest)
    darkest_nb_median = np.nanmedian(nb[darkest, ...], axis=-1)
    ROADs = []
    start_ctime = time.time()
    for arr in [img[darkest, ...], darkest_nb_median]:
        anomaly = np.abs(arr[..., np.newaxis] - nb[darkest, ...]).transpose(0, 3, 1, 2).reshape(-1, shape[1], shape[2])
        ROADs.append(np.nanmean(np.partition(anomaly, k, axis=0)[:k, ...], axis=0))
    ROAD_norm = ROADs[0] - ROADs[1]
    del ROADs
    ROAD_mask = np.full_like(ROAD_norm, False, dtype=bool)
    ROAD_mask[:row_margins[0], :] = True
    ROAD_mask[-row_margins[1]:, :] = True
    ROAD_mask[:, :col_margins[0]] = True
    ROAD_mask[:, -col_margins[1]:] = True
    ROAD_sigma = np.std(ROAD_norm[~ROAD_mask])
    ROAD_weight = np.where(ROAD_mask, ROAD_sigma * margin_weight_fact,
                           np.exp(-0.5 * ROAD_norm ** 2 / ROAD_sigma ** 2))
    ROAD_ctime = time.time()

    weights_nb = gen_neighborhood(ROAD_weight[np.newaxis, ...], ranges, force_sym=False)
    weights_ctime = time.time()

    clean_med, med_mask = weighted_median(nb, weights_nb, -1)
    clean_med_ctime = time.time()
    del nb, weights_nb
    anomaly = img - clean_med
    if E0 is not None:
        T_factors = T_model(T, T0, E0)
        anomaly /= T_factors[:, np.newaxis, np.newaxis]
        mean_T_factor = np.nanmean(T_factors)
    else:
        mean_T_factor = 1.0
    temporal_weights = 1 / np.mean(img, axis=1, keepdims=True)
    hpix_map, _ = weighted_median(anomaly, temporal_weights, 0)
    time_med_ctime = time.time()

    hpix_map = np.where(hpix_map > hpix_threshold, hpix_map, 0.0)
    img_before = times - t0 < 0
    return {"map": hpix_map, "ROAD_weight": ROAD_weight, "mean_T_factor": mean_T_factor,
            "anomaly_before": anomaly[img_before, ...], "times_before": times[img_before],
            "anomaly_after": anomaly[~img_before, ...], "times_after": times[~img_before],
            "ctime": {"ROAD": ROAD_ctime - start_ctime, "weights": weights_ctime - ROAD_ctime,
                      "clean_med": clean_med_ctime - weights_ctime, "time_med": time_med_ctime - clean_med_ctime}}


def hpix_trans_times(hpix0, hpix1, anomaly, times, change_threshold=7):
    diff = hpix1 - hpix0
    undefined = np.abs(diff) < change_threshold
    squares = np.sum((anomaly - hpix1[np.newaxis, ...]) ** 2, axis=0)
    squares = np.where(undefined, 0.0, 0.5 * squares / diff)
    misfit = squares[np.newaxis, ...] - np.cumsum(anomaly + 0.5 * (hpix0 + hpix1)[np.newaxis, ...], axis=0)
    return np.where(undefined, np.nan, times[np.argmin(misfit, axis=0)])


def gen_ref_hpix_map_ROAD(img, T, hws=(0, 1, 1), K=4, t_sigma=3, T0=253, E0=0.63, T_model=calc_T_factor):
    shape = img.shape
    print(f"processing {shape}")
    k = shape[0] * K
    ranges = [np.arange(-hw, hw + 1) for hw in hws]

    darkest = np.argpartition(np.percentile(img, 90, axis=(1, 2)), 2000)
    nb = gen_neighborhood(img, ranges, force_sym=False)
    ROAD_runs = []
    darkest_nb_median = np.nanmedian(nb[darkest, ...], axis=-1)
    for arr in [img[darkest, ...], darkest_nb_median]:
        anomaly = np.abs(arr[..., np.newaxis] - nb[darkest, ...]).transpose(0, 3, 1, 2).reshape(-1, shape[1], shape[2])
        ROAD_runs.append(np.nanmean(np.partition(anomaly, k, axis=0)[:k, ...], axis=0))
        # anomaly = np.nanmean(np.partition(np.abs(arr[..., np.newaxis] - nb), K, axis=-1)[..., :K], axis=-1)
        # ROAD_runs.append(np.nanmedian(anomaly, axis=0))

    del anomaly, darkest_nb_median
    margins = 2, 5
    # ransac = linear_model.RANSACRegressor()
    # ransac.fit(ROAD_runs[0].flatten()[:, np.newaxis], ROAD_runs[1].flatten())
    # outlier_mask = np.logical_not(ransac.inlier_mask_).reshape(ROAD_runs[0].shape)
    # slope = ransac.estimator_.coef_
    # intercept = ransac.estimator_.intercept_
    # ROAD_norm = ROAD_runs[0] - intercept - ROAD_runs[1] * slope
    ROAD_norm = ROAD_runs[0] - ROAD_runs[1]
    ROAD_norm_clip = ROAD_norm[margins[1]:-margins[1], margins[0]:-margins[0]]
    ROAD_weight = np.where(np.isnan(ROAD_norm), 0.0, np.exp(-0.5 * ROAD_norm ** 2 / np.std(ROAD_norm_clip) ** 2))
    # Weights for every entry in nb. Missing values (nan) appear in same places as nb. These values must be set to zero
    # in weights, and any finite number in nb.
    weights_nb = gen_neighborhood(ROAD_weight[np.newaxis, ...], ranges, force_sym=False)
    # img_mask = np.isnan(weights_nb)[0, :, :]
    # weights_nb = np.nan_to_num(weights_nb, 0.0)

    outlier_mask = ROAD_norm > 3.5
    # nb_clean = gen_neighborhood(np.where(outlier_mask[np.newaxis, :, :], np.nan, img), ranges, force_sym=False)
    # clean_nb = np.where(mask, 0.0, nb)
    # clean_weights = np.broadcast_to(ROAD_weight[np.newaxis, :, :, np.newaxis], nb.shape)
    clean_med, med_mask = weighted_median(nb, weights_nb, -1)
    # clean_med = np.percentile(np.nan_to_num(nb, 0.0), 50, method="inverted_cdf", axis=-1,
    #                           weights=np.broadcast_to(weights_nb, nb.shape))
    T_factors = T_model(T, T0, E0)
    anomaly = (img - clean_med) / T_factors[:, np.newaxis, np.newaxis]
    # hpix_map = np.median(anomaly, axis=0)  * T_model(T.mean(), T0, E0)
    # hpix_map = np.percentile(anomaly, 10, axis=0) * T_model(T.mean(), T0, E0)
    temporal_weights = 1 / np.mean(img, axis=1, keepdims=True)  # Downweight obs. by column mean
    hpix_map, _ = weighted_median(anomaly, temporal_weights, 0)
    deviation = np.where(med_mask, 0, anomaly - hpix_map[np.newaxis, ...])
    outliers = deviation > t_sigma * np.std(deviation, axis=0, keepdims=True)  # Find outlying obs.
    temporal_weights = np.where(outliers, 0.0, temporal_weights)
    hpix_map, _ = weighted_median(anomaly, temporal_weights, 0)   # REcalculate with outliers discarded

    # hpix_map = weighted_nanmean_outliers(anomaly, np.broadcast_to(1 / np.mean(img, axis=1, keepdims=True), img.shape),
    #                                     axis=0, k=3)
    # return np.where(ROAD_mask, hpix_map, 0), hpix_map, ROAD_runs[0], ROAD_runs[1]
    return {"ROAD_raw": ROAD_runs[0], "ROAD_bgr": ROAD_runs[1], "ROAD_norm": ROAD_norm, "ROAD_weight": ROAD_weight,
            "outliers": outlier_mask, "hpix_map": hpix_map}


def weighted_median(nb, nb_weights, axis, mask_if_all_nans=False):
    if mask_if_all_nans:
        mask = np.sum(~np.isnan(nb_weights), axis=axis).astype(bool)
    else:
        mask = ~np.sum(np.isnan(nb_weights), axis=axis).astype(bool)

    return np.percentile(np.nan_to_num(nb, 0.0), 50, method="inverted_cdf", axis=axis,
                         weights=np.broadcast_to(np.nan_to_num(nb_weights, 0.0), nb.shape)), mask


def calc_ROAD_ref(data, hws):
    ranges = [np.arange(-hw, hw + 1) for hw in hws]
    grad = np.zeros_like(data)
    grad[:, :, :-1] = np.abs(data[:, :, 1:] - data[:, :, :-1])
    grad[:, :, -1] = np.abs(data[:, :, -1] - data[:, :, -2])
    return np.mean(np.nanmedian(gen_neighborhood(grad, ranges, force_sym=True), axis=-1), axis=0)


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

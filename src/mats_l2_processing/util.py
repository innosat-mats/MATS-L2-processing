import datetime as DT
from multiprocessing import Pool
import numpy as np
from itertools import chain, repeat


def get_filter(channel):
    filters = {"IR1": 1, "IR2": 4, "IR3": 3, "IR4": 2, "UV1": 5, "UV2": 6, "all": -1}
    try:
        filt = filters[channel]
    except Exception:
        raise ValueError(f"Invalid channel: {channel}!")

    if filt < 0:
        return {}
    else:
        return {'CCDSEL': [filt, filt]}


def DT2seconds(dts):
    return (dts - DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc)) / DT.timedelta(0, 1)


def seconds2DT(s):
    return DT.timedelta(0, s) + DT.datetime(2000, 1, 1, 0, 0, tzinfo=DT.timezone.utc)


def get_image(data, idx, var, size=None):
    res = {"num_image": idx}
    if size is not None:
        res["num_images"] = size
    for v in var:
        if hasattr(data[v], 'shape') and len(data[v].shape) > 1:
            res[v] = data[v][idx, ...]
        else:
            res[v] = data[v][idx]
    return res


def multiprocess(func, dataset, image_args, nproc, common_args, unzip=False, stack=False):
    assert nproc >= 1, "Invalid number of processes specified!"
    size = dataset["size"]
    images = [get_image(dataset, i, image_args, size=size) for i in range(size)]
    if nproc == 1:  # Serial processing (implemented separately to simplify debugging)
        res = []
        # if len(common_args) > 0:
        for image in images:
            res.append(func(image, common_args))
        # else:
        #     for image in images:
        #         res.append(func(image))
    else:  # Actual multiprocessing
        # if len(common_args) > 0:
        with Pool(processes=nproc) as pool:
            res = pool.starmap(func, zip(images, repeat(common_args)))
        # else:
        #     with Pool(processes=nproc) as pool:
        #         res = pool.starmap(func, images)

    if unzip:
        unzipped = []
        for i in range(len(res[0])):
            unzipped.append(np.array(list(chain.from_iterable([image_res[i] for image_res in res]))))
        return unzipped
    elif stack:
        return np.stack(res, axis=0)
    else:
        return res


def running_mean(data, hw):
    res = np.zeros_like(data)
    width = int(2 * hw + 1)
    res[hw:-hw] = np.convolve(data, np.ones(width) / width, mode='valid')
    res[0:hw] = res[hw]
    res[-hw:] = res[-hw - 1]
    return res


def center_grid(grid):
    return (grid[:-1] + grid[1:]) / 2


def print_times(times, titles):
    assert len(titles) + 1 == len(times)
    res = ""
    for i, title in enumerate(titles):
        res += f"{title}: {times[i + 1] - times[i]:.3e} s, "
    res += f" {times[-1] - times[0]:.3e} s total."
    print(res)


def geoid_radius(latitude):
    '''
    Function from GEOS5 class.
    GEOID_RADIUS calculates the radius of the geoid at the given latitude
    [Re] = geoid_radius(latitude) calculates the radius of the geoid (km)
    at the given latitude (degrees).
    ----------------------------------------------------------------
            Craig Haley 11-06-04
    ---------------------------------------------------------------
    '''
    # DEGREE = np.pi / 180.0
    EQRAD = 6378.14 * 1000
    FLAT = 1.0 / 298.257
    Rmax = EQRAD
    Rmin = Rmax * (1.0 - FLAT)
    Re = np.sqrt(1. / (np.cos(latitude)**2 / Rmax**2 + np.sin(latitude)**2 / Rmin**2))
    return Re


def sph2cart(r, phi, theta):
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def cart2sph(pos):
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    longitude = np.arctan2(y, x)
    latitude = np.arcsin(z / radius)

    return radius, longitude, latitude


def make_grid_proto(proto, offset=0.0, scaling=1.0):
    if (type(proto) is int) or (type(proto) is float):
        return int(proto)
    assert type(proto) is list, "Malformed grid specification!"
    if type(proto[0]) is tuple:
        for i, interval in enumerate(proto):
            assert type(interval) is tuple, "Malformed grid specification!"
            assert len(interval) == 3, "Malformed grid specification!"
            if i == 0:
                res = np.arange(*interval)
            else:
                res = np.concatenate((res, np.arange(*interval)))
    else:
        res = np.array(proto)
    return res * scaling + offset


def grid_from_proto(proto, lims):
    if (type(proto) is int) or (type(proto) is float):
        grid_len = int(proto)
        assert grid_len > 0, "Malformed grid_spec parameter!"
        return np.linspace(lims[0], lims[1], grid_len)
    elif isinstance(proto, np.ndarray):
        assert len(proto.shape) == 1
        return proto[np.logical_and(proto > lims[0], proto < lims[1])]
    else:
        raise ValueError(f"Malformed grid_spec parameter: {type(proto)}")


def array_shift(data, shift):
    shift_lower = int(np.floor(shift))
    theta = shift - shift_lower
    res = np.roll(data, shift_lower) * (1 - theta) + np.roll(data, shift_lower + 1) * theta
    if 2 * shift_lower + 1 > 0:
        res[:(shift_lower + 1), ...] = np.expand_dims(data[0, ...], axis=0)
    else:
        res[shift_lower:, ...] = np.expand_dims(data[-1, ...], axis=0)
    return res


def array_shifts(data, shifts, axis):
    assert len(shifts.shape) == len(shifts.shape) + 1
    assert data.shape[axis] == len(shifts)

    wdata = data.copy()
    wdata = np.swapaxes(wdata, 0, axis)
    it = np.nditer(shifts, flags=['multi_index'])
    for shift in it:
        wdata[:, *it.multi_index] = array_shift(wdata[:, *it.multi_index], shift)
    wdata = np.swapaxes(wdata, 0, axis)
    return wdata


def combine_dsets(dicts):
    if len(dicts) == 1:
        return dicts[0]
    var = list(dicts[0].keys())
    for i, d in enumerate(dicts[1:]):
        if list(d.keys()) != var:
            raise ValueError(f"Dataset {i + 1} have different variables than the first one! Abort")

#    res = {}
#    for v in var:
#        if type(dicts[0][v]) is :


def dict_contents(d):
    for key in d.keys():
        if type(d[key]) is np.ndarray:
            print(f"{key}: ndarray {d[key].shape}, {d[key].dtype}")
        elif type(d[key]) is np.ma.core.MaskedArray:
            print(f"{key}: masked ndarray {d[key].shape}, {d[key].dtype}")
        elif type(d[key]) is list:
            print(f"{key}: list {len(d[key])}")
        else:
            print(f"{key}: {type(d[key])}")


def filter_dict(dictionary, valid):
    d = dictionary.copy()
    vld = valid.astype(bool)
    for key in d.keys():
        if (type(d[key]) is np.ndarray) or (type(d[key]) is np.ma.core.MaskedArray):
            if not len(vld) == d[key].shape[0]:
                raise RuntimeError("Invalid shape of metadata entry!")
            d[key] = d[key][vld, ...]
    if 'size' in d.keys():
        d['size'] = vld.sum()
    return d


def ecef2wgs84(pos):
    # x, y and z are scalars or vectors in meters
    # x = np.array([x]).reshape(np.array([x]).shape[-1], 1)
    # y = np.array([y]).reshape(np.array([y]).shape[-1], 1)
    # z = np.array([z]).reshape(np.array([z]).shape[-1], 1)

    a = 6378137
    # a_sq = a ** 2
    # e = 8.181919084261345e-2
    e_sq = 6.69437999014e-3

    f = 1 / 298.257223563
    b = a * (1 - f)

    x, y, z = [pos[:, k] for k in range(3)]

    # calculations:
    r = np.sqrt(x**2 + y**2)
    ep_sq = (a**2 - b**2) / b**2
    ee = (a**2 - b**2)
    f = (54 * b**2) * (z ** 2)
    g = r**2 + (1 - e_sq) * (z ** 2) - e_sq * ee * 2
    c = (e_sq**2) * f * r**2 / (g**3)
    s = (1 + c + np.sqrt(c**2 + 2 * c)) ** (1 / 3.)
    p = f / (3. * (g ** 2) * (s + (1. / s) + 1)**2)
    q = np.sqrt(1 + 2 * p * e_sq ** 2)
    r_0 = -(p * e_sq * r) / (1 + q) +\
        np.sqrt(0.5 * (a ** 2) * (1 + (1. / q)) - p * (z ** 2) * (1 - e_sq) / (q * (1 + q)) - 0.5 * p * (r ** 2))
    u = np.sqrt((r - e_sq * r_0) ** 2 + z ** 2)
    v = np.sqrt((r - e_sq * r_0) ** 2 + (1 - e_sq) * z ** 2)
    z_0 = (b ** 2) * z / (a * v)
    h = u * (1 - b ** 2 / (a * v))
    phi = np.arctan((z + ep_sq * z_0) / r)
    lambd = np.arctan2(y, x)
    return phi, lambd, h

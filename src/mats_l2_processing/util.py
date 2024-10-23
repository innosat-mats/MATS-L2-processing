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
        if len(common_args) > 0:
            for image in images:
                res.append(func(image, common_args))
        else:
            for image in images:
                res.append(func(image))
    else:  # Actual multiprocessing
        if len(common_args) > 0:
            with Pool(processes=nproc) as pool:
                res = pool.starmap(func, zip(images, repeat(common_args)))
        else:
            with Pool(processes=nproc) as pool:
                res = pool.starmap(func, images)

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

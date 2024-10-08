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


def multiprocess(func, dataset, image_args, nproc, common_args, unzip=False):
    assert nproc >= 1, "Invalid number of processes specified!"
    size = dataset["size"]
    images = [get_image(dataset, i, image_args, size=size) for i in range(size)]
    if nproc == 1:  # Serial processing (implemented separately to simplify debugging)
        res = []
        for image in images:
            res.append(func(image, common_args))
    else:  # Actual multiprocessing
        with Pool(processes=nproc) as pool:
            res = pool.starmap(func, zip(images, repeat(common_args)))

    if unzip:
        unzipped = []
        for i in range(len(res[0])):
            unzipped.append(np.array(list(chain.from_iterable([image_res[i] for image_res in res]))))
        return unzipped
    else:
        return res


def running_mean(data, hw):
    res = np.zeros_like(data)
    width = int(2 * hw + 1)
    res[hw:-hw] = np.convolve(data, np.ones(width) / width, mode='valid')
    res[0:hw] = res[hw]
    res[-hw:] = res[-hw - 1]
    return res


def print_times(times, titles):
    assert len(titles) + 1 == len(times)
    res = ""
    for i, title in enumerate(titles):
        res += f"{title}: {times[i + 1] - times[i]:.3e} s, "
    res += f" {times[-1] - times[0]:.3e} s total."
    print(res)

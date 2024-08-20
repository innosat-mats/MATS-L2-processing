import numpy as np
from scipy.interpolate import RectBivariateSpline


def get_background(jb, mean_time, clim_data, ver_apr, t_apr=None):
    alt, lat, lon = jb["alt"] / 1000, jb["lat"], jb["lon"]
    month = mean_time.month

    # ver = 2e3 * 1e6 * stats.norm.pdf(alt, 88, 4.5) + 2e2 * 1e6 * np.exp(-(alt - 60) / 20)
    T, o2, ver = [np.zeros_like(alt) for c in range(3)]
    so2 = clim_data.o2.sel(month=month)
    sT = clim_data.T.sel(month=month)

    min_lat, max_lat = np.min(jb["lat"]), np.max(jb["lat"])
    center_lat_range = (2 * min_lat + max_lat) / 3, (min_lat + 2 * max_lat) / 3
    lat_averaged = ver_apr["VER"][0, ...].copy()

    lat_averaged[:, ver_apr["latitude"] < center_lat_range[0]] = np.nan
    lat_averaged[:, ver_apr["latitude"] > center_lat_range[1]] = np.nan
    lat_averaged = np.broadcast_to(np.nanmean(lat_averaged, axis=1)[:, np.newaxis], ver_apr["VER"][0, ...].shape)
    sver = RectBivariateSpline(ver_apr["altitude"], ver_apr["latitude"], lat_averaged)
    if t_apr is not None:
        sT = RectBivariateSpline(t_apr["altitude"], t_apr["latitude"], t_apr["T"][0, ...])
    # sT = RectBivariateSpline(t_apr["z"], t_apr["lat"], t_apr["T"][0, ...])

    for i in range(T.shape[1]):
        for j in range(T.shape[2]):
            # T[:, i, j] = sT(alt[:, i, j], lat[i, j])[:, 0]
            T[:, i, j] = sT.interp(lat=lat[i, j], z=alt[:, i, j])
            o2[:, i, j] = so2.interp(lat=lat[i, j], z=alt[:, i, j])
            ver[:, i, j] = sver(alt[:, i, j], lat[i, j])[:, 0]
    o2 *= 1e-6

    # Experimental sideload of VER apriori from result:
    # with nc.Dataset('/home/lk/tests/T_tomo/stest11/first/t4_L2.nc', 'r') as nf:
    #    vert = nf["VER"][0, :, :, :].data
    # assert vert.shape == ver.shape, f"VER:{ver.shape}, T:{T.shape}"
    # ver = vert.copy()

    assert not np.isnan(T).any(), "T a priori has NaN's!"
    assert not np.isnan(o2).any(), "o2 a priori has NaN's!"
    assert not np.isnan(ver).any(), "ver a priori has NaN's!"
    return o2, [ver, T]

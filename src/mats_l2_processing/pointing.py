import numpy as np
from mats_l2_processing.io import read_ncdf
from mats_l2_processing.util import ecef2wgs84, seconds2DT, multiprocess

from scipy.interpolate import RegularGridInterpolator, CubicSpline, bisplev
from scipy.optimize import minimize_scalar
from skyfield.api import load, wgs84
from scipy.spatial.transform import Rotation as R
from skyfield.units import Distance
from skyfield.positionlib import ICRF
from skyfield.framelib import itrs


class Pointing():
    def __init__(self, metadata, conf, const):
        verify_channels(conf, metadata)
        self.channels = conf.CHANNELS

        if conf.DISTORTION_CORRECTION:
            dist_data = read_ncdf(conf.DISTORTION_DATA, None, get_units=False)
        else:
            dist_data = None

        self.deg_map = {}
        num_meta = len(self.channels) if conf.SEP_CHN_LOS else 1
        for chid in range(num_meta):
            self.deg_map[self.channels[chid]] = get_deg_map({v: metadata[chid][v][0] if len(metadata[chid][v].shape) > 0
                                                            else metadata[v] for v in const.POINTING_DATA},
                                                            dist_data=dist_data)

    def pix_deg(self, image, xpix, ypix):
        return self.deg_map[image["channel"]][:, xpix, ypix]

    def get_deg_map(self, image):
        return self.deg_map[image["channel"]]

    def chn_map(self, chn=None):
        if chn is None:
            if len(self.channels) > 1:
                raise ValueError("Handling multi-channel data, hence channel must be specified to get pointing data!")
            chn = self.channels[0]
        return self.deg_map[chn]


def sparse_tp_data(image, deg_map, nx=5, ny=10):
    if nx > 0:
        xpixels = np.append(np.arange(0, image['NCOL'], nx), image['NCOL'])
    else:
        xpixels = np.array([int(np.floor(image['NCOL'] / 2))])
    ypixels = np.append(np.arange(0, image['NROW'] - 1, ny), image["NROW"] - 1)
    t = load.timescale().from_datetime(image['time'])
    ecipos = image['afsGnssStateJ2000'][0:3]
    quat = R.from_quat(np.roll(image['afsAttitudeState'], -1))
    qprime = R.from_quat(image['qprime'])

    xxpixels, yypixels = np.meshgrid(xpixels, ypixels, indexing="ij")
    xxdeg, yydeg = [deg_map[c, xxpixels, yypixels] for c in range(2)]
    TPheights = np.empty_like(xxpixels)
    TPpos = np.empty(tuple(list(xxpixels.shape) + [3]))
    with np.nditer([xxdeg, yydeg], flags=["multi_index"]) as it:
        eci2ecef = R.from_matrix(itrs.rotation_at(t))
        for xx, yy in it:
            los = R.from_euler('XYZ', [0, yy, xx], degrees=True).apply([1, 0, 0])
            ecivec = quat.apply(qprime.apply(los))
            tangent = findtangent(t, ecipos, ecivec)
            TPheights[*it.multi_index] = tangent.fun
            TPpos[*it.multi_index, :] = eci2ecef.apply(ecipos + tangent.x * ecivec)

    return xpixels, ypixels, TPheights, TPpos


def center_tp_pos_ecef_image(image, cargs):
    # cargs is dict with deg_map, central col. number (c_col), and ref. altitude (ref_alt)
    ypixels = np.append(np.arange(0, image['NROW'] - 1, cargs["ny"]), image["NROW"] - 1)
    t = load.timescale().from_datetime(image['time'])
    ecipos = image['afsGnssStateJ2000'][0:3]
    quat = R.from_quat(np.roll(image['afsAttitudeState'], -1))
    qprime = R.from_quat(image['qprime'])
    eci_to_ecef = R.from_matrix(itrs.rotation_at(t))
    xdeg, ydeg = [cargs["deg_map"][c, cargs["c_col"], ypixels] for c in range(2)]

    TPheights = np.empty_like(xdeg)
    TPpos = np.empty(xdeg.shape + (3,))
    for i in range(len(xdeg)):
        los = R.from_euler('XYZ', [0, ydeg[i], xdeg[i]], degrees=True).apply([1, 0, 0])
        ecivec = quat.apply(qprime.apply(los))
        tangent = findtangent(t, ecipos, ecivec)
        TPheights[i] = tangent.fun
        TPpos[i, :] = ecipos + tangent.x * ecivec

    idx = np.searchsorted(TPheights, cargs["ref_alt"]) - 1
    theta = (cargs["ref_alt"] - TPheights[idx]) / (TPheights[idx + 1] - TPheights[idx])
    res = eci_to_ecef.apply(TPpos[idx, :] * theta + TPpos[idx + 1, :] * (1 - theta))
    return res


def calc_track_ecef(metadata, deg_map, nproc, image_args, c_col=23, ny=10, ref_alt=8e4):
    common_args = {"deg_map": deg_map, "c_col": c_col, "ny": ny, "ref_alt": ref_alt}
    return multiprocess(center_tp_pos_ecef_image, metadata, image_args, nproc, common_args, stack=True)


def faster_heights(image, pointing, ny=5, cols=None, rows=None, full_coords=False, deg_map=None):
    fullxgrid = np.arange(image['NCOL'] + 1) if cols is None else cols
    fullygrid = np.arange(image['NROW']) if rows is None else rows
    if cols is not None and len(cols) > 1:
        nx = np.minimum(5, int(np.floor((image['NCOL'] + 1) / 4)))
    else:
        nx = int(np.floor(image['NCOL'] / 2))

    if pointing is None:
        if deg_map is None:
            raise ValueError("Neither pointing object or explicit deg_map was provided!")
        dmap = deg_map
    else:
        dmap = pointing.get_deg_map(image)

    xpixels, ypixels, TPheights, TPpos = sparse_tp_data(image, dmap, nx=nx, ny=ny)

    interpolator = RegularGridInterpolator((xpixels, ypixels), TPheights, method='cubic')
    XX, YY = np.meshgrid(fullxgrid, fullygrid, sparse=True)
    res = interpolator((XX, YY))

    if full_coords:
        res = {"TPheightPixel": res}
        for i, vname in enumerate(["TPECEFx", "TPECEFy", "TPECEFz"]):
            interpolator = RegularGridInterpolator((xpixels, ypixels), TPpos[..., i], method='cubic')
            res[vname] = interpolator((XX, YY))

    return res


def TP_data(image, pointing, var, ny=10, cols=None, rows=None, planets_file=None):
    assert len(var) > 0
    can_be_calculated = ["height", "lat", "lon", "sza"]
    if not set(var).issubset(can_be_calculated):
        raise RuntimeError(f"Only the following variables can be calculated: {can_be_calculated}")

    if cols is None or len(cols) > 1:
        nx = np.minimum(5, int(np.floor((image['NCOL'] + 1) / 9)))
    else:
        nx = int(np.floor(image['NCOL'] / 2))
    xpixels, ypixels, TPheights, TPpos = sparse_tp_data(image, pointing.get_deg_map(image), nx=nx, ny=ny)

    fullxgrid = np.arange(image['NCOL'] + 1) if cols is None else cols
    fullygrid = np.arange(image['NROW']) if rows is None else rows
    print(nx, ny, TPheights.shape)
    XX, YY = np.meshgrid(fullxgrid, fullygrid, sparse=True)

    on_ref_grid = {"height": TPheights}

    if len(set(var) & set(["lon", "lat", "sza"])) > 0:
        timescale = load.timescale()
        t = timescale.from_datetime(image["time"])
        # dt = seconds2DT(image["time"])
        # eci2ecef = R.from_matrix(itrs.rotation_at(t))
        # TPwgs = [x.reshape(TPheights.shape) for x in ecef2wgs84(eci2ecef.apply(TPpos.reshape(-1, 3)))]
        TPwgs = [x.reshape(TPheights.shape) for x in ecef2wgs84(TPpos.reshape(-1, 3))]
        on_ref_grid["lat"], on_ref_grid["lon"] = [np.rad2deg(TPwgs[i]) for i in range(2)]

        if "sza" in var:
            if planets_file is None:
                raise RuntimeError("Planetary data needed for solar zenith angle calculation!")
            planets = load(planets_file)
            out = np.zeros_like(on_ref_grid["height"], dtype=float)
            with np.nditer([on_ref_grid[x] for x in ["height", "lat", "lon"]] + [out], [],
                           [['readonly'], ['readonly'], ['readonly'], ['writeonly', 'allocate']]) as it:
                for height, lat, lon, sza in it:
                    tpPos = planets['earth'] + wgs84.latlon(lat, lon, elevation_m=height)
                    sza[...] = 90.0 - tpPos.at(t).observe(planets['sun']).apparent().altaz()[0].degrees
                on_ref_grid["sza"] = it.operands[3]
            planets.close()

    return {v: RegularGridInterpolator((xpixels, ypixels), on_ref_grid[v], method='cubic')((XX, YY)) for v in var}


def col_heights(image, x, pointing, ypixels=None, spline=False, splineTPpos=False):
    if ypixels is None:
        ypixels = np.arrange(image["NROW"])
    d = image['time']
    ts = load.timescale()
    t = ts.from_datetime(d)
    ecipos = image['afsGnssStateJ2000'][0:3]
    q = image['afsAttitudeState']
    quat = R.from_quat(np.roll(q, -1))
    qprime = R.from_quat(image['qprime'])
    ths = np.zeros_like(ypixels)
    TPpos = np.zeros((len(ypixels), 3))
    xdeg, ydeg = pointing.get_deg_map(image)[:, x, ypixels]
    for iy, y in enumerate(ydeg):
        los = R.from_euler('XYZ', [0, y, xdeg], degrees=True).apply([1, 0, 0])
        ecivec = quat.apply(qprime.apply(los))
        res = findtangent(t, ecipos, ecivec)
        TPpos[iy, :] = ecipos + res.x * ecivec
        ths[iy] = res.fun
    if spline:
        return CubicSpline(ypixels, ths)
    elif splineTPpos:
        return CubicSpline(ypixels, TPpos)
    else:
        return ths


def findtangent(t, pos, FOV, bracket=(1e5, 3e5)):
    res = minimize_scalar(funheight, args=(t, pos, FOV), bracket=bracket)
    return res


def funheight(s, t, pos, FOV):
    newp = pos + s * FOV
    newp = ICRF(Distance(m=newp).au, t=t, center=399)
    return wgs84.subpoint(newp).elevation.m


def fast_heights(image, pointing, nx=5, ny=10):
    xpixels = np.append(np.arange(0, image['NCOL'], nx), image['NCOL'])
    ypixels = np.append(np.arange(0, image['NROW'] - 1, ny), image["NROW"] - 1)
    ths_tmp = np.zeros([xpixels.shape[0], ypixels.shape[0]])
    for i, col in enumerate(xpixels):
        ths_tmp[i, :] = col_heights(image, col, pointing, ypixels=ypixels, spline=False)
    interpolator = RegularGridInterpolator((xpixels, ypixels), ths_tmp, method='cubic')
    fullxgrid = np.arange(image['NCOL'] + 1)
    fullygrid = np.arange(image['NROW'])
    XX, YY = np.meshgrid(fullxgrid, fullygrid, sparse=True)
    return interpolator((XX, YY))


def get_deg_map(mdata, dist_data):
    h = 6.9  # height of the CCD in mm
    d = 27.6  # width of the CCD in mm
    f = 261  # effective focal length in mm
    y_disp = h / (f * 511)
    x_disp = d / (f * 2048)

    # xpixx, ypixx = np.meshgrid(np.arange(mdata["NCOL"] + 1), np.arange(mdata["NROW"]), indexing='ij')
    xpix = np.arange(mdata["NCOL"] + 1)
    ypix = np.arange(mdata["NROW"])

    if mdata['channel'] in ["IR1", "IR3", "UV1", "UV2"]:
        x_full = 2048 - mdata["NCSKIP"] + mdata["NCBINCCDColumns"] * (xpix - mdata["NCOL"] - 0.5)
    else:
        x_full = mdata["NCSKIP"] + mdata["NCBINCCDColumns"] * (xpix + 0.5)

    y_full = mdata["NRSKIP"] + mdata["NRBIN"] * (ypix + 0.5)
    xx_full, yy_full = np.meshgrid(x_full, y_full, indexing='ij')

    if dist_data is None:
        xdistortion, ydistortion = [np.zeros_like(xx_full).T for _ in range(2)]
    else:
        tckx, tcky = [[dist_data[f"{mdata['channel']}_spline{coord}_{t}"] for t in ["t", "c", "k"]] + [3, 3]
                      for coord in ["x", "y"]]
        xdistortion, ydistortion = [bisplev(y_full, x_full, tck).squeeze() for tck in [tckx, tcky]]

    xdeg = np.rad2deg(np.arctan(x_disp * (xx_full - 2047.0 / 2 - xdistortion.T)))
    ydeg = np.rad2deg(np.arctan(y_disp * (yy_full - 510.0 / 2 - ydistortion.T)))
    return np.stack([xdeg, ydeg], axis=0)


def verify_channels(conf, metadata):
    limb_channels = ["IR1", "IR2", "IR3", "IR4", "UV1", "UV2"]

    num_chn = len(conf.CHANNELS)
    num_meta = len(metadata)

    if num_chn < 1:
        raise ValueError("Configuration error: empty channel list!")
    invalid_ch = set(conf.CHANNELS).difference(set(limb_channels))
    if len(invalid_ch) > 0:
        raise ValueError("Invalid channels {invalid_ch} specified!")
    if num_meta < 1:
        raise ValueError("No metadata found: were correct input files supplied and read?")
    if num_meta > 1:
        assert num_meta == num_chn, "Configuration requires separate input file for each channel" +\
            f", but got {num_meta} input files for {num_chn} channels!"
        assert set([metadata[i]["channel"][0] for i in range(num_meta)]) == set(conf.CHANNELS), \
            "The input files do not match the channel specification in conf. file! Abort!"
    if num_meta == 1 and conf.SEP_CHN_LOS:
        assert num_chn == 1, f"Conf. requires {num_chn} channels in sep. input files, but got 1 file only!"

import numpy as np
from mats_l2_processing.io import read_ncdf
from scipy.interpolate import RegularGridInterpolator, CubicSpline, bisplev
from scipy.optimize import minimize_scalar
from skyfield.api import load
from scipy.spatial.transform import Rotation as R
from skyfield.api import wgs84
from skyfield.units import Distance
from skyfield.positionlib import ICRF


class Pointing():
    def __init__(self, metadata, conf, const):
        limb_channels = ["IR1", "IR2", "IR3", "IR4", "UV1", "UV2"]
        self.channels = list(set(metadata["channel"].tolist()).intersection(limb_channels))
        if len(self.channels) < 1:
            raise ValueError("Invalid data supplied for pointiing initialization!")

        if conf.DISTORTION_CORRECTION:
            dist_data = read_ncdf(conf.DISTORTION_DATA, None, get_units=False)
        else:
            dist_data = None

        self.deg_map = {}
        for chn in self.channels:
            idx = metadata["channel"].tolist().index(chn)
            self.deg_map[chn] = get_deg_map({v: metadata[v][idx] if len(metadata[v].shape) > 0 else metadata[v]
                                             for v in const.POINTING_DATA}, dist_data=dist_data)

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
    t = load.timescale().from_datetime(image['EXPDate'])
    ecipos = image['afsGnssStateJ2000'][0:3]
    quat = R.from_quat(np.roll(image['afsAttitudeState'], -1))
    qprime = R.from_quat(image['qprime'])

    xxpixels, yypixels = np.meshgrid(xpixels, ypixels, indexing="ij")
    xxdeg, yydeg = [deg_map[c, xxpixels, yypixels] for c in range(2)]
    TPheights = np.empty_like(xxpixels)
    TPpos = np.empty(tuple(list(xxpixels.shape) + [3]))
    with np.nditer([xxdeg, yydeg], flags=["multi_index"]) as it:
        for xx, yy in it:
            los = R.from_euler('XYZ', [0, yy, xx], degrees=True).apply([1, 0, 0])
            ecivec = quat.apply(qprime.apply(los))
            tangent = findtangent(t, ecipos, ecivec)
            TPheights[*it.multi_index] = tangent.fun
            TPpos[*it.multi_index, :] = ecipos + tangent.x * ecivec

    return xpixels, ypixels, TPheights, TPpos


def faster_heights(image, pointing, ny=10, cols=None, rows=None):
    fullxgrid = np.arange(image['NCOL'] + 1) if cols is None else cols
    fullygrid = np.arange(image['NROW']) if rows is None else rows
    if len(cols) > 1:
        nx = np.minimum(5, int(np.floor((image['NCOL'] + 1) / 5)))
    else:
        nx = int(np.floor(image['NCOL'] / 2))
    xpixels, ypixels, TPheights, _ = sparse_tp_data(image, pointing.get_deg_map(image), nx=nx, ny=ny)
    interpolator = RegularGridInterpolator((xpixels, ypixels), TPheights, method='cubic')
    XX, YY = np.meshgrid(fullxgrid, fullygrid, sparse=True)
    return interpolator((XX, YY))


def col_heights(image, x, pointing, ypixels=None, spline=False, splineTPpos=False):
    if ypixels is None:
        ypixels = np.arrange(image["NROW"])
    d = image['EXPDate']
    ts = load.timescale()
    t = ts.from_datetime(d)
    ecipos = image['afsGnssStateJ2000'][0:3]
    q = image['afsAttitudeState']
    quat = R.from_quat(np.roll(q, -1))
    qprime = R.from_quat(image['qprime'])
    ths = np.zeros_like(ypixels)
    TPpos = np.zeros((len(ypixels), 3))
    xdeg, ydeg = pointing.get_deg_map(image)[:, x, ypixels]
    breakpoint()
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

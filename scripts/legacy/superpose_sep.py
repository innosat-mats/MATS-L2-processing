import sys
import numpy as np
import netCDF4 as nc

from mats_l2_processing.obs import time_offsets, cross_maps, reinterpolate3, remove_background_ds
from mats_l2_processing.io import read_multi_ncdf, add_ncdf_vars
from mats_l2_processing.parameters import make_conf
from mats_l2_processing.pointing import Pointing


CHNAMES = ["IR1", "IR2", "IR3", "IR4"]
# Channel effective widths in nm
CH_WIDTHS = [3.577769605779391, 8.1656558203897, 3.192647612468147, 3.2126844284028753]
# Relative intensities of Rayleigh single scattering of solar spectrum, normalized to IR4
CH_RAYLEIGH_SCALES = [1.10046208, 1.09399409, 1.19713362, 1.0]
# Usage: python superpose.py [IR1-IR4 file names, IR1 to be appended]


def remove_scalars(file):
    with nc.Dataset(file, 'r+') as nf:
        # strtype = nf["DataLevel"].dtype
        vs = nf.variables.copy()
        for v in vs:
            is_number = np.issubdtype(nf[v].dtype, np.integer) or np.issubdtype(nf[v].dtype, np.floating)
            if (nf[v].ndim == 0) and is_number:
                print(v, nf[v].dtype)
                value, vdtype = nf[v][:], nf[v].dtype
                nf.renameVariable(v, f"{v}_scalar")
                ncvar = nf.createVariable(v, vdtype, ("time",))
                ncvar[:] = value


def main():
    files = sys.argv[1].split(",")
    num_chn = len(files)
    conf_file = sys.argv[2]
    if len(sys.argv) > 3:
        destray = [float(x) for x in sys.argv[3:3 + num_chn]]
    else:
        destray = [1.0 for _ in range(num_chn)]
    conf, const = make_conf("superpose", conf_file, {})

    # Adapt to changes in zarr datasets
    all_vars = const.ALL_VARS
    all_vars.remove("EXPDate")
    # all_vars.remove("CCDSEL")
    all_vars = all_vars + ["time", "TPECEFx", "TPECEFy", "TPECEFz"]
    times, channels = [], []

    # Get image times first to ensure image alignement
    for file in files:
        with nc.Dataset(file, 'r') as nf:
            times.append(nf["time"][:] * 1e-9)
            channels.append(nf["channel"][:])

    # Align images on different channels
    offsets, numimg = time_offsets(times)

    # Read in data
    data, version = read_multi_ncdf(files, all_vars, offsets=offsets, numimg=numimg)
    for chn in range(len(data)):
        data[chn]["EXPDate"] = data[chn]["time"] * 1e-9
    deg_maps = [Pointing(data[chn], conf, const).chn_map() for chn in range(len(data))]
    deg_maps = [np.swapaxes(dm, 0, 2) for dm in deg_maps]

    # Adapt to changes in zarr datasets
    center_row, center_col = [int(np.round(data[0][var] / 2.0)) for var in ["NROW", "NCOL"]]
    #center_row, center_col = [int(np.round(data[0][var] / 2.0)) for var in ["NROW", "NCOL"]]

    TPecef = np.zeros((numimg, 4))
    CCDSEL_vals = [1, 4, 3, 2]
    for i, name in enumerate(["TPECEFx", "TPECEFy", "TPECEFz"]):
        TPecef[:, i] = data[0][name][:, center_row, center_col]
    for i in range(len(data)):
        data[i]["EXPDate"] = times[i]
        data[i]["CCDSEL"] = np.ones_like(times[i]) * CCDSEL_vals[i]

    # Calculate the positions of IR1 pixels w.r.t. the image centres of other channels
    cmaps = cross_maps(data, const.CCD_VARS, deg_maps)
    # Interpolate all IR channels on IR1 grid
    reint_ds = reinterpolate3(data, deg_maps, cmaps, destray=[True for _ in range(len(channels) - 1)])
    reint_cal = reinterpolate3(data, deg_maps, cmaps, destray=[False for _ in range(len(channels) - 1)])

    reint_ds *= 1e-13
    reint_cal *= 1e-13
    reint_lc = [reint_cal[i, :, :, :] - destray[i + 1] * (reint_cal[i, :, :, :] - reint_ds[i, :, :, :])
                for i in range(num_chn - 1)]
    main_lc = data[0]["ImageCalibrated"] - destray[0] * [data[0]["ImageCalibrated"] - data[0]["ImageDestrayed"]]

    # if version < 0.55:
    #    ir1, ir2, ir3, ir4 = [imag * 1000 / np.mean(data[i]["TEXPMS"]) for i, imag in enumerate([ir1, ir2, ir3, ir4])]
    # else:
    # ir1, ir2, ir3, ir4 = [x * 1e-13 for x in [ir1, ir2, ir3, ir4]]

    # IR1 and IR2 on the same (IR1) grid with backgrounds removed
    irc = remove_background_1ch(main_lc, reint_lc[0],  const.CH_WIDTHS[CHNNAMES.index(channels[0])],
                                [const.CH_RAYLEIGH_SCALES[CHNNAMES.index(name)] for name in channels[:2],
                                recal=conf.RECAL_FAC_IR)
    irc = np.maximum(irc, 0)

    out_spec = [(f"{channels[0]}c", f"{channels[0]} destrayed image with background removed", irc)]

    for k, chn in enumerate(channels[1:]):
        out_spec.append((f"{chn}_ds", f"{chn} destrayed image reinterpolated on {channels[0]} grid", reint_ds[k]))
        out_spec.append((f"{chn}_cal", f"{chn} calibrated image reinterpolated on {channels[0]} grid", reint_cal[k]))

    # Append the results to IR1 file as new variables
    add_ncdf_vars(files[0], "ImageCalibrated", out_spec)
    add_ncdf_vars(files[0], "time", [("EXPDate", "Exposure start time", times[0])],
                  units=[("EXPDate", "Seconds since 2000.01.01 00:00 UTC")])
    add_ncdf_vars(files[0], "qprime", [("afsTangentPointECEF", "Position of center tangent point of IR1", TPecef)])
    add_ncdf_vars(files[0], "time", [("CCDSEL", "Detector number", np.ones_like(times[0]))])
    remove_scalars(files[0])


if __name__ == "__main__":
    main()

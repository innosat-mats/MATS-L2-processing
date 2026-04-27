import sys
import numpy as np
import netCDF4 as nc

from mats_l2_processing.obs_preprocessing import time_offsets, cross_maps, reinterpolate3, remove_background_ds_sep
from mats_l2_processing.io import read_multi_ncdf, add_ncdf_vars
from mats_l2_processing.parameters import make_conf
from mats_l2_processing.pointing import Pointing


CHNAMES = ["AG", "IR3", "IR4"]
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
    files = sys.argv[1:len(CHNAMES) + 1]
    conf_file = sys.argv[len(CHNAMES) + 1]
    if len(sys.argv) > len(CHNAMES) + 2:
        destray = [float(x) for x in sys.argv[len(CHNAMES) + 2:len(CHNAMES) + 5]]
    else:
        destray = [1.0 for x in CHNAMES]
    conf, const = make_conf("superpose", conf_file, {})
    
    # Adapt to changes in zarr datasets
    all_vars = const.ALL_VARS
    all_vars.remove("EXPDate")
    # all_vars.remove("CCDSEL")
    all_vars = all_vars + ["time", "TPECEFx", "TPECEFy", "TPECEFz"]
    times = []

    # Get image times first to ensure image alignement
    for file in files:
        with nc.Dataset(file, 'r') as nf:
            times.append(nf["time"][:] * 1e-9)

    # Align images on different channels
    offsets, numimg = time_offsets(times)

    # Read in data
    data, version = read_multi_ncdf(files, all_vars, offsets=offsets, numimg=numimg)
    for chn in range(len(data)):
        data[chn]["EXPDate"] = data[chn]["time"] * 1e-9
    chnid = data[0]["channel"][0]
    deg_maps = [Pointing(data[chn], conf, const).chn_map() for chn in range(len(data))]
    deg_maps = [np.swapaxes(dm, 0, 2) for dm in deg_maps]

    

    # Adapt to changes in zarr datasets
    center_row, center_col = [int(np.round(data[0][var] / 2.0)) for var in ["NROW", "NCOL"]]
    # center_row, center_col = [int(np.round(data[0][var] / 2.0)) for var in ["NROW", "NCOL"]]

    TPecef = np.zeros((numimg, 4))
    chsel = 1 if data[0]['channel'][0] == 'IR1' else 4
    print(chsel)
    CCDSEL_vals = [chsel, 3, 2]
    for i, name in enumerate(["TPECEFx", "TPECEFy", "TPECEFz"]):
        TPecef[:, i] = data[0][name][:, center_row, center_col]
    for i in range(len(data)):
        data[i]["EXPDate"] = times[i]
        data[i]["CCDSEL"] = np.ones_like(times[i]) * CCDSEL_vals[i]

    # Calculate the positions of IR1 pixels w.r.t. the image centres of other channels
    cmaps = cross_maps(data, const.CCD_VARS, deg_maps)
    # Interpolate all IR channels on IR1 grid
    reint_ds = reinterpolate3(data, deg_maps, cmaps, destray=[True, True])
    reint_cal = reinterpolate3(data, deg_maps, cmaps, destray=[False, False])

    reint_ds *= 1e-13
    reint_cal *= 1e-13

    ir3, ir4 = [reint_cal[i, :, :, :] - destray[i + 1] * (reint_cal[i, :, :, :] - reint_ds[i, :, :, :])
                for i in range(2)]
    ag_cal, ag_ds = [data[0][name][:, :, :] * 1e-13 for name in ["ImageCalibrated", "ImageDestrayed"]]
    ag = ag_cal - destray[0] * (ag_cal - ag_ds)

    # if version < 0.55:
    #    ir1, ir2, ir3, ir4 = [imag * 1000 / np.mean(data[i]["TEXPMS"]) for i, imag in enumerate([ir1, ir2, ir3, ir4])]
    # else:
    # ir1, ir2, ir3, ir4 = [x * 1e-13 for x in [ir1, ir2, ir3, ir4]]

    # IR1 and IR2 on the same (IR1) grid with backgrounds removed
    agf = remove_background_ds_sep(ag, ir3, ir4, const.CH_WIDTHS, const.CH_RAYLEIGH_SCALES, chnid,
                               recal=conf.RECAL_FAC_IR)
    agf = np.maximum(agf, 0)
    # ir2c = np.maximum(ir2c, 0)

    # Append the results to IR1 file as new variables
    add_ncdf_vars(files[0], "ImageCalibrated",
                  # [("IR2_ds", "IR2 destrayed image reinterpolated on IR1 grid", reint_ds[0]),
                  [("IR3_ds", "reinterpolated IR3 destrayed image", reint_ds[0]),
                   ("IR4_ds", "reinterpolated IR4 destrayed image", reint_ds[1]),
                   # ("IR2_cal", "IR2 calibrated image reinterpolated on IR1 grid", reint_cal[0]),
                   ("IR3_cal", "reinterpolated IR3 calibrated image", reint_cal[0]),
                   ("IR4_cal", "reinterpolated IR4 calibrated image", reint_cal[1]),
                   ("ImageFinal", "destrayed image with background removed", agf)])
                   # ("IR2c", "IR2 destrayed image with background removed", ir2c)])
    add_ncdf_vars(files[0], "time", [("EXPDate", "Exposure start time", times[0])],
                  units=[("EXPDate", "Seconds since 2000.01.01 00:00 UTC")])
    add_ncdf_vars(files[0], "qprime", [("afsTangentPointECEF", "Position of center tangent point of IR1", TPecef)])
    add_ncdf_vars(files[0], "time", [("CCDSEL", "Detector number", np.ones_like(times[0]))])
    remove_scalars(files[0])


if __name__ == "__main__":
    main()

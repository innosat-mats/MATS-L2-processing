import sys
import numpy as np

from mats_l2_processing.obs import time_offsets, cross_maps, reinterpolate, remove_background
from mats_l2_processing.io import read_multi_ncdf, add_ncdf_vars


CHNAMES = ["IR1", "IR2", "IR3", "IR4"]
CCD_VARS = ['CCDSEL', 'NCSKIP', 'NRBIN', 'NCOL', 'NRSKIP', 'NROW', 'NCBINCCDColumns',
            "EXPDate", "qprime", "afsAttitudeState", "TEXPMS"]
ALL_VARS = CCD_VARS + ["ImageCalibrated"]

# Usage: python superpose.py [IR1-IR4 file names, IR1 to be appended]


def main():
    files = sys.argv[1:5]
    assert len(files) == len(CHNAMES)

    # Align images on different channels
    offsets, numimg = time_offsets(files)

    # Read in data
    data, version = read_multi_ncdf(files, ALL_VARS, offsets=offsets, numimg=numimg)
    # Calculate the positions of IR1 pixels w.r.t. the image centres of other channels
    deg_maps, cmaps = cross_maps(data, CCD_VARS)
    # Interpolate all IR channels on IR1 grid
    reint = reinterpolate(data, deg_maps, cmaps)
    ir2, ir3, ir4 = [reint[i, :, :, :] for i in range(3)]
    ir1 = data[0]["ImageCalibrated"][:ir2.shape[0], :, :]

    if version < 0.55:
        ir1, ir2, ir3, ir4 = [imag * 1000 / np.mean(data[i]["TEXPMS"]) for i, imag in enumerate([ir1, ir2, ir3, ir4])]
    else:
        ir1, ir2, ir3, ir4 = [x * 0.1 for x in [ir1, ir2, ir3, ir4]]

    # IR1 and IR2 on the same (IR1) grid with backgrounds removed
    ir1c, ir2c = remove_background(ir1, ir2, ir3, ir4)
    # Append the results to IR1 file as new variables
    add_ncdf_vars(files[0], "ImageCalibrated",
                  [("IR2", "IR2 calibrated image reinterpolated on IR1 grid", ir2),
                   ("IR3", "IR3 calibrated image reinterpolated on IR1 grid", ir3),
                   ("IR4", "IR4 calibrated image reinterpolated on IR1 grid", ir4),
                   ("IR1c", "IR1 calibrated image with background removed", ir1c),
                   ("IR2c", "IR2 calibrated image with background removed", ir2c)])


if __name__ == "__main__":
    main()

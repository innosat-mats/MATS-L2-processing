import sys

from mats_l2_processing.io import add_ncdf_vars, read_L1_ncdf
from mats_l2_processing.obs_preprocessing import denoise, separate_scattered_stray
from mats_l2_processing.parameters import make_conf

conf_file, L1b_file = sys.argv[1:3]
conf, const = make_conf("destray", conf_file, {})

L1 = read_L1_ncdf(L1b_file, var=["ImageCalibrated", "TPsza", "channel"])
chn = L1["channel"][0]
if chn not in ["IR3", "IR4"]:
    raise ValueError(f"This stray light separation script is for IR3 and IR4, but got {chn}!")

denoised = denoise(L1["ImageCalibrated"], conf.IRB_DENOISE_HW, conf.IRB_DENOISE_THR)
add_ncdf_vars(L1b_file, "ImageCalibrated", [("ImageDenoised", "Calibrated image with noise removed", denoised)])
scat, rayleigh = separate_scattered_stray(denoised, conf.FIT_REF_ROWS[chn], conf.FIT_BOT_ROW[chn],
                                          const.IRB_R_SCALE_HEIGHT / const.IRB_Y_PITCH,
                                          valid=L1["TPsza"] < conf.SCAT_MAX_SZA)
add_ncdf_vars(L1b_file, "ImageCalibrated",
              [("ImageStrayScattered", "Stray light due to scattering inside the instrument", scat),
               ("ImageDescat", "Image with scattered stray light removed", rayleigh),
               # ("ImageFit", "Full smoothed image (for testing)", r_og + scat_og),
               # ("ImageSum", "Full smoothed image (for testing)", scat + rayleigh)
               ])

import sys
import numpy as np
from mats_l2_processing.io import read_L1_ncdf, add_ncdf_vars, ncdf_filter_dim
from mats_l2_processing.parameters import make_conf, get_updated_conf
from mats_l2_processing.pointing import Pointing
from mats_l2_processing.obs_preprocessing import find_images, cross_maps, reinterpolate_irreg

IR1f, IR2f, out_file = sys.argv[1:4]
conf, const = make_conf("superpose", "conf.py", {})
conf = get_updated_conf(conf, {"CHANNELS": ["IR1", "IR2"], "SEP_CHN_LOS": True})

IR1d = read_L1_ncdf(IR1f, var=const.CCD_VARS + ["ImageCalibrated", "TPsza", "satlat", "satlon"], center_times=True)
IR2d = read_L1_ncdf(IR2f, var=const.CCD_VARS + ["ImageCalibrated"], center_times=True)

saa = np.logical_and(np.logical_and(IR1d["satlat"] > -50, IR1d["satlat"] < 0),
                     np.logical_and(IR1d["satlon"] > -90, IR1d["satlon"] < 40))
valid = np.logical_and(IR1d["TPsza"] > 100, np.logical_not(saa))

valid, IR2idx = find_images(IR1d["time_s"], [IR2d["time_s"]], valid_ref=valid)

pointing = Pointing([IR1d, IR2d], conf, const)
deg_maps = [np.swapaxes(pointing.chn_map(chn=chn), 0, 2) for chn in ["IR1", "IR2"]]
cmaps = cross_maps([IR1d, IR2d], const.CCD_VARS, deg_maps)
IR2c = reinterpolate_irreg(IR2d["ImageCalibrated"], IR2idx, deg_maps[1], cmaps[1])

ncdf_filter_dim(IR1f, "time", np.where(valid)[0], out_file)
add_ncdf_vars(out_file, "ImageCalibrated",
              [("IR2c", "IR2 final image reinterpolated on IR1 grid", IR2c * const.CHN_WIDTHS["IR2"]),
               ("ImageFinal", "Final image for L2 processing",
                IR1d["ImageCalibrated"][valid, :, :] * const.CHN_WIDTHS["IR1"])])


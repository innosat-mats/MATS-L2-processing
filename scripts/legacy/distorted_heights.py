from mats_l2_processing.pointing import Pointing, faster_TP_wgs
from mats_l2_processing.io import read_L1_ncdf, append_gen_ncdf
from mats_l2_processing import parameters
from mats_l2_processing.util import multiprocess
import sys

conf, const = parameters.make_conf("heights", "conf.py", {})
metadata = read_L1_ncdf(sys.argv[1], var=const.POINTING_DATA)
pointing = Pointing(metadata, conf, const)
heights, lats, lons = multiprocess(faster_TP_wgs, metadata, const.POINTING_DATA, int(sys.argv[2]), pointing,
                                   unzip=True, stack=False)
shape = [metadata["size"], -1, heights.shape[1]]
append_gen_ncdf(sys.argv[1], {"TPHeights_dist": ["Tangent point heights, with distortion compensation",
                                                 "meter", heights.reshape(shape), ("EXPDate", "im_row", "im_col")],
                              "TPlat_dist": ["Tangent point latitude, with distortion compensation",
                                             "degree_north", lats.reshape(shape), ("EXPDate", "im_row", "im_col")],
                              "TPlon_dist": ["Tangent point longitude, with distortion compensation",
                                             "degree_east", lons.reshape(shape), ("EXPDate", "im_row", "im_col")]})

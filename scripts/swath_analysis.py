import numpy as np
import netCDF4 as nc
import sys

from scipy.spatial.transform import Rotation as R
from skyfield import api as sfapi
from skyfield.framelib import itrs


from mats_l2_processing.io import read_ncdf
from mats_l2_processing.pointing import Pointing, sparse_tp_data
from mats_l2_processing.parameters import make_conf
from mats_l2_processing.util import get_image, seconds2DT

qprimes = {
    "UV1": [-0.706580513080722, -0.004295847702272496, 0.7075972934329131, -0.0056209032653599155],
    "UV2": [-0.7031617259933777, 0.00360906852416851, 0.7110204941424538, -0.000647017636943995],
    "IR1": [-0.7057723224737625, 0.0035383645153150116, 0.7083764861841564, 0.008698426750597564],
    "IR2": [-0.7053262360001455, -6.124729680245154e-05, 0.7088757046532004, -0.0031831448387146235],
    "IR3": [-0.7055046242225401, 0.0022405671666281793, 0.7086970257935958, 0.0025943574725911466]}

conf, const = make_conf("heights", "conf.py", {})

metadata = read_ncdf(sys.argv[1], None)
num_img = len(metadata["EXPDate"])
metadata["EXPDate_s"] = metadata["EXPDate"].copy()
metadata["EXPDate"] = np.array([seconds2DT(s) for s in metadata["EXPDate_s"]])
metadata["qprime"] = np.broadcast_to(np.array(qprimes[metadata["channel"][0]])[np.newaxis, :], (num_img, 4))
variables = list(metadata.keys())
variables.remove("im_row")
variables.remove("im_col")


pointing = Pointing(metadata, conf, const)
deg_map = pointing.chn_map()

nx, ny = [int(np.floor(length / 2)) for length in deg_map.shape[1:]]
TPheights = np.zeros((num_img, 3, 3))
TPpos = np.zeros((num_img, 3, 3, 3))
satpos = np.zeros((num_img, 3))

rot_sat_channel = R.from_quat(metadata['qprime'][0, :])
timescale = sfapi.load.timescale()

for i in range(num_img):
    image = get_image(metadata, i, variables)
    _, _, TPheights[i, ...], TPpos_eci = sparse_tp_data(image, deg_map, nx=nx, ny=ny)
    rot_sat_eci = R.from_quat(np.roll(image['afsAttitudeState'], -1))
    current_ts = timescale.from_datetime(image["EXPDate"])
    eci_to_ecef = R.from_matrix(itrs.rotation_at(current_ts))

    satpos[i, :] = eci_to_ecef.apply(image['afsGnssStateJ2000'][0:3])
    for k in range(3):
        for l in range(3):
            TPpos[i, k, l, :] = eci_to_ecef.apply(TPpos_eci[k, l, :])

    if i % 100 == 0:
        print(f"{i}/{num_img}")

with nc.Dataset(sys.argv[1], 'r+') as nf:
    nf.createDimension("xpix", 3)
    nf.createDimension("ypix", 3)
    nf.createDimension("coord", 3)

    ncvar = nf.createVariable("satpos", 'f8', ("num_images", "coord"))
    ncvar.long_name = "Satellite position in Cartesian ECEF"
    ncvar.unit = "meter"
    ncvar[:] = satpos
    ncvar = nf.createVariable("TPheight", 'f8', ("num_images", "xpix", "ypix"))
    ncvar.long_name = "Tangent point heights for selected pixels"
    ncvar.unit = "meter"
    ncvar[:] = TPheights

    ncvar = nf.createVariable("TPpos", 'f8', ("num_images", "xpix", "ypix", "coord"))
    ncvar.long_name = "Tangent point positions in Cartesian ECEF for selected pixels"
    ncvar.unit = "meter"
    ncvar[:] = TPpos

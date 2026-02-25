import sys
import numpy as np
import netCDF4 as nc
from mats_l2_processing.io import read_ncdf, add_ncdf_vars

USEFUL_VARS = ['TPssa', 'TPsza', 'nadir_az', 'nadir_sza', 'satheight', 'satlat', 'satlon']


def dim_filter(ncfile, dimensions):
    res = []
    with nc.Dataset(ncfile, 'r') as nf:
        for v in nf.variables:
            if (nf[v].dimensions == dimensions):
                res.append(v)
    return res


L2_file, L1_file = sys.argv[1:3]
L2_img_time = read_ncdf(L2_file, ["img_time"])["img_time"][:]
L1_img_time = read_ncdf(L1_file, ["time"])["time"] * 1e-9
vars_to_add = list(set(dim_filter(L1_file, ("time",))) & set(USEFUL_VARS))
print(f"Variables that will be added: {vars_to_add}")

idxs = [np.argmin(np.abs(L1_img_time - t)) for t in L2_img_time]
assert np.max(np.abs(L1_img_time[idxs] - L2_img_time)) < 0.1
with nc.Dataset(L1_file, 'r') as nf:
    vars_data = [(v, nf[v].long_name, nf[v][:][idxs]) for v in vars_to_add]
    vars_units = {(v, nf[v].units) for v in vars_to_add}
add_ncdf_vars(L2_file, "img_time", vars_data, units=vars_units)

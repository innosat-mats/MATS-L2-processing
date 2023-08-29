import numpy as np
import pandas as pd
from datetime import datetime, timezone
from mats_utils.rawdata.read_data import read_MATS_data
from mats_utils.geolocation.coordinates import col_heights, satpos
from mats_l1_processing.pointing import pix_deg
import matplotlib.pylab as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from skyfield import api as sfapi
from skyfield.framelib import itrs
from skyfield.positionlib import Geocentric, ICRF
from skyfield.units import Distance
import xarray as xr
from numpy.linalg import inv
import mats_l2_processing.oem as oem
import pickle
import scipy.sparse as sp
import time


def remove_empty_columns(matrix):

    _,I = np.nonzero(np.sum(matrix,0))
    _,I_0 = np.where(np.sum(matrix,0)==0)
    cleaned_matrix = matrix[:,I]

    return cleaned_matrix, I_0,I

def reinsert_zeros(vector, indexes):
    for idx in indexes:
        vector.insert(idx, 0)

# def get_3d_field(k, edges, profiles, SeDiag, xa, Sadiag):

#     y = profiles.reshape(-1)
#     y = np.matrix(y).T

#     xa = np.matrix(xa).T

#     Sa_inv=sp.diags(np.ones([xa.shape[0]]),0).astype('float32') * (1/np.max(y)) * 1e8
#     Se_inv=sp.diags(np.ones([k_reduced.shape[0]]),0).astype('float32') * (1/np.max(y))

#     x_hat = oem.oem_basic_sparse_2(y, k_reduced, xa, Se_inv, Sa_inv, maxiter=1000)

#     return x_hat

def do_inversion(k,y):
    #k_reduced,empty_cols,filled_cols = remove_empty_columns(k)    
    k_reduced = k.tocsc()
    xa=np.ones([k_reduced.shape[1]])
    Sa_inv=sp.diags(np.ones([xa.shape[0]]),0).astype('float32') * (1/np.max(y)) * 1e8
    Se_inv=sp.diags(np.ones([k_reduced.shape[0]]),0).astype('float32') * (1/np.max(y))
    xa=0*xa
    xa = np.matrix(xa).T
    y = np.matrix(y).T
    #%%
    start_time = time.time()
    x_hat = oem.oem_basic_sparse_2(y, k_reduced, xa, Se_inv, Sa_inv, maxiter=1000)
    #x_hat_old = x_hat
    #x_hat = np.zeros([k.shape[1],1])
    #x_hat[filled_cols] = x_hat_old


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    return x_hat


# # %%
# plt.plot(y)
# plt.plot(k_reduced.dot(x_hat),':')
# plt.show()
# # %%
# x_hat_old = x_hat
# x_hat = np.zeros([k.shape[1],1])
# x_hat[filled_cols] = x_hat_old

# # %%
# x_hat_reshape1 = np.array(x_hat).reshape(len(edges[0])-1
# ,len(edges[1])-1
# ,len(edges[2])-1)
# plt.plot(x_hat_reshape1[:,3,:])
# plt.show()

# #%%

# rs = ((edges[0][0:-1]+edges[0][1:])/2)
# lons = (edges[1][0:-1]+edges[1][1:])/2
# lats = (edges[2][0:-1]+edges[2][1:])/2

# #%%
# filename = "/home/olemar/Projects/Universitetet/MATS/MATS-analysis/Donal/retrievals/ecef_to_local.pkl"
# with open(filename, "rb") as file:
#     ecef_to_local = pickle.load(file)

# #%%
# ret_ecef=ecef_to_local.inv().apply(np.array([rs[0]*np.cos(lats),np.zeros(len(lats)),rs[0]*np.sin(lats)]).T)# %%
# ret_lats=np.rad2deg(cart2sph(ret_ecef)[:,2])
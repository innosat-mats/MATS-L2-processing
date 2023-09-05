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


def do_inversion(k, y, Sa_inv=None, Se_inv=None, xa=None):
    """Do inversion

    Detailed description

    Args:
        k: 
        y

    Returns:
        ad
    """
    k_reduced = k.tocsc()
    if xa == None:
        xa=np.ones([k_reduced.shape[1]])
        xa=0*xa
    if Sa_inv == None:
        Sa_inv=sp.diags(np.ones([xa.shape[0]]),0).astype('float32') * (1/np.max(y)) * 1e8
    if Se_inv == None: 
        Se_inv=sp.diags(np.ones([k_reduced.shape[0]]),0).astype('float32') * (1/np.max(y))
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

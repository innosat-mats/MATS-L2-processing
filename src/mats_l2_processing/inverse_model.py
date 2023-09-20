import numpy as np
import mats_l2_processing.oem as oem
import scipy.sparse as sp
import time
from mats_l2_processing.grids import geoid_radius
from scipy import stats


def generate_xa_from_gaussian(altgrid,width=5000,meanheight=90000):
    xa= np.exp(-1/2*(altgrid-meanheight)**2/width**2)

    return xa.flatten()

def do_inversion(k, y, Sa_inv=None, Se_inv=None, xa=None, method='spsolve'):
    """Do inversion

    Detailed description

    Args:
        k: 
        y

    Returns:
        ad
    """
    k_reduced = k.tocsc()
    if xa is None:
        xa=np.ones([k_reduced.shape[1]])
        xa=0*xa
    if Sa_inv == None:
        Sa_inv=sp.diags(np.ones([xa.shape[0]]),0).astype('float32') * (1/np.max(y)) * 1e6
    if Se_inv == None: 
        Se_inv=sp.diags(np.ones([k_reduced.shape[0]]),0).astype('float32') * (1/np.max(y))
    #%%
    start_time = time.time()
    x_hat = oem.oem_basic_sparse_2(y, k_reduced, xa, Se_inv, Sa_inv, maxiter=1000,method=method)
    #x_hat_old = x_hat
    #x_hat = np.zeros([k.shape[1],1])
    #x_hat[filled_cols] = x_hat_old


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    return x_hat

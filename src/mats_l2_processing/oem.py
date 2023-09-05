import scipy as sci
import numpy as np
from scipy import sparse

def oem_basic_sparse_2(y, K, xa, Seinv, Sainv, maxiter):
    
    S = (K.T @ Seinv) @ K + Sainv
    KSe = K.T @ Seinv
    KSey = KSe @ (y-K @ xa)
    xhat = sci.sparse.linalg.spsolve(S,KSey)                              
    xhat = xhat + xa.T
    
    return xhat
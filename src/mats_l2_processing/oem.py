import scipy as sci
import numpy as np
from scipy import sparse

def oem_basic_sparse_2(y, K, xa, Seinv, Sainv, maxiter):
    
    S = (K.T.dot(Seinv)).dot(K) + Sainv
    KSe = (K.T).dot(Seinv)
    KSey = KSe.dot(y-K.dot(xa))
    xhat = sci.sparse.linalg.spsolve(S,KSey)                              
    xhat = np.expand_dims(xhat,1) + xa[:,0]
    
    return xhat
import scipy as sci
import numpy as np
from scipy import sparse

def oem_basic_sparse_2(y, K, xa, Seinv, Sainv, maxiter,method='spsolve'):

    if method == 'spsolve':    
        S = (K.T @ Seinv) @ K + Sainv
        KSe = K.T @ Seinv
        KSey = KSe @ (y-K @ xa)
        xhat = sci.sparse.linalg.spsolve(S,KSey)                              
        xhat = xhat + xa.T

    elif method == 'cgm':
        S = (K.T @ Seinv) @ K + Sainv
        KSe = K.T @ Seinv
        KSey = KSe @ (y - K @ xa)
        precond_diag = 1.0 / S.diagonal()
        xhat, exit_code = sparse.linalg.cg(S.tocsr(), KSey, M=sparse.diags(precond_diag, dtype='float32'))
        if exit_code < 0:
            raise RuntimeError("Conjugate gradients error!")
        if exit_code > 0:
            print("Warning: Conjugate gradients solver failed to achieve " +
                f"the required tolerance in {exit_code} iterations.")
        xhat = xhat + xa.T

    return xhat

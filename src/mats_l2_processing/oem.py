import scipy as sci
import numpy as np
from scipy import sparse
import logging
import time


def oem_basic_sparse_2(y, K, xa, Seinv, Sainv, maxiter,method='spsolve', ys=None, xsc=None):
    if xsc is not None:
        xas = xa / xsc
        K = K.multiply(xsc[np.newaxis, :])
    else:
        xas = xa

    if ys is None:
        ys = K @ xas

    if method == 'spsolve': 
        S = (K.T @ Seinv) @ K + Sainv
        KSe = K.T @ Seinv
        KSey = KSe @ (y - ys)
        xhat = sci.sparse.linalg.spsolve(S, KSey)
        xhat = xhat + xas.T

    elif method == 'cgm':
        S = (K.T @ Seinv) @ K + Sainv
        KSe = K.T @ Seinv
        KSey = KSe @ (y - ys)
        precond_diag = 1.0 / S.diagonal()
        xhat, exit_code = sparse.linalg.cg(S.tocsr(), KSey, M=sparse.diags(precond_diag, dtype='float32'))
        if exit_code < 0:
            raise RuntimeError("Conjugate gradients error!")
        if exit_code > 0:
            print("Warning: Conjugate gradients solver failed to achieve " +
                f"the required tolerance in {exit_code} iterations.")
        xhat += xas.T

    if xsc is not None:
        xhat *= xsc

    return xhat


def lm_iter(xa, xsc, xp, y, fx, K, Seinv, Sainv, lmp):
    tic = time.time()
    xp_r = xp - xa
    Ks = K.multiply(xsc[np.newaxis, :])

    KSey = Sainv @ xp_r + (Ks.T @ Seinv) @ (fx - y)
    S = (Ks.T @ Seinv) @ Ks + Sainv + lmp * sparse.eye(Sainv.shape[0])
    precond_diag = 1.0 / S.diagonal()
    dx, exit_code = sparse.linalg.cg(S.tocsr(), KSey, M=sparse.diags(precond_diag, dtype='float32'))
    if exit_code < 0:
        raise RuntimeError("Conjugate gradients error!")
    if exit_code > 0:
        print("Warning: Conjugate gradients solver failed to achieve " +
              f"the required tolerance in {exit_code} iterations.")
    logging.log(15, f"Solver: calculated new atmosphere state in {time.time() - tic:.3f} s.")
    return xp - dx


def cost_func(x, xa, y, fx, Seinv, Sainv):
    xr = x - xa
    yr = y - fx
    return yr.T @ Seinv @ yr + xr.T @ Sainv @ xr


def csr_clear_row(csr, row):
    csr.data[csr.indptr[row]:csr.indptr[row + 1]] = 0

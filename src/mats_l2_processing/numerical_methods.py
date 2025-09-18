from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import norm
from scipy import sparse
import logging
import time
from sparse_dot_mkl import dot_product_mkl as mdot


class Implicit_sparse_matrix(ABC):
    def __init__(self, shape):
        self.shape = shape

    @abstractmethod
    def dot(vec):
        pass


class LM_mkl_sparse_matrix(Implicit_sparse_matrix):
    def __init__(self, K, Seinv, Sainv, lm=0):
        super().__init__(K.shape)
        self.K = K
        self.Seinv = Seinv
        self.Sainv = Sainv
        self.lm = lm

    def dot(self, vec):
        return mdot(self.K.T, self.Seinv @ mdot(self.K, vec)) + mdot(self.Sainv, vec) + self.lm * vec


def cg_solve(A, b, x_init=None, atol=0, rtol=1e-5, maxiter=5000):
    # mdot(K.T.tocsr(), Seinv @ K.tocsr()) + Sainv

    threshold = np.maximum(rtol * norm(b, ord=2), atol)

    if x_init is None:
        x = np.zeros(A.shape[1])
        residual = b
    else:
        x = x_init
        residual = b - A.dot(x)

    normr = np.dot(residual, residual)
    sqnormr = np.sqrt(normr)
    if sqnormr < threshold:
        logging.warn("CG: Accepted initial guess!")
        return x, sqnormr, 0

    p = residual

    for k in range(maxiter):
        # old_residual = residual
        old_normr = normr

        Ap = A.dot(p)
        alpha = normr / np.dot(p, Ap)
        x += alpha * p
        residual -= alpha * Ap
        normr = np.dot(residual, residual)

        sqnormr = np.sqrt(normr)
        # logging.log(15, f"CG: step {k} - {sqnormr / threshold}")
        if sqnormr < threshold:
            logging.info(f"CG: converged in {k + 1} steps.")
            # rr = b - mdot(A, x)
            # logging.log(15, f"{sqnormr}, {norm(rr, ord=2)}, {norm(b, ord=2)}")
            return x, sqnormr, k + 1

        beta = normr / old_normr
        p = residual + beta * p

    logging.warn(f"CG: Did not achieve convergence in {maxiter} steps!")
    logging.warn(f"CG: Accepting result despite residual being {sqnormr / threshold} times too large!")
    return x, sqnormr, k + 1


def cost_func(x, xa, y, fx, Seinv, Sainv, debug_nan=False):
    xr = x - xa
    yr = nannorm(y - fx, "fx", abort=(not debug_nan))
    obs = yr.T @ Seinv @ yr
    reg = xr.T @ Sainv @ xr
    print(f"Misfit calculation: observational part - {obs:.2e}, regularisation part {reg:.2e}")
    # print(f"xr max: {xr.max()}, yr max: {yr.max}")
    # print(y.shape, fx.shape, Seinv.shape, x.shape, xa.shape, Sainv.shape)}")
    # print(y.shape, fx.shape, Seinv.shape, x.shape, xa.shape, Sainv.shape)
    # return yr.T @ Seinv @ yr + xr.T @ Sainv @ xr
    return obs + reg


def csr_clear_row(csr, row):
    csr.data[csr.indptr[row]:csr.indptr[row + 1]] = 0


def nannorm(data, calc_name, abort=True):
    if np.isnan(data).any():
        if abort:
            raise RuntimeError(f"ERROR: Nan's encountered in {calc_name} calculation! Abort!")
        else:
            logging.warn(f"WARNING: Nan's encountered in {calc_name} calculation, resetting norms to zero.")
            return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        return data

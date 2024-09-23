import numpy as np
from numpy.linalg import norm
from scipy import sparse
import logging
import time
from sparse_dot_mkl import dot_product_mkl as mdot
from functools import partial


def oem_basic_sparse_2(y, K, xa, Seinv, Sainv, maxiter, method='spsolve', ys=None, xsc=None):
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
        xhat = sparse.linalg.spsolve(S, KSey)
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


def lm_iter_og(xa, xsc, xp, y, fx, K, Seinv, Sainv, lmp):
    tic = time.time()
    xp_r = xp - xa
    Ks = K.multiply(xsc[np.newaxis, :])
    print(f"lmp type : {type(lmp)}")
    print(f"K type : {type(K)}")
    print(f"Sainv type : {type(Sainv)}")
    print(f"Seinv type : {type(Seinv)}")
    print(f"Ks shape: {Ks.shape}")
    print(f"y shape: {y.shape}")
    print(f"xp shape: {xp_r.shape}")
    print(f"xsc shape: {xsc.shape}")

    KSey = Sainv @ xp_r + Ks.T @ (Seinv @ (fx - y))
    S = (Ks.T @ Seinv) @ Ks + Sainv + lmp * sparse.eye(Sainv.shape[0])
    print(f"Solver preparation took {time.time() - tic:.3f} s.")
    precond_diag = 1.0 / S.diagonal()
    dx, exit_code = sparse.linalg.cg(S.tocsr(), KSey, M=sparse.diags(precond_diag, dtype='float32'))
    # print(type(S.tocsr()))
    # dx = smkl.sparse_qr_solve_mkl(S.tocsr(), KSey, cast=True)
    # exit_code = 0
    if exit_code < 0:
        raise RuntimeError("Conjugate gradients error!")
    if exit_code > 0:
        print("Warning: Conjugate gradients solver failed to achieve " +
              f"the required tolerance in {exit_code} iterations.")
    logging.log(15, f"Solver: calculated new atmosphere state in {time.time() - tic:.3f} s.")
    return xp - dx


def lm_iter(xa, xsc, xp, y, fx, K, Seinv, Sainv, lmp):
    tic = time.time()
    xp_r = xp - xa
    # Kr = sparse.linalg.csr_matrix(K)
    Sainvr = Sainv.tocsr()
    Ks = K.multiply(xsc[np.newaxis, :])
    Ksr = Ks.tocsr()
    KsTr = Ks.T.tocsr()
    del Ks
    tks = time.time()
    logging.log(15, f"Compute Ks: {tks - tic:.3f} s.")
    KSey = Sainvr @ xp_r + KsTr @ (Seinv @ (fx - y))
    tsey = time.time()
    logging.log(15, f"Compute Ksey: {tsey - tks:.3f} s.")
    S = mdot(KsTr, Seinv @ Ksr) + Sainvr + lmp * sparse.eye(Sainv.shape[0])
    ts = time.time()
    logging.log(15, f"Compute S: {ts - tsey:.3f} s.")
    logging.log(15, f"Solver preparation took {ts - tic:.3f} s.")
    logging.log(15, f"S shape: {S.shape}, Ksey shape: {KSey.shape}, xp shape: {xp.shape}")

    dx, resid_norm, iters = mkl_cg(sparse.csr_matrix(S), KSey, np.zeros_like(xp))
    # dx = smkl.sparse_qr_solve_mkl(sparse.csr_matrix(S), KSey, cast=True)
    # if exit_code < 0:
    #     raise RuntimeError("Conjugate gradients error!")
    # if exit_code > 0:
    #     print("Warning: Conjugate gradients solver failed to achieve " +
    #           f"the required tolerance in {exit_code} iterations.")
    logging.log(15, f"Solver: calculated new atmosphere state in {time.time() - tic:.3f} s.")
    return xp - dx


def mkl_lm_prep_og(K, xsc, Sainv, Seinv):
    tic = time.time()

    Ks = K.multiply(xsc[np.newaxis, :])
    KsTr = Ks.T.tocsr()
    S = mdot(KsTr, Seinv @ Ks.tocsr()) + Sainv

    logging.info(f"LM prep: Jacobian preprocessing for LM copleted in {time.time() - tic:.3f} s.")
    return (S, KsTr)


def mkl_lm_prep(K, xsc, Sainv, Seinv):
    tic = time.time()

    # K = K.multiply(xsc[np.newaxis, :])
    S = mdot(K.T.tocsr(), Seinv @ K.tocsr()) + Sainv

    logging.info(f"LM prep: Jacobian preprocessing for LM copleted in {time.time() - tic:.3f} s.")
    return S


def mkl_iter_wprep(xa, xp, y, fx, Sainv, Seinv, lmp, prep, K, conf):
    tic = time.time()

    KSey = Sainv @ (xp - xa) + K.T.tocsr() @ (Seinv @ (fx - y))
    dx, resid_norm, iters = mkl_cg(sparse.csr_matrix(prep + lmp * sparse.eye(Sainv.shape[0])), KSey, np.zeros_like(xp),
                                   atol=conf.CG_ATOL, rtol=conf.CG_RTOL, maxiter=conf.CG_MAX_STEPS)
    logging.log(15, f"Solver: calculated new atmosphere state in {time.time() - tic:.3f} s.")
    return xp - dx


def mkl_iter_implicit(xa, xp, y, fx, Sainv, Seinv, lmp, K, conf, debug_nan=False):
    tic = time.time()
    KSey = mdot(Sainv, xp - xa) + mdot(K.T, Seinv @ nannorm((fx - y), "fx", abort=(not debug_nan)))
    logging.log(15, "Solver: b calculated.")
    dx, resid_norm, iters = mkl_cg_implicit(K, KSey, Seinv, Sainv, lmp, x_init=np.zeros_like(xp),
                                            atol=conf.CG_ATOL, rtol=conf.CG_RTOL, maxiter=conf.CG_MAX_STEPS)
    logging.log(15, f"Solver: calculated new atmosphere state in {time.time() - tic:.3f} s.")
    return xp - dx


def mkl_iter_wprep_og(xa, xp, y, fx, Sainv, Seinv, lmp, prep, conf):
    tic = time.time()

    S, KsTr = prep
    KSey = Sainv @ (xp - xa) + KsTr @ (Seinv @ (fx - y))
    dx, resid_norm, iters = mkl_cg(sparse.csr_matrix(S + lmp * sparse.eye(Sainv.shape[0])), KSey, np.zeros_like(xp),
                                   atol=conf.CG_ATOL, rtol=conf.CG_RTOL, maxiter=conf.CG_MAX_STEPS)
    logging.log(15, f"Solver: calculated new atmosphere state in {time.time() - tic:.3f} s.")
    return xp - dx


def mkl_cg_explicit(A, b, x_init=None, atol=0, rtol=1e-5, maxiter=5000):

    threshold = np.maximum(rtol * norm(b, ord=2), atol)

    if x_init is None:
        x = np.zeros(A.shape[1])
        residual = b
    else:
        x = x_init
        residual = b - mdot(A, x)

    normr = np.dot(residual, residual)
    sqnormr = np.sqrt(normr)
    if sqnormr < threshold:
        logging.warn("CG: Accepted initial guess!")
        return x, sqnormr, 0

    p = residual

    for k in range(maxiter):
        # old_residual = residual
        old_normr = normr

        Ap = mdot(A, p)
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


def mkl_cg(A, b, **kwargs):
    return mkl_cg_core(partial(mdot, matrix_a=A), A.shape[1], b, **kwargs)


def mkl_cg_implicit(K, b, Seinv, Sainv, lm, **kwargs):
    if type(Seinv) is sparse.eye:
        Adot = partial(mkl_Adot_diag_seinv, K=K, Seinv_dval=Seinv[0, 0], Sainv=Sainv, lm=lm)
    elif type(Seinv) is sparse._dia.dia_matrix:
        Adot = partial(mkl_Adot, K=K, Seinv=Seinv, Sainv=Sainv, lm=lm)
    else:
        raise ValueError(f"Unsupported Seinv type {type(Seinv)}!")
    logging.log(15, "Solver: starting CG...")
    return mkl_cg_core(Adot, K.shape[1], b, **kwargs)


def mkl_cg_core(Adot, A_numrows, b, x_init=None, atol=0, rtol=1e-5, maxiter=5000):
    # mdot(K.T.tocsr(), Seinv @ K.tocsr()) + Sainv

    threshold = np.maximum(rtol * norm(b, ord=2), atol)

    if x_init is None:
        x = np.zeros(A_numrows)
        residual = b
    else:
        x = x_init
        residual = b - Adot(x)

    normr = np.dot(residual, residual)
    sqnormr = np.sqrt(normr)
    if sqnormr < threshold:
        logging.warn("CG: Accepted initial guess!")
        return x, sqnormr, 0

    p = residual

    for k in range(maxiter):
        # old_residual = residual
        old_normr = normr

        Ap = Adot(p)
        alpha = normr / np.dot(p, Ap)
        x += alpha * p
        residual -= alpha * Ap
        normr = np.dot(residual, residual)

        sqnormr = np.sqrt(normr)
        logging.log(15, f"CG: step {k} - {sqnormr / threshold}")
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


def mkl_Adot_diag_seinv(vec, K, Seinv_dval, Sainv, lm):
    # This implements matrix vector product S @ vec, with S = K.T @ Seinv @ K + Sainv + lm * I
    # in case of diagonal Seinv
    return Seinv_dval * mdot(K.T, mdot(K, vec)) + mdot(Sainv, vec) + lm * vec


def mkl_Adot(vec, K, Seinv, Sainv, lm):
    # This implements matrix vector product S @ vec, with S = K.T @ Seinv @ K + Sainv + lm * I
    return mdot(K.T, Seinv @ mdot(K, vec)) + mdot(Sainv, vec) + lm * vec


def cost_func(x, xa, y, fx, Seinv, Sainv, debug_nan=False):
    xr = x - xa
    yr = nannorm(y - fx, "fx", abort=(not debug_nan))
    print(y.shape, fx.shape, Seinv.shape, x.shape, xa.shape, Sainv.shape)
    return yr.T @ Seinv @ yr + xr.T @ Sainv @ xr


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

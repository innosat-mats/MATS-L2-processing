import numpy as np
import mats_l2_processing.oem as oem
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
import time
from mats_l2_processing.io import L2_write_init, L2_write_iter
from mats_l2_processing.forward_model import calc_K
import logging


def generate_xa_from_gaussian(altgrid, width=5000, meanheight=90000):
    xa = np.exp(-1 / 2 * (altgrid - meanheight)**2 / width**2)
    return xa.flatten()


def generate_xa_from_alt_profile(altgrid, profile_xa, profile_alts):
    spline = CubicSpline(profile_alts, profile_xa, extrapolate=True)
    return spline(altgrid).flatten()


def do_inversion(k, y, Sa_inv=None, Se_inv=None, xa=None, ys=None, method='spsolve', xsc=None):
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
        xa = np.ones([k_reduced.shape[1]])
        xa = 0 * xa
    if Sa_inv is None:
        Sa_inv = sp.diags(np.ones([xa.shape[0]]), 0).astype('float32') * (1 / np.max(y)) * 1e6
    if Se_inv is None:
        Se_inv = sp.diags(np.ones([k_reduced.shape[0]]), 0).astype('float32') * (1 / np.max(y))
    start_time = time.time()
    x_hat = oem.oem_basic_sparse_2(y, k_reduced, xa, Se_inv, Sa_inv, maxiter=1000, method=method, ys=ys, xsc=xsc)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    return x_hat


def x2atm(x, shape, scales):
    size = shape[0] * shape[1] * shape[2]
    return [x[:size].reshape(shape) * scales[0], x[size:].reshape(shape) * scales[1]]


def reset_invalid(x, scales, bounds):
    assert not np.isnan(x).any(), "Nan's found in solution! Abort!"
    size = int(len(x) / len(scales))
    lims = np.zeros((2, len(x)))
    for i in range(len(scales)):
        for j in range(2):
            lims[j, i * size: (i + 1) * size] = bounds[i][j] / scales[i]

    x = np.maximum(x, lims[0, :])
    x = np.minimum(x, lims[1, :])
    return x


def tp_alt_range(K, j_aux, alt_range):
    tan_alts = j_aux["tan_alts"].flatten()
    invalid = np.logical_or(tan_alts < alt_range[0] * 1e3, tan_alts > alt_range[1] * 1e3)
    invalid = np.hstack((invalid, invalid))
    K2 = K.tocsr()
    for i, v in enumerate(invalid):
        if v:
            oem.csr_clear_row(K2, i)
    num_valid = len(invalid) - np.sum(invalid)
    logging.info(f"{num_valid} observations ({num_valid / len(invalid) * 100:.2f}%) in the specified altitude range.")
    return K2.tocoo()


def contributions(x, xa, y, fx, Seinv, terms, channels, atm_vars):
    obs_size = int(len(y) / len(channels))
    assert y.shape == fx.shape
    yr = y - fx
    for i, chn in enumerate(channels):
        yc = np.zeros_like(y)
        yc[i * obs_size:(i + 1) * obs_size] = yr[i * obs_size:(i + 1) * obs_size]
        logging.info(f"Contributions: {chn} {(yc.T @ Seinv) @ yc:.2e}")

    atm_size = int(len(x) / len(atm_vars))
    assert x.shape == xa.shape
    xr = x - xa
    for i, v in enumerate(atm_vars):
        xc = np.zeros_like(x)
        xc[i * atm_size:(i + 1) * atm_size] = xr[i * atm_size:(i + 1) * atm_size]
        for name, term in terms.items():
            logging.info(f"Contributions: {v} {name} {(xc.T @ term) @ xc:.2e}")


def limit_alt(jb, y, tan_alts, alt_range):
    # Setting altitude range in which observations will be considered
    valid_alt = np.logical_and(tan_alts > alt_range[0] * 1e3, tan_alts < alt_range[1] * 1e3)
    y_ar = y.copy()
    for k in range(y.shape[0]):
        y_ar[k, :, :] = np.where(valid_alt, y[k, :, :], 0.0)
    y_ar = y_ar.flatten()
    valid_alt = valid_alt.reshape((-1, len(jb["columns"]), len(jb["rows"])))
    logging.info(f"Using measurements in the altitude range {alt_range[0]}-{alt_range[1]} km.")
    logging.info(f"They amount to {np.mean(valid_alt) * 100:.2f}% of all measurements.")
    return y_ar, valid_alt


def lm_solve(atm_apr, o2, obs, geo, conf, rt_data, nproc, prefix,
             save_K=False, load_K=False, debug_nan=False):
    scales = (conf.VER_SCALE, conf.T_SCALE)
    bounds = (conf.VER_BOUNDS, conf.T_BOUNDS)
    lm_start_time = time.time()
    # tracemalloc.start()

    # Initializing common variables
    channels, atm_vars = ["IR1", "IR2"], ["VER", "T"]
    atm_shape = atm_apr[0].shape
    atm_size = len(atm_apr[0].flatten())
    assert len(bounds) == len(scales)
    xa = np.hstack([atm_apr[i].flatten() / scales[i] for i in range(len(scales))])
    xsc = np.hstack([sc * np.ones(atm_size) for sc in scales])
    y_ar, valid_alt = limit_alt(geo, obs["y"], obs["tan_alts"], conf.TP_ALT_RANGE)

    # Initializing iteration variables
    xp = xa
    lm_par = conf.LM_PAR_0
    K = None

    #  Initial Jacobian
    if load_K:
        K = sp.load_npz(f"{prefix}_1_K.npz")
        fxp = np.load(f"{prefix}_1_fxp.npz")
        logging.info("Jacobian loaded from file.")
    else:
        K, fxp = calc_K(geo, rt_data, o2, x2atm(xp, atm_shape, scales), xsc, valid_alt, nproc)
        if save_K:
            sp.save_npz(f"{prefix}_1_K.npz", K)
            with open(f"{prefix}_1_fxp.npz", 'wb') as f:
                np.save(f, fxp)

    L2_write_init(geo, prefix, atm_apr, obs["y"], fxp, np.reshape(obs["tan_alts"],
                  (-1, len(geo["columns"]), len(geo["rows"]))), geo["k_alt_range"], conf.TP_ALT_RANGE)
    cf_p = oem.cost_func(xa, xa, y_ar, fxp.flatten(), obs["Se_inv"], geo["Sa_inv"])
    logging.info(f"Initial misfit: {cf_p:.2e}")

    #  Main iteration
    for it in range(1, conf.LM_IT_MAX + 1):
        # sols, cfs, fxs = [], [], []  # In
        best_sol, sol = {}, {}
        if it > 1:
            del K  # Clear previous Jacobian
            K, fxp = calc_K(geo, rt_data, o2, x2atm(xp, atm_shape, scales), xsc, valid_alt, nproc, debug_nan=debug_nan)

        lms = [lm_par * float(conf.LM_FAC) ** x for x in range(-1, conf.LM_MAX_FACTS_PER_ITER + 1)]
        for j, lm in enumerate(lms):
            # xhat = oem.lm_iter(xa, xsc, xp, y_ar, fxp.flatten(), K, Seinv, Sainv, lm)
            sol["lm"] = lm
            xhat = oem.mkl_iter_implicit(xa, xp, y_ar, fxp.flatten(), geo["Sa_inv"], obs["Se_inv"],
                                         lm, K, conf, debug_nan=debug_nan)
            sol["sol"] = reset_invalid(xhat, scales, bounds)
            sol["fx"] = calc_K(geo, rt_data, o2, x2atm(sol["sol"], atm_shape, scales), xsc,
                               valid_alt, nproc, fx_only=True, debug_nan=debug_nan)[1]
            sol["cf"] = oem.cost_func(xa, sol["sol"], y_ar, sol["fx"].flatten(), obs["Se_inv"], geo["Sa_inv"],
                                      debug_nan=debug_nan)
            logging.info(f"Iteration {it}: derived solution with lambda={lm:.2e}, misfit {sol['cf']:.2e}.")
            contributions(sol["sol"], xa, y_ar, sol["fx"].flatten(), obs["Se_inv"], geo["terms"], channels, atm_vars)

            if j == 0:
                best_sol = sol.copy()
            else:
                if sol["cf"] < best_sol["cf"]:
                    best_sol = sol.copy()
                cf_ratio = best_sol["cf"] / cf_p
                if cf_ratio > 1.0:
                    if j == len(lms) - 1:
                        logging.critical(f"Iteration {it} failed to converge!")
                        raise RuntimeError("Solution failed to converge!")
                    continue
                else:
                    converged = (cf_ratio > conf.CONV_CRITERION)
                    final = converged or (it == conf.LM_IT_MAX)
                    if converged and (best_sol["lm"] == sol["lm"]):
                        continue
                    sol = {}
                    xp, fxp, lm_par, cf_p = best_sol["sol"], best_sol["fx"], best_sol["lm"], best_sol["cf"]
                    logging.info(f"Iteration {it}: accepted solution with lambda={lm_par:.2e}" +
                                 f", misfit {best_sol['cf']:.2e} ({cf_ratio:.2f} of previous)")
                    L2_write_iter(prefix, x2atm(xp, atm_shape, scales), fxp, -1 if final else it)
                    if not final:
                        break
                    logging.info("Convergence reached!" if converged else "Max. number of iterations reached!")
                    logging.info(f"Total LM run time: {time.time() - lm_start_time} s.")
                    return


# def solve_1D(atm_apr, y, Seinv, Sainv, terms, conf, jb, rt_data, o2, nproc, prefix, debug_nan=False):

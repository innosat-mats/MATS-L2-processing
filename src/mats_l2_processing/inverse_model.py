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


def expand_to_shape(data, shape, axis):
    assert len(data.shape) == 1
    assert len(data) == shape[axis]

    other_axes = tuple([x for x in range(len(shape)) if x != axis])
    return np.broadcast_to(np.expand_dims(data, axis=other_axes), shape)


def tikhonov_diff_op(shape, axis, axis_grid, volume_factors=None):
    """
    Creates a sparse matrix L_{axis} with a first order Tikhonov differentiation operator.
    This works for general rectilinear grids, not just regular ones.

    Args:
    shape - spatial grid shape for which operator is to by created;
    axis - number axis along which the operator is meant to differentiate;
    axis_grid - spatial grid coordinate values along that axis (1-D ndarray, must be monotonic);
    volume_factors - this is needed for proper scaling of Sa_inv components in case of
                     irregular grid spacing.

    Returns:
    Sparse matrix (grid size x grid size)
    """
    assert (axis >= 0) and (axis < len(shape))
    assert len(axis_grid) == shape[axis]

    step = 1
    for i in range(axis + 1, len(shape)):
        step *= shape[i]

    grid_steps = axis_grid[1:] - axis_grid[:-1]
    assert all(grid_steps > 0) or all(grid_steps < 0)
    scalings = np.zeros(shape[axis])
    scalings[:-1] = 1.0 / grid_steps
    scalings = expand_to_shape(scalings, shape, axis)
    scalings = scalings.flatten()
    if volume_factors is not None:
        scalings *= volume_factors

    # return sp.diags(-scalings, offsets=0, dtype='float32') + sp.diags(scalings[:-step], offsets=step, dtype='float32')
    return sp.diags([-scalings, scalings], offsets=[0, step], dtype='float64')


def tikhonov_laplacian_op(grids, volume_factors, aspect_ratio=None):
    """
    Creates a sparse matrix operator that implements the Laplacian.

    Args:
    grids - tuple of 1-D ndarrays, grids along each axis (works with general rectilinear grid)

    Returns:
    L - sparse matrix (grid size x grid size).
    """
    shape = np.array([len(grid) for grid in grids])
    size = np.prod(shape)
    L = sp.dia_array((size, size), dtype='float64')
    if aspect_ratio is None:
        aspect_ratio = np.ones(len(grids))

    for axis, grid in enumerate(grids):
        step = 1
        for i in range(axis + 1, len(shape)):
            step *= shape[i]
        steps_up, steps_down, diag_main, diag_up, diag_down = (np.zeros_like(grid) for _ in range(5))
        steps_up[:-1] = grid[1:] - grid[:-1]
        steps_down[1:] = steps_up[:-1].copy()
        diag_main[1:-1] = - 2.0 / steps_up[1:-1] / steps_down[1:-1]
        diag_up[1:-1] = 2.0 / steps_up[1:-1] / (steps_up[1:-1] + steps_down[1:-1])
        diag_down[1:-1] = 2.0 / steps_down[1:-1] / (steps_up[1:-1] + steps_down[1:-1])
        diag_main, diag_up, diag_down = [expand_to_shape(arr, shape, axis).flatten()
                                         for arr in [diag_main, diag_up, diag_down]]
        if volume_factors is not None:
            diag_main, diag_up, diag_down = [arr * volume_factors
                                             for arr in [diag_main, diag_up, diag_down]]
        L += aspect_ratio[axis] * sp.diags([diag_down[step:], diag_main, diag_up[:-step]],
                                           offsets=[-step, 0, step], dtype='float64')
    return L


def Sa_inv_tikhonov(grid, weight_0, diff_weights=[0.0, 0.0, 0.0], laplacian_weight=0.0,
                    volume_factors=False, store_terms=False, aspect_ratio=False):
    """
    Creates Sa_inv for Tikhonov regularisation on general rectilinear grid.
    In n dimentions (i.e. len(grid) == n):

    Sa_inv = weight_0 ** 2 * I + diff_weights[0] ** 2 * L_0.T @ L_0 +...+ diff_weights[n-1] ** 2 * L_{n-1}.T @ L_{n-1}

    where L_{k} is the first order Tikhonov differentiation operator along k-th axis.

    Args:
    grid - tuple of 1-D ndarrays, each represent grid along each axis in order,
    weight_0 - float,
    diff_weights - list of float, regularisation weights for each axis,
    volume_factors - scale the matrix elements according to volume of each grid cell.
                     This is only relevant for general rectilinear (i.e. not regular) grids.
    store_terms - store each term of Sa_inv separately. Useful for retrieval parameter determination.

    Returns:
    Sa_inv - sparse materix,
    terms - The different terms that are added to construct Sa_inv stored separately. "None" if store_terms is False.

    """
    assert all([len(g.shape) == 1 for g in grid])
    shape = [len(g) for g in grid]

    diagonal = np.ones(shape)
    if volume_factors:
        for i, g_i in enumerate(grid):
            dimensions_i = np.zeros_like(g_i)
            lengths = g_i[1:] - g_i[:-1]
            dimensions_i[:-1] = 0.5 * lengths
            dimensions_i[1:] += 0.5 * lengths
            diagonal *= expand_to_shape(dimensions_i, shape, i)
        diagonal /= np.sum(diagonal)
    diagonal = diagonal.flatten()
    volumes = np.sqrt(diagonal)

    Sa_inv = sp.diags(diagonal * weight_0 ** 2).astype('float64')
    terms = {"Zero-order": Sa_inv.copy()} if store_terms else {}

    names = ["Altitude gradient", "Across-track gradient", "Along-track gradient"]
    for i, g_i in enumerate(grid):
        if diff_weights[i] == 0:
            continue
        L_i = tikhonov_diff_op(shape, i, g_i, volume_factors=(volumes if volume_factors else None))
        term = diff_weights[i] ** 2 * (L_i.T @ L_i)
        Sa_inv += term
        if store_terms:
            terms[names[i]] = term.copy()
        del L_i

    if laplacian_weight != 0:
        L = tikhonov_laplacian_op(grid, volume_factors=(volumes if volume_factors else None),
                                  aspect_ratio=aspect_ratio)
        term = laplacian_weight ** 2 * (L.T @ L)
        Sa_inv += term
        if store_terms:
            terms["Laplacian"] = term.copy()

    return Sa_inv, terms


def Sa_inv_multivariate(grid, weights, volume_factors=False, store_terms=False, aspect_ratio=False):
    """
    Builds Sa_inv for retrievals with two variables on the same grid.

    Args:
    grid - tuple of 1-D ndarrays, each represent grid along each axis in order,
    weights - list of tuples of the form (weight_0, diff_weights[0], diff_weights[1],
    diff_weights[2], laplacian_weights) for each variable.
    volume_factors - scale the matrix elements according to volume of each grid cell.
                     This is only relevant for general rectilinear (i.e. not regular) grids.
    store_terms - store each term of Sa_inv separately. Useful for retrieval parameter determination.

    Returns:
    Sa_inv - sparse matrix.
    terms - The different terms that are added to construct Sa_inv stored separately. "None" if store_terms is False.
    """

    assert len(weights) > 1
    assert all([len(w) == 5 for w in weights])
    Sas = [Sa_inv_tikhonov(grid, w[0], diff_weights=w[1:4], laplacian_weight=w[4], volume_factors=volume_factors,
                           store_terms=store_terms, aspect_ratio=aspect_ratio) for w in weights]

    if store_terms:
        terms = {name: sp.block_diag([Sas[j][1][name] for j in range(len(Sas))]) for name in Sas[0][1].keys()}
    else:
        terms = None
    return sp.block_diag([x[0] for x in Sas]), terms


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


def lm_solve(atm_apr, y, tan_alts, Seinv, Sainv, terms, conf, jb, rt_data, o2, nproc, prefix,
             save_K=False, load_K=False, verify=False):
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
    y_ar, valid_alt = limit_alt(jb, y, tan_alts, conf.TP_ALT_RANGE)

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
        K, fxp = calc_K(jb, rt_data, o2, x2atm(xp, atm_shape, scales), xsc, valid_alt, nproc)
        if save_K:
            sp.save_npz(f"{prefix}_1_K.npz", K)
            with open(f"{prefix}_1_fxp.npz", 'wb') as f:
                np.save(f, fxp)

    L2_write_init(jb, prefix, atm_apr, y, fxp, np.reshape(tan_alts, (-1, len(jb["columns"]), len(jb["rows"]))),
                  jb["k_alt_range"], conf.TP_ALT_RANGE)
    cf_p = oem.cost_func(xa, xa, y_ar, fxp.flatten(), Seinv, Sainv)
    logging.info(f"Initial misfit: {cf_p:.2e}")

    #  Main iteration
    for it in range(1, conf.LM_IT_MAX + 1):
        # sols, cfs, fxs = [], [], []  # In
        best_sol, sol = {}, {}
        if it > 1:
            del K  # Clear previous Jacobian
            K, fxp = calc_K(jb, rt_data, o2, x2atm(xp, atm_shape, scales), xsc, valid_alt, nproc, verify_results=verify)

        lms = [lm_par * float(conf.LM_FAC) ** x for x in range(-1, conf.LM_MAX_FACTS_PER_ITER + 1)]
        for j, lm in enumerate(lms):
            # xhat = oem.lm_iter(xa, xsc, xp, y_ar, fxp.flatten(), K, Seinv, Sainv, lm)
            sol["lm"] = lm
            xhat = oem.mkl_iter_implicit(xa, xp, y_ar, fxp.flatten(), Sainv, Seinv, lm, K, conf)
            sol["sol"] = reset_invalid(xhat, scales, bounds)
            sol["fx"] = calc_K(jb, rt_data, o2, x2atm(sol["sol"], atm_shape, scales), xsc,
                               valid_alt, nproc, fx_only=True, verify_results=verify)[1]
            sol["cf"] = oem.cost_func(xa, sol["sol"], y_ar, sol["fx"].flatten(), Seinv, Sainv)
            logging.info(f"Iteration {it}: derived solution with lambda={lm:.2e}, misfit {sol['cf']:.2e}.")
            contributions(sol["sol"], xa, y_ar, sol["fx"].flatten(), Seinv, terms, channels, atm_vars)

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

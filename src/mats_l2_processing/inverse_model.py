import numpy as np
import mats_l2_processing.oem as oem
import scipy.sparse as sp
import time
from mats_l2_processing.grids import geoid_radius
from scipy import stats


def generate_xa_from_gaussian(altgrid, width=5000, meanheight=90000):
    xa = np.exp(-1 / 2 * (altgrid - meanheight)**2 / width**2)

    return xa.flatten()


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
    return sp.diags([-scalings, scalings], offsets=[0, step], dtype='float32')


def tikhonov_laplacian_op(grids, volume_factors):
    """
    Creates a sparse matrix operator that implements the Laplacian.

    Args:
    grids - tuple of 1-D ndarrays, grids along each axis (works with general rectilinear grid)

    Returns:
    L - sparse matrix (grid size x grid size).
    """
    shape = np.array([len(grid) for grid in grids])
    size = np.prod(shape)
    L = sp.dia_array((size, size), dtype='float32')

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
        L += sp.diags([diag_down[step:], diag_main, diag_up[:-step]], offsets=[-step, 0, step], dtype='float32')
    return L


def Sa_inv_tikhonov(grid, weight_0, diff_weights=[0.0, 0.0, 0.0], laplacian_weight=0.0,
                    volume_factors=False, store_terms=False):
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
        diagonal /= np.mean(diagonal)
    diagonal = diagonal.flatten()

    Sa_inv = sp.diags((diagonal * weight_0) ** 2).astype('float32')
    terms = {"Zero-order": Sa_inv.copy()} if store_terms else {}

    names = ["Altitude gradient", "Across-track gradient", "Along-track gradient"]
    for i, g_i in enumerate(grid):
        if diff_weights[i] == 0:
            continue
        L_i = tikhonov_diff_op(shape, i, g_i, volume_factors=(diagonal if volume_factors else None))
        term = diff_weights[i] ** 2 * (L_i.T @ L_i)
        Sa_inv += term
        if store_terms:
            terms[names[i]] = term.copy()
    del L_i

    if laplacian_weight != 0:
        L = tikhonov_laplacian_op(grid, volume_factors=(diagonal if volume_factors else None))
        term = laplacian_weight ** 2 * (L.T @ L)
        Sa_inv += term
        if store_terms:
            terms["Laplacian"] = term.copy()

    return Sa_inv, terms


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
    x_hat = oem.oem_basic_sparse_2(y, k_reduced, xa, Se_inv, Sa_inv, maxiter=1000, method=method)
    #x_hat_old = x_hat
    #x_hat = np.zeros([k.shape[1],1])
    #x_hat[filled_cols] = x_hat_old

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    return x_hat

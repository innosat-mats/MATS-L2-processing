import logging
import numpy as np
import scipy.sparse as sp
import time
from sparse_dot_mkl import dot_product_mkl

from mats_l2_processing.numerical_methods import LM_matrix, cg_solve
from mats_l2_processing.io import append_gen_ncdf


def calc_diagnostic(diag_data, grid, method, fname, conf, const):
    if method == 'avk':
        if "c_matrix_rows" not in diag_data.keys():
            calc_c_matrix_rows(diag_data, grid, conf)
        calc_avk(diag_data, grid, conf)
        write_diag_for_grid(diag_data, "avk_rows", grid, conf, fname,
                            "_avk_", " averaging kernel contribution from ", None,
                            write_cross_terms=True, scales=np.ones(len(conf.RET_QTY)))
    elif method == 'noise_err':
        if "c_matrix_rows" not in diag_data.keys():
            calc_c_matrix_rows(diag_data, grid, conf)
        if "gain_rows" not in diag_data.keys():
            calc_gain(diag_data, grid, conf)
        calc_noise_err(diag_data, grid, conf)
        write_scalar_diag(diag_data, "noise_err", grid, conf, fname, "_noise_err",
                          " error due to measurement noise", 'qty')
    elif method == 'smooth_err':
        if "c_matrix_rows" not in diag_data.keys():
            calc_c_matrix_rows(diag_data, grid, conf)
        if "avk_rows" not in diag_data.keys():
            calc_avk(diag_data, grid, conf)
        calc_smooth_err(diag_data, grid, conf)
        write_scalar_diag(diag_data, "smooth_err", grid, conf, fname, "_smooth_err",
                          " error due to smoothing", 'qty')
    else:
        raise NotImplementedError(f"Diagnostic {method} not implemented!")


def calc_c_matrix_rows(diag_data, grid, conf):
    A = LM_matrix(diag_data["K"], diag_data["Se_inv"], diag_data["Sa_inv"], lm=0)

    numidxs = len(diag_data["idxs"])
    res = []
    for c, idx in enumerate(diag_data["idxs"]):
        tic = time.time()
        b = np.zeros(diag_data["Sa_inv"].shape[0])
        b[idx] = 1
        row, _, _ = cg_solve(A, b, maxiter=30000, rtol=conf.DIAG_CG_RTOL)
        res.append(row)
        logging.log(15, f"Diagnostics: calculated C matrix row {c + 1} of {numidxs} in {time.time() - tic:.1f} s.")

    diag_data["c_matrix_rows"] = np.stack(res, axis=0)


def calc_avk(diag_data, grid, conf):
    mdot = dot_product_mkl if type(diag_data["K"]) is sp.csr_matrix else np.dot
    numidxs = len(diag_data["idxs"])

    tic = time.time()
    avk = np.zeros_like(diag_data["c_matrix_rows"])

    for i, idx in enumerate(diag_data["idxs"]):
        avk[i, :] = mdot(diag_data["K"].T, diag_data["Se_inv"] @ mdot(diag_data["K"], diag_data["c_matrix_rows"][i, :]))
    diag_data["avk_rows"] = avk
    logging.log(15, f"Diagnostics: calculated avk for {numidxs} rows in {time.time() - tic:.1f} s.")


def calc_gain(diag_data, grid, conf):
    mdot = dot_product_mkl if type(diag_data["K"]) is sp.csr_matrix else np.dot
    numidxs = len(diag_data["idxs"])

    tic = time.time()
    gain = np.zeros((numidxs, diag_data["K"].shape[0]))
    for i, idx in enumerate(diag_data["idxs"]):
        gain[i, :] = diag_data["Se_inv"] @ mdot(diag_data["K"], diag_data["c_matrix_rows"][i, :])
    diag_data["gain_rows"] = gain
    logging.log(15, f"Diagnostics: calculated gain for {numidxs} rows in {time.time() - tic:.1f} s.")


def calc_noise_err(diag_data, grid, conf):
    numidxs = len(diag_data["idxs"])

    tic = time.time()
    noise_err = np.zeros(numidxs)

    Se_inv_diagonal = diag_data["Se_inv"].diagonal()
    Se = sp.diags(1 / Se_inv_diagonal, 0).astype('float32')
    for i in range(numidxs):
        noise_err[i] = np.dot(diag_data["gain_rows"][i, :], Se @ diag_data["gain_rows"][i, :])
    diag_data["noise_err"] = noise_err
    logging.log(15, f"Diagnostics: calculated noise error for {numidxs} rows in {time.time() - tic:.1f} s.")


def calc_smooth_err(diag_data, grid, conf):
    numidxs = len(diag_data["idxs"])

    tic = time.time()
    smooth_err = np.zeros(numidxs)
    for i in range(numidxs):
        b = diag_data["avk_rows"][i, :].copy()
        b[diag_data["idxs"][i]] -= 1
        Sab, _, _ = cg_solve(diag_data["Sa_inv"], b, maxiter=30000, rtol=conf.DIAG_CG_RTOL)
        smooth_err[i] = np.dot(Sab, b)
    diag_data["smooth_err"] = smooth_err
    logging.log(15, f"Diagnostics: calculated smoothing error for {numidxs} rows in {time.time() - tic:.1f} s.")


#  Ncdf output ----------------------------------------------------------------------------------------


def write_diag_base_info(grid, grid_idxs, fname):
    num_diag = len(grid_idxs)
    idxs = np.stack(grid_idxs, axis=0)
    coords = np.zeros_like(idxs, dtype=np.float64)
    for c, coord in enumerate(grid.points):
        coords[:, c] = coord[idxs[:, c]] / grid.scalings[c] * 1e3

    dim_pars = {"diag_point": ("Enumeration of points for which diagnostics were calculated", None,
                               np.arange(num_diag)),
                "diag_coord": ("Diagnostics point coordinate number", None, np.arange(len(grid.points)))}
    ncvars = {"diag_point_index": ("Grid indices of points for which diagnostics were calculated", None,
                                   idxs, ("diag_point", "diag_coord")),
              "diag_point_coord": ("Coordinates of points for which diagnostics were calculated", "meter",
                                   coords, ("diag_point", "diag_coord"))}
    append_gen_ncdf(fname, ncvars, dim_spec=dim_pars, overwrite=True)


def write_scalar_diag(diag_data, diag_vname, grid, conf, fname, short_suffix, long_suffix, units, scales=None):
    num_diag = diag_data[diag_vname].shape[0]
    numpoints = int(num_diag / len(conf.DIAG_QTY))
    unit_list = [grid.ncpar[qty][1] for qty in conf.DIAG_QTY] if units == 'qty' else [units for _ in conf.DIAG_QTY]
    ncvars = {}
    for i, pqty in enumerate(conf.DIAG_QTY):
        qty_points = np.arange(num_diag, dtype=np.int64)[diag_data["qty_idxs"] == i]
        assert len(qty_points) == numpoints, f"{qty_points} {numpoints}"
        qty_data = np.stack([diag_data[diag_vname][p] for p in qty_points], axis=0)
        ncvars[f"{pqty}{short_suffix}"] = (f"{pqty}{long_suffix}", unit_list[i], qty_data, ("diag_point",))
    append_gen_ncdf(fname, ncvars)


def write_diag_for_grid(diag_data, diag_vname, grid, conf, fname, short_suffix, long_suffix, units,
                        write_cross_terms=False, scales=None):
    num_diag = diag_data[diag_vname].shape[0]
    numpoints = int(num_diag / len(conf.DIAG_QTY))
    dims = ("diag_point",) + grid.dim_names
    unit_list = [grid.ncpar[qty][1] for qty in conf.RET_QTY] if units == 'qty' else [units for _ in conf.RET_QTY]
    ncvars = {}
    for i, pqty in enumerate(conf.DIAG_QTY):
        qty_points = np.arange(num_diag, dtype=np.int64)[diag_data["qty_idxs"] == i]
        assert len(qty_points) == numpoints, f"{qty_points} {numpoints}"
        aqtys = grid.ret_qty if write_cross_terms else [pqty]
        for j, aqty in enumerate(grid.ret_qty):
            if aqty not in aqtys:
                continue
            if write_cross_terms:
                short_name = f"{pqty}{short_suffix}{aqty}"
                long_name = f"{grid.ncpar[pqty][0]}{long_suffix}{aqty}"
            else:
                short_name = f"{pqty}{short_suffix}"
                long_name = f"{grid.ncpar[pqty][0]}{long_suffix}"
            qty_data = np.stack([grid.vec2atm(diag_data[diag_vname][p, :], custom_scales=scales)[j]
                                 for p in qty_points], axis=0)
            ncvars[short_name] = (long_name, unit_list[j], qty_data, dims)

    append_gen_ncdf(fname, ncvars)

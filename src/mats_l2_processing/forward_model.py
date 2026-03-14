from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
import time
import logging

import mats_l2_processing.interpolator as interpolator
from mats_l2_processing.util import multiprocess


class Forward_model(ABC):
    def __init__(self, conf, const, grid, obs, aux, combine_images=None, debug_nan=False):
        self.grid = grid
        self.obs = obs
        self.sep_chn_los = conf.SEP_CHN_LOS

        self.channels = obs.channels
        self.num_channels = len(obs.channels)

        for i in range(len(aux)):
            assert aux[i].shape == tuple(grid.atm_shape[1:]), f"Auxiliary data variable {i} " +\
                "has shape {aux[i].shape}, but grid is of shape {grid.atm_shape[1:]}"
        self.aux = aux

        ndims = len(grid.edges)
        self.sparse = (ndims > 1)
        if self.sparse:
            self.valid_row = Forward_model.valid_row_sparse
            self.hstack, self.vstack = sp.hstack, sp.vstack
        else:
            self.valid_row = Forward_model.valid_row_dense
            self.hstack, self.vstack = np.hstack, np.vstack

        self.combine_images = (ndims > 1) if combine_images is None else combine_images
        if ndims > 1 and not self.combine_images:
            raise ValueError("Configuration error: combine_images is False, but {ndims}-dimentional " +
                             "retrieval is to be performed. This does not make sense.")

        interp_type = conf.INTERPOLATOR
        if interp_type == 'LINEAR':
            if ndims == 1:
                self.interp = interpolator.Linear_interpolator_1D(grid)
            elif ndims == 3:
                self.interp = interpolator.Trilinear_interpolator_3D(grid)
            else:
                raise NotImplementedError(f"LINEAR interolator for {ndims} dimensions not implemented!")
        else:
            raise NotImplementedError(f"Unknown interpolator type {interp_type}!")

        self.valid_points = np.logical_and(self.grid.alt > conf.RET_ALT_RANGE[0] * 1e3,
                                           self.grid.alt < conf.RET_ALT_RANGE[1] * 1e3)
        self.numobs = len(self.obs.valid_obs.flatten())
        self.num_simages = obs.num_simages
        self.im_points = len(obs.columns) * len(obs.rows)
        self.fx_image_shape = (self.num_channels, len(obs.columns), len(obs.rows))
        self.K_image_shape = (self.num_channels * self.im_points, len(self.grid.ret_qty) * self.grid.npoints)
        self.fwdm_vars = const.NEEDED_DATA
        self.debug_nan = debug_nan  # Not implemented yet!
        self.stepsize = grid.stepsize

    @abstractmethod
    def _fwdm_los(self, pos, atm, aux):
        pass

    @abstractmethod
    def _fwdm_jac_los(self, pos, atm, aux):
        pass

    @staticmethod
    def valid_row_sparse(matrix, is_valid, factor=1.0):
        assert matrix.shape[0] == 1, "The matrix must be a row vector!"
        valid_idx = is_valid[matrix.col]
        return sp.coo_matrix((matrix.data[valid_idx], (matrix.row[valid_idx], matrix.col[valid_idx])),
                             shape=matrix.shape).multiply(factor)

    @staticmethod
    def valid_row_dense(matrix, is_valid, factor=1.0):
        return np.where(is_valid, matrix, 0.0) * factor

    def _calc_fwdm_jac_image(self, num_simage, common_args):
        rt_time, los_time, st_time, wh_time = 0.0, 0.0, 0.0, 0.0
        tic = time.time()
        fx_im = np.zeros(self.fx_image_shape)
        valid_obs_image = self.obs.valid_obs[:, num_simage, ...]
        jacs = [[] for _ in range(self.num_channels)]

        if self.combine_images:
            valid_points_flat = self.valid_points.flatten()
            im_atm, im_aux = [item.copy() for item in common_args]
        else:
            valid_points_flat = self.valid_points[num_simage, :].flatten()
            im_atm, im_aux = [[qty[num_simage, ...].copy() for qty in dset] for dset in common_args]

        if self.sparse:
            empty_row = sp.coo_matrix((1, self.K_image_shape[1]), dtype=np.float64)
        else:
            empty_row = np.zeros((1, self.K_image_shape[1]))

        if self.sep_chn_los:
            for j in range(self.num_channels):
                tic_los = time.time()
                im_los = self.obs.calc_los_image(num_simage, j)
                los_time += time.time() - tic_los
                jac_row_list = []
                for idx in np.ndindex(self.fx_image_shape[1:]):
                    if valid_obs_image[j, idx[0], idx[1]]:
                        tic_rt = time.time()
                        fx_los, grads = self._fwdm_jac_los(im_los[idx[0]][idx[1]], im_atm, im_aux)
                        rt_time += time.time() - tic_rt
                        fx_im[j, idx[0], idx[1]] = fx_los[j]
                        tic_wh = time.time()
                        grads = [self.valid_row(grads[i][j], valid_points_flat, factor=scale)
                                 for i, scale in enumerate(self.grid.scales)]
                        jac_row_list.append(self.hstack(grads))
                        wh_time += time.time() - tic_wh
                    else:
                        jac_row_list.append(empty_row)
                tic_st = time.time()
                jacs[j] = self.vstack(jac_row_list)
                st_time += time.time() - tic_st
        else:
            tic_los = time.time()
            im_los = self.obs.calc_los_image(num_simage, 0)
            los_time += time.time() - tic_los
            jac_row_lists = [[] for _ in range(self.num_channels)]
            for idx in np.ndindex(self.fx_image_shape[1:]):
                if valid_obs_image[0, idx[0], idx[1]]:
                    tic_rt = time.time()
                    fx_im[:, idx[0], idx[1]], grads = self._fwdm_jac_los(im_los[idx[0]][idx[1]], im_atm, im_aux)
                    tic_wh = time.time()
                    rt_time += tic_wh - tic_rt
                    for j in range(self.num_channels):
                        for i, scale in enumerate(self.grid.scales):
                            grads[i][j] = self.valid_row(grads[i][j], valid_points_flat, factor=scale)
                        jac_row_lists[j].append(self.hstack([grads[i][j] for i in range(len(self.grid.scales))]))
                    wh_time += time.time() - tic_wh
                else:
                    for j in range(self.num_channels):
                        jac_row_lists[j].append(empty_row)
            jacs = [self.vstack(jac_row_list) for jac_row_list in jac_row_lists]

        for jac in jacs:
            if np.isnan(jac.data if self.sparse else jac).any():
                raise RuntimeError("Jacobian: NaN encountered in jacobian (image {num_simage})! Abort!")
        logging.log(15, f"Jacobian: Image {num_simage} processed in {time.time() - tic:.1f} s" +
                    f"(LOS:{los_time:.1f}, RT:{rt_time:.1f}, ST:{st_time:.1f}, WH:{wh_time:.1f})")
        return jacs, fx_im

    def calc_fwdm_jac(self, atm, processes=1):
        res = multiprocess(self._calc_fwdm_jac_image, self.num_simages, [], processes, [atm, self.aux],
                           unzip=False, numbers_only=True)

        stack_start = time.time()
        if self.combine_images:
            jacs, fx = res.pop(0)
            fx = fx[:, np.newaxis, :, :]
            jacs = [[jac] for jac in jacs]
            while len(res) > 0:
                im_jacs, im_fx = res.pop(0)
                for j, im_jac in enumerate(im_jacs):
                    # jacs[j] = sp.vstack([jacs[j], im_jac])
                    jacs[j].append(im_jac)
                fx = np.concatenate([fx, im_fx[:, np.newaxis, :, :]], axis=1)

            jac = self.vstack(jacs.pop(0))
            while len(jacs) > 0:
                jac = sp.vstack([jac] + jacs.pop(0))
            if self.sparse:
                jac = jac.tocsr()
        else:
            jac, fx = [], []
            while len(res) > 0:
                im_jacs, im_fx = res.pop(0)
                jac.append(self.vstack(im_jacs))
                fx.append(im_fx)
            if self.sparse:
                jac = [j.tocsr() for j in jac]
        del im_jacs, im_fx

        logging.log(15, f"Jacobian: Stacking took {time.time() - stack_start:.1f} s")
        if np.isnan(fx).any():
            raise RuntimeError("Forward model: Nan's encountered in fx! Abort!")
        return jac, fx

    def _calc_fwdm_image(self, num_simage, common_args):
        tic = time.time()
        fx_im = np.zeros(self.fx_image_shape)
        if self.combine_images:
            im_atm, im_aux = [item.copy() for item in common_args]
        else:
            im_atm, im_aux = [[qty[num_simage, ...].copy() for qty in dset] for dset in common_args]
        chn_range = list(range(self.num_channels)) if self.sep_chn_los else [0]
        for j in chn_range:
            im_los = self.obs.calc_los_image(num_simage, j)
            for idx in np.ndindex(self.fx_image_shape[1:]):
                if self.obs.valid_obs[j, num_simage, idx[0], idx[1]]:
                    fxres = self._fwdm_los(im_los[idx[0]][idx[1]], im_atm, im_aux)
                    if self.sep_chn_los:
                        fx_im[j, *idx] = fxres[j]
                    else:
                        fx_im[:, *idx] = fxres
        logging.log(15, f"Forward model: Image {num_simage} processed in {time.time()-tic:.1f} s.")
        return fx_im

    def calc_fwdm(self, atm, processes=1):
        fx = multiprocess(self._calc_fwdm_image, self.num_simages, [], processes, [atm, self.aux],
                          stack=True, numbers_only=True)
        fx = np.transpose(fx, axes=(1, 0, 2, 3))
        if np.isnan(fx).any():
            raise RuntimeError("Forward model: Nan's encountered in fx! Abort!")
        return fx

    def _get_qty_id(self, qty):
        if qty in self.grid.ret_qty:
            return (0, self.grid.ret_qty.index(qty))
        if qty in self.grid.aux_qty:
            return (1, self.grid.aux_qty.index(qty))
        else:
            raise ValueError("Forward model needs the quantity {qty}, but it is neither retrieved nor auxiliary!")


class Forward_model_temp_abs(Forward_model):
    def __init__(self, conf, const, grid, metadata, aux, rt_data, combine_images=False):
        super().__init__(conf, const, grid, metadata, aux, combine_images=combine_images)
        self.ver_id, self.t_id, self.o2_id = [self._get_qty_id(qty) for qty in ["VER", "T", "O2"]]
        if self.o2_id[0] == 0:
            raise ValueError("This forward model cannot retrieve O2 density, check your conf!")
        self.rt_data = rt_data
        self.startT = np.linspace(100, 600, 501)
        self.Tstep = self.startT[1] - self.startT[0]

    def _interp_T(self, vals, tables):
        ix = np.array(np.floor((vals - self.startT[0]) / self.Tstep), dtype=int)
        w0 = (self.startT[ix + 1] - vals) / self.Tstep
        w1 = (vals - self.startT[ix]) / self.Tstep
        return [w0 * tab[ix, :].T + w1 * tab[ix + 1, :].T for tab in tables]

    def _fwdm_jac_los(self, pos, atm, aux):
        grads = [None for x in self.grid.ret_qty]
        pathVER, pathT, patho2, pWeights = self.interp.interpolate(pos, [[atm, aux][idx[0]][idx[1]] for idx in
                                                                         [self.ver_id, self.t_id, self.o2_id]])
        sigmas, sigmas_pTgrad, emissions, emissions_pTgrad = self._interp_T(pathT, [self.rt_data[name] for name in
                                                             ["sigma", "sigma_grad", "emission", "emission_grad"]])
        exp_tau = np.exp(-(sigmas * patho2).cumsum(axis=1) * self.stepsize * 1e2) * (self.stepsize / 4 / np.pi * 1e6)
        del sigmas

        if not self.t_id[0]:  # if retrieving temperature
            grads[self.t_id[1]] = self.interp.grad_path2grid(self.rt_data["filters"] @
                                                             (exp_tau * emissions_pTgrad) * pathVER, pWeights)
        del emissions_pTgrad
        path_tau_em = exp_tau * emissions
        if not self.t_id[0]:  # if retrieving temperature
            grads[self.t_id[1]] = self.interp.grad_add(grads[self.t_id[1]],
                self.interp.grad_path2grid(self.rt_data["filters"] @
                (np.flip(np.cumsum(np.flip(path_tau_em * pathVER, axis=1), axis=1), axis=1) * sigmas_pTgrad) *
                patho2, pWeights), m2_factor=-1)
        del sigmas_pTgrad, patho2
        path_tau_em = self.rt_data["filters"] @ path_tau_em
        res = np.sum(path_tau_em * pathVER, axis=1)
        if not self.ver_id[0]:  # if retrieving VER
            grads[self.ver_id[1]] = self.interp.grad_path2grid(path_tau_em, pWeights)
        self.interp.postprocess_grads(grads)
        return res, grads

    def _fwdm_los(self, pos, atm, aux):
        pathVER, pathT, patho2, pWeights = self.interp.interpolate(pos, [[atm, aux][idx[0]][idx[1]] for idx in
                                                                         [self.ver_id, self.t_id, self.o2_id]])
        sigmas, emissions = self._interp_T(pathT, [self.rt_data[name] for name in ["sigma", "emission"]])
        exp_tau = np.exp(-(sigmas * patho2).cumsum(axis=1) * self.stepsize * 1e2) * (self.stepsize / 4 / np.pi * 1e6)
        del sigmas
        path_tau_em = self.rt_data["filters"] @ (exp_tau * emissions)
        return np.sum(path_tau_em * pathVER, axis=1)

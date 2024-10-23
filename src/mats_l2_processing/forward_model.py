from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
import time
import logging

import mats_l2_processing.interpolator as interpolator
from mats_l2_processing.util import multiprocess


class Forward_model(ABC):
    def __init__(self, conf, const, grid, metadata, aux, combine_images=False, debug_nan=False):
        self.grid = grid
        self.metadata = metadata
        self.combine_images = combine_images

        for i in range(len(aux)):
            assert aux[i].shape == tuple(grid.atm_shape[1:]), f"Auxiliary data variable {i} " +\
                "has shape {aux[i].shape}, but grid is of shape {grid.atm_shape[1:]}"
        self.aux = aux

        interp_type = conf.INTERPOLATOR
        ndims = len(grid.edges)
        if interp_type == 'LINEAR':
            if ndims == 1:
                self.interp = interpolator.Linear_interpolator_1D(grid)
            elif ndims == 3:
                self.interp = interpolator.Trilinear_interpolator_3D(grid)
            else:
                raise NotImplementedError(f"LINEAR interolator for {ndims} dimensions not implemented!")
        else:
            raise NotImplementedError(f"Unknown interpolator type {interp_type}!")

        self.channels = conf.CHANNELS
        self.valid_points = np.logical_and(self.grid.alt > conf.RET_ALT_RANGE[0] * 1e3,
                                           self.grid.alt < conf.RET_ALT_RANGE[1] * 1e3)
        self.valid_obs = np.logical_and(self.grid.TP_heights > conf.TP_ALT_RANGE[0] * 1e3,
                                        self.grid.TP_heights < conf.TP_ALT_RANGE[1] * 1e3)
        assert self.valid_obs.any(), "All observations flagged as invalid, hence forward model has nothing to do!"
        self.fx_image_shape = (len(self.channels), len(self.grid.columns), len(self.grid.rows))
        self.K_image_shape = (len(self.channels) * len(self.grid.columns) * len(self.grid.rows),
                              len(self.grid.ret_qty) * self.grid.npoints)
        self.fwdm_vars = const.NEEDED_DATA
        self.debug_nan = debug_nan  # Not implemented yet!

        self.stepsize = grid.stepsize

    @abstractmethod
    def _fwdm_los(self, pos, atm):
        pass

    @abstractmethod
    def _fwdm_jac_los(self, pos, atm):
        pass

    def _calc_fwdm_jac_image(self, image, atm):
        tic = time.time()
        im_los = self.grid.calc_los_image(image)
        fx_im = np.zeros(self.fx_image_shape)
        valid_obs_image = self.valid_obs[image['num_image'], ...]
        if self.combine_images:
            jac = sp.lil_array((self.K_image_shape[0] * image["num_images"], self.K_image_shape[1]))
            im_atm = atm.copy()
            jac_row_idx = len(self.channels) * len(self.grid.columns) * image['num_image']
        else:
            jac = sp.lil_array(*self.K_image_shape)
            im_atm = atm[image["num_image"], ...].copy()
            jac_row_idx = 0

        for idx in np.ndindex(self.fx_image_shape[1:]):
            if valid_obs_image[idx[0], idx[1]]:
                fx_im[:, idx[0], idx[1]], grads = self._fwdm_jac_los(im_los[idx[0]][idx[1]], im_atm)
                grad_stack = np.stack(grads, axis=0).reshape(len(self.grid.ret_qty), len(self.channels), -1)
                grad_stack[:, :, ~self.valid_points.reshape(-1)] = 0.0
                for j in range(len(self.channels)):
                    jac[j * self.grid.npoints + jac_row_idx, :] = grad_stack[:, j, :].reshape(-1)
            jac_row_idx += 1

        logging.log(15, f"Jacobian: Image {image['num_image']} processed in {time.time()-tic:.1f} s.")
        return jac, fx_im

    def _calc_fwdm_image(self, image, atm):
        tic = time.time()
        logging.info(len(atm))
        im_los = self.grid.calc_los_image(image)
        fx_im = np.zeros(self.fx_image_shape)
        im_atm = atm.copy() if self.combine_images else atm[image["num_image"], ...].copy()
        for idx in np.ndindex(self.fx_image_shape[1:]):
            if self.valid_obs[*idx]:
                fx_im[:, *idx] = self._fwdm_los(im_los[idx[0]][idx[1]], im_atm)
        logging.log(15, f"Forward model: Image {image['num_image']} processed in {time.time()-tic:.1f} s.")
        return fx_im

    def calc_fwdm(self, atm, processes=1):
        fx = multiprocess(self._calc_fwdm_image, self.metadata, self.fwdm_vars, processes, atm, stack=True)
        if np.isnan(fx).any():
            raise RuntimeError("Forward model: Nan's encountered in fx! Abort!")
        return fx

    def calc_fwdm_jac(self, atm, processes=1):
        res = multiprocess(self._calc_fwdm_jac_image, self.metadata, self.fwdm_vars, processes, atm,
                           unzip=not self.combine_images)

        stack_start = time.time()
        if self.combine_images:
            jac, fx = res.pop(0)
            fx = fx[:, np.newaxis, :, :]
            while len(res) > 0:
                im_jac, im_fx = res.pop(0)
                jac = jac + im_jac
                fx = np.concatenate([fx, im_fx[:, np.newaxis, :, :]], axis=1)
            del im_jac, im_fx
        else:
            jac, fx = res
            del res

        check_start = time.time()
        logging.log(15, f"Jacobian: Stacking took {check_start - stack_start:.1f} s")

        if np.isnan(fx).any():
            raise RuntimeError("Forward model: Nan's encountered in fx! Abort!")
        if np.isnan(jac.max()):
            raise RuntimeError("Jacobian: Nan's encountered in jacobian! Abort!")
        return jac, fx


class Forward_model_temp_abs(Forward_model):
    def __init__(self, conf, const, grid, metadata, o2, rt_data, combine_images=False):
        super().__init__(conf, const, grid, metadata, [o2], combine_images=combine_images)
        self.ver_idx = self.grid.ret_qty.index("VER")
        self.t_idx = self.grid.ret_qty.index("T")
        self.o2_idx = 0
        self.rt_data = rt_data
        self.startT = np.linspace(100, 600, 501)
        self.Tstep = self.startT[1] - self.startT[0]

    def _interp_T(self, vals, tables):
        ix = np.array(np.floor((vals - self.startT[0]) / self.Tstep), dtype=int)
        w0 = (self.startT[ix + 1] - vals) / self.Tstep
        w1 = (vals - self.startT[ix]) / self.Tstep
        return [w0 * tab[ix, :].T + w1 * tab[ix + 1, :].T for tab in tables]

    def _fwdm_jac_los(self, pos, atm):
        pathVER, pathT, patho2, pWeights = self.interp.interpolate(pos, [atm[self.ver_idx], atm[self.t_idx],
                                                                         self.aux[self.o2_idx]])
        sigmas, sigmas_pTgrad, emissions, emissions_pTgrad = self._interp_T(pathT, [self.rt_data[name] for name in
                                                             ["sigma", "sigma_grad", "emission", "emission_grad"]])
        exp_tau = np.exp(-(sigmas * patho2).cumsum(axis=1) * self.stepsize * 1e2) * (self.stepsize / 4 / np.pi * 1e6)
        del sigmas
        grad_Temps = self.interp.grad_path2grid(self.rt_data["filters"] @
                                                (exp_tau * emissions_pTgrad) * pathVER, pWeights)
        del emissions_pTgrad
        path_tau_em = exp_tau * emissions
        grad_Temps -= self.interp.grad_path2grid(self.rt_data["filters"] @ (np.flip(np.cumsum(np.flip(path_tau_em *
            pathVER, axis=1), axis=1), axis=1) * sigmas_pTgrad) * patho2, pWeights)
        del sigmas_pTgrad, patho2
        path_tau_em = self.rt_data["filters"] @ path_tau_em
        res = np.sum(path_tau_em * pathVER, axis=1)
        grad_VER = self.interp.grad_path2grid(path_tau_em, pWeights)
        return res, [grad_VER, grad_Temps]

    def _fwdm_los(self, pos, atm):
        pathVER, pathT, patho2, _ = self.interp.interpolate(pos, [atm[self.ver_idx], atm[self.t_idx],
                                                                  self.aux[self.o2_idx]])
        sigmas, emissions = self._interp_T(pathT, [self.rt_data[name] for name in ["sigma", "emission"]])
        exp_tau = np.exp(-(sigmas * patho2).cumsum(axis=1) * self.stepsize * 1e2) * (self.stepsize / 4 / np.pi * 1e6)
        del sigmas
        path_tau_em = self.rt_data["filters"] @ (exp_tau * emissions)
        return np.sum(path_tau_em * pathVER, axis=1)

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
        self.numobs = len(self.valid_obs.flatten())
        assert self.valid_obs.any(), "All observations flagged as invalid, hence forward model has nothing to do!"
        self.im_points = len(self.grid.columns) * len(self.grid.rows)
        self.fx_image_shape = (len(self.channels), len(self.grid.columns), len(self.grid.rows))
        self.K_image_shape = (len(self.channels) * self.im_points, len(self.grid.ret_qty) * self.grid.npoints)
        self.fwdm_vars = const.NEEDED_DATA
        self.debug_nan = debug_nan  # Not implemented yet!
        self.stepsize = grid.stepsize

    @abstractmethod
    def _fwdm_los(self, pos, atm, aux):
        pass

    @abstractmethod
    def _fwdm_jac_los(self, pos, atm, aux):
        pass

    def _calc_fwdm_jac_image(self, image, common_args):
        tic = time.time()
        im_los = self.grid.calc_los_image(image, [])
        fx_im = np.zeros(self.fx_image_shape)
        valid_obs_image = self.valid_obs[image['num_image'], ...]
        if self.combine_images:
            valid_points_stack = np.concatenate([self.valid_points.flatten() for qty in self.grid.ret_qty])
            jac = sp.lil_array((self.K_image_shape[0] * image["num_images"], self.K_image_shape[1]))
            im_atm, im_aux = [item.copy() for item in common_args]
            jac_row_idx = self.im_points * image['num_image']
            num_pixels = self.im_points * image['num_images']
        else:
            valid_points_stack = np.concatenate([self.valid_points[image["num_image"], :].flatten()
                                                 for qty in self.grid.ret_qty])
            jac = sp.lil_array((self.K_image_shape[0], self.K_image_shape[1]))
            im_atm, im_aux = [[qty[image["num_image"], ...].copy() for qty in dset] for dset in common_args]
            jac_row_idx, num_pixels = 0, self.im_points

        for idx in np.ndindex(self.fx_image_shape[1:]):
            if valid_obs_image[idx[0], idx[1]]:
                fx_im[:, idx[0], idx[1]], grads = self._fwdm_jac_los(im_los[idx[0]][idx[1]], im_atm, im_aux)
                grads = [grads[i] * scale for i, scale in enumerate(self.grid.scales)]
                grad_stack = np.stack(grads, axis=0).reshape(len(self.grid.ret_qty), len(self.channels), -1)
                # grad_stack[:, :, ~self.valid_points.reshape(-1)] = 0.0
                for j in range(len(self.channels)):
                    jac[j * num_pixels + jac_row_idx, :] = np.where(valid_points_stack, grad_stack[:, j, :].flatten(),
                                                                    0.0)
                #for j in range(2):
                #    print([jac[j * num_pixels + jac_row_idx, f] for f, b in enumerate(valid_points_stack) if b])
                #    print([jac[j * num_pixels + jac_row_idx, f] for f, b in enumerate(~valid_points_stack) if b])
                #breakpoint()

            jac_row_idx += 1

        # scale_vec = np.concatenate([sc * np.ones(self.grid.atm_size) for sc in self.grid.scales])
        # logging.info("{scale_vec.shape}, {jac.shape}")
        # jac = jac.multiply(scale_vec[np.newaxis, :])
        # print(type(jac))
        logging.log(15, f"Jacobian: Image {image['num_image']} processed in {time.time()-tic:.1f} s.")
        return jac.tocsr(), fx_im

    def _calc_fwdm_image(self, image, common_args):
        tic = time.time()
        im_los = self.grid.calc_los_image(image, [])
        fx_im = np.zeros(self.fx_image_shape)
        if self.combine_images:
            im_atm, im_aux = [item.copy() for item in common_args]
        else:
            im_atm, im_aux = [[qty[image["num_image"], ...].copy() for qty in dset] for dset in common_args]
        for idx in np.ndindex(self.fx_image_shape[1:]):
            if self.valid_obs[image["num_image"], idx[0], idx[1]]:
                fx_im[:, *idx] = self._fwdm_los(im_los[idx[0]][idx[1]], im_atm, im_aux)
        logging.log(15, f"Forward model: Image {image['num_image']} processed in {time.time()-tic:.1f} s.")
        return fx_im

    def calc_fwdm(self, atm, processes=1):
        fx = multiprocess(self._calc_fwdm_image, self.metadata, self.fwdm_vars, processes, [atm, self.aux], stack=True)
        fx = np.transpose(fx, axes=(1, 0, 2, 3))
        if np.isnan(fx).any():
            raise RuntimeError("Forward model: Nan's encountered in fx! Abort!")
        return fx

    def calc_fwdm_jac(self, atm, processes=1):
        res = multiprocess(self._calc_fwdm_jac_image, self.metadata, self.fwdm_vars, processes, [atm, self.aux],
                           unzip=False)

        stack_start = time.time()
        if self.combine_images:
            jac, fx = res.pop(0)
            fx = fx[:, np.newaxis, :, :]
            while len(res) > 0:
                im_jac, im_fx = res.pop(0)
                jac = jac + im_jac
                fx = np.concatenate([fx, im_fx[:, np.newaxis, :, :]], axis=1)
        else:
            jac, fx = [], []
            while len(res) > 0:
                im_jac, im_fx = res.pop(0)
                jac.append(im_jac)
                fx.append(im_fx)
        del im_jac, im_fx

        check_start = time.time()
        logging.log(15, f"Jacobian: Stacking took {check_start - stack_start:.1f} s")
        if np.isnan(fx).any():
            raise RuntimeError("Forward model: Nan's encountered in fx! Abort!")
        # if np.isnan(jac.max()):  TODO: find an effcient implementation for this!
        #     raise RuntimeError("Jacobian: Nan's encountered in jacobian! Abort!")
        return jac, fx

    def prepare_obs(self, conf, data, var):
        obs = np.empty((len(self.channels), len(self.grid.img_time), len(self.grid.columns), len(self.grid.rows)))
        idxs = np.meshgrid(self.grid.rows, self.grid.columns, indexing='ij')
        for i, chn in enumerate(self.channels):
            obs[i, ...] = conf.NCDF_OBS_FACTOR * np.transpose(data[var[chn]][:, idxs[0], idxs[1]], axes=(0, 2, 1))
        return obs

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
        # self.ver_idx = self.grid.ret_qty.index("VER")
        # self.t_idx = self.grid.ret_qty.index("T")
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
        # pathVER, pathT, patho2, pWeights = self.interp.interpolate(pos, [atm[self.ver_idx], atm[self.t_idx],
        #                                                                 self.aux[self.o2_idx]])
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
            grads[self.t_id[1]] -= self.interp.grad_path2grid(self.rt_data["filters"] @
                (np.flip(np.cumsum(np.flip(path_tau_em * pathVER, axis=1), axis=1), axis=1) * sigmas_pTgrad) *
                patho2, pWeights)
        del sigmas_pTgrad, patho2
        path_tau_em = self.rt_data["filters"] @ path_tau_em
        res = np.sum(path_tau_em * pathVER, axis=1)
        if not self.ver_id[0]:  # if retrieving VER
            grads[self.ver_id[1]] = self.interp.grad_path2grid(path_tau_em, pWeights)
        return res, grads

    def _fwdm_los(self, pos, atm, aux):
        # pathVER, pathT, patho2, _ = self.interp.interpolate(pos, [atm[self.ver_idx], atm[self.t_idx],
        #                                                          self.aux[self.o2_idx]])
        pathVER, pathT, patho2, pWeights = self.interp.interpolate(pos, [[atm, aux][idx[0]][idx[1]] for idx in
                                                                         [self.ver_id, self.t_id, self.o2_id]])
        sigmas, emissions = self._interp_T(pathT, [self.rt_data[name] for name in ["sigma", "emission"]])
        exp_tau = np.exp(-(sigmas * patho2).cumsum(axis=1) * self.stepsize * 1e2) * (self.stepsize / 4 / np.pi * 1e6)
        del sigmas
        path_tau_em = self.rt_data["filters"] @ (exp_tau * emissions)
        return np.sum(path_tau_em * pathVER, axis=1)

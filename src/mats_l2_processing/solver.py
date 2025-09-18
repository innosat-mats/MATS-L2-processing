from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
from itertools import repeat

from sparse_dot_mkl import dot_product_mkl as mdot
import time
import logging
import mats_l2_processing.numerical_methods as num


class Solver(ABC):
    def __init__(self, fwdm, obs, conf, Sa_inv, Se_inv, atm_apr, Sa_terms):
        self.fwdm = fwdm
        # tph = fwdm.grid.TP_heights
        self.y = obs.flatten()
        self.y_ar = np.where(fwdm.valid_obs[np.newaxis, ...], obs, 0.0).flatten()
        self.Sa_inv = Sa_inv
        self.Se_inv = Se_inv
        self.xa = self.fwdm.grid.atm2vec(atm_apr)
        self.Sa_terms = {"Sa": self.Sa_inv} if Sa_terms is None else Sa_terms

    # @abstractmethod
    # def solve(self, nproc):
    #    pass

    def contributions(self, x, xa, y, fx):
        assert y.shape == fx.shape
        yr = y - fx
        y_se = yr.T @ self.Se_inv
        for i, chn in enumerate(self.fwdm.channels):
            id0, id1 = i * self.fwdm.numobs, (i + 1) * self.fwdm.numobs
            logging.info(f"Contributions: {chn} {np.dot(y_se[id0:id1], yr[id0:id1]):.2e}")

        assert x.shape == xa.shape
        xr = x - xa
        numpoints = self.fwdm.grid.npoints
        logging.info(f"{self.fwdm.grid.atm_shape}, {numpoints}, {len(x)}")
        for term_name, term in self.Sa_terms.items():
            xt = (xr.T @ term)
            for i, qty in enumerate(self.fwdm.grid.ret_qty):
                id0, id1, weight = i * numpoints, (i + 1) * numpoints, self.fwdm.grid.scales[i]
                logging.info(f"Contributions: {qty} {term_name} {np.dot(xt[id0:id1], xr[id0:id1]):.2e}")


class Linear_solver(Solver):
    def __init__(self, fwdm, obs, conf, Sa_inv, Se_inv, atm_apr, fname=None, Sa_terms=None, atm_init=None):
        super().__init__(fwdm, obs, conf, Sa_inv, Se_inv, atm_apr, Sa_terms)
        self.prefix = fname
        self.atm_init = atm_init
        self.fname = f"{fname}_L2_lin.nc"
        self.conf = conf
        req_obs_shape = (len(fwdm.channels), len(fwdm.grid.img_time), len(fwdm.grid.columns), len(fwdm.grid.rows))
        assert obs.shape == req_obs_shape, f"obs must be of shape {req_obs_shape}, but got {obs.shape}!"
        self.obs_shape = req_obs_shape

    def solve(self, nproc, jac=None, fx=None):
        if self.prefix is not None:
            self.fwdm.grid.write_grid_ncdf(self.fname)
            self.fwdm.grid.write_atm_ncdf(self.fname, self.fwdm.grid.vec2atm(self.xa), atm_suffix="_apr",
                                          atm_suffix_long=", a priori")

        atm_init = self.fwdm.grid.vec2atm(self.xa) if self.atm_init is None else self.atm_init
        if jac is None:
            jac, fx = self.fwdm.calc_fwdm_jac(atm_init, nproc)
        # elif fx is None:
        #    fx = self.fwdm.calc_fwdm(atm_init, nproc)

        if self.prefix is not None:
            if self.fwdm.grid.combine_images:
                fx_f = fx.reshape(self.obs_shape)
            else:
                fx_f = np.concatenate([fx_im[:, np.newaxis, :, :] for fx_im in fx], axis=1)
            self.fwdm.grid.write_obs_ncdf(self.fname, fx_f, self.fwdm.channels, obs_suffix="_sim_apr",
                                          obs_suffix_long="forward model simulation from a priori")
            del fx_f
            self.fwdm.grid.write_obs_ncdf(self.fname, self.y.reshape(self.obs_shape), self.fwdm.channels,
                                          obs_suffix_long=", MATS observation")

        xpr = np.zeros_like(self.xa) if self.atm_init is None else self.fwdm.grid.atm2vec(atm_init) - self.xa
        if self.fwdm.grid.combine_images:
            sol = self._mkl_step((xpr, jac, fx.flatten(), self.y_ar), [])
        else:
            yari = self.y_ar.reshape(self.obs_shape)
            im_args = [(self.fwdm.grid.get_imvec(xpr, i), jac[i], fx[i].flatten(), yari[:, i, :, :].flatten())
                       for i in range(self.fwdm.grid.atm_shape[1])]
            del yari
            with Pool(processes=nproc) as pool:
                sols = pool.starmap(self._mkl_step, zip(im_args, repeat([])))
            sol = self.fwdm.grid.imvecs2vec(sols)
        atm_res = self.fwdm.grid.vec2atm(sol)

        if self.prefix is not None:
            self.fwdm.grid.write_atm_ncdf(self.fname, atm_res, atm_suffix="", atm_suffix_long=", final result")
        return atm_res

    def _mkl_step(self, im_args, common_args):
        xpr, K, fx, y_ar = im_args
        b = mdot(self.Sa_inv, xpr) +\
            mdot(K.T, self.Se_inv @ num.nannorm((fx - y_ar), "fx", abort=(not self.fwdm.debug_nan)))
        A = num.LM_mkl_sparse_matrix(K, self.Se_inv, self.Sa_inv, lm=0)
        dx, resid_norm, iters = num.cg_solve(A, b, x_init=np.zeros_like(xpr), atol=self.conf.CG_ATOL,
                                             rtol=self.conf.CG_RTOL, maxiter=self.conf.CG_MAX_STEPS)
        return xpr - dx


class Lavenberg_marquardt_solver(Solver):
    def __init__(self, fwdm, obs, conf, Sa_inv, Se_inv, atm_apr, fname,
                 save_jac=False, load_jac=False, Sa_terms=None, init_ncdf=False):
        super().__init__(fwdm, obs, conf, Sa_inv, Se_inv, atm_apr, Sa_terms)
        self.prefix = fname
        self.save_jac = save_jac
        self.load_jac = load_jac
        self.atm_apr = atm_apr
        self.conf = conf
        self.init_ncdf = init_ncdf

        self.fname = f"{fname}_L2.nc"
        req_obs_shape = (len(fwdm.channels), len(fwdm.grid.img_time), len(fwdm.grid.columns), len(fwdm.grid.rows))
        assert obs.shape == req_obs_shape, f"obs must be of shape {req_obs_shape}, but got {obs.shape}!"

    def _L2_write_init(self, fx):
        atts = {"ret_min_height": self.conf.RET_ALT_RANGE[0] * 1e3,
                "ret_max_height": self.conf.RET_ALT_RANGE[1] * 1e3,
                "TP_min_height": self.conf.TP_ALT_RANGE[0] * 1e3,
                "TP_max_height": self.conf.TP_ALT_RANGE[1] * 1e3}
        if self.init_ncdf:
            self.fwdm.grid.write_grid_ncdf(self.fname, atts)
            self.fwdm.grid.write_atm_ncdf(self.fname, self.atm_apr, atm_suffix="_apr", atm_suffix_long=", a priori")
            atts = {}
        self.fwdm.grid.write_obs_ncdf(self.fname, self.y.reshape(fx.shape), self.fwdm.channels,
                                      obs_suffix_long=", MATS observation")
        self.fwdm.grid.write_obs_ncdf(self.fname, fx, self.fwdm.channels, obs_suffix="_sim_apr", attributes=atts,
                                      obs_suffix_long="forward model simulation based on a priori")

    def _L2_write_iter(self, atm, fx, it_id):
        # it_id is iteration number if positive, -1 if final value
        if it_id < 0:
            suffix = ""
            long_suffix = ", final result"
        else:
            suffix = f"_it_{it_id}"
            long_suffix = f", iteration {it_id}"

        self.fwdm.grid.write_atm_ncdf(self.fname, atm, atm_suffix=suffix, atm_suffix_long=long_suffix)
        logging.log(15, f"fx shape: {fx.shape}")
        self.fwdm.grid.write_obs_ncdf(self.fname, fx, self.fwdm.channels, obs_suffix=f"_sim{suffix}",
                                      obs_suffix_long=f", forward model simulation for {long_suffix}")

    def _mkl_LM_iteration(self, xp, K, fx, lm):
        tic = time.time()
        b = mdot(self.Sa_inv, xp - self.xa) +\
            mdot(K.T, self.Se_inv @ num.nannorm((fx - self.y_ar), "fx", abort=(not self.fwdm.debug_nan)))
        logging.log(15, "Solver: b calculated.")
        A = num.LM_mkl_sparse_matrix(K, self.Se_inv, self.Sa_inv, lm=lm)
        dx, resid_norm, iters = num.cg_solve(A, b, x_init=np.zeros_like(xp), atol=self.conf.CG_ATOL,
                                             rtol=self.conf.CG_RTOL, maxiter=self.conf.CG_MAX_STEPS)
        logging.log(15, f"Solver: calculated new atmosphere state in {time.time() - tic:.3f} s.")
        return xp - dx

    def solve(self, nproc):
        lm_start_time = time.time()

        # Initializing iteration variables
        xp = self.xa
        lm_par = self.conf.LM_PAR_0
        K = None

        #  Initial Jacobian
        if self.load_jac:
            K = sp.load_npz(f"{self.prefix}_1_K.npz")
            fxp = np.load(f"{self.prefix}_1_fxp.npz")
            logging.info("Jacobian loaded from file.")

            print(f"xa mean: {np.mean(self.xa)}, xp mean: {np.mean(xp)},  y_ar mean: {np.mean(self.y_ar)}," +
                  f"  fxp mean: {np.mean(fxp)}")
            print(f"Sa_inv trace: {self.Sa_inv.trace()}, Se_inv trace: {self.Se_inv.trace()}.")

            Temp_test_vec = np.concatenate([np.zeros(self.fwdm.grid.npoints), np.ones(self.fwdm.grid.npoints)])
            VER_test_vec = np.concatenate([np.ones(self.fwdm.grid.npoints), np.zeros(self.fwdm.grid.npoints)])
            obs_test_vec = np.ones(K.shape[0])

            logging.info(f"K shape: {K.shape}, T test shape: {Temp_test_vec.shape}, obs test shape: {obs_test_vec.shape}")
            logging.info(f"VER gradiant sum: {obs_test_vec.T @ K @ VER_test_vec}," +
                         f"T gradiant sum: {obs_test_vec.T @ K @ Temp_test_vec}")
        else:
            K, fxp = self.fwdm.calc_fwdm_jac(self.atm_apr, nproc)
        if self.save_jac:
            sp.save_npz(f"{self.prefix}_1_K.npz", K)
            with open(f"{self.prefix}_1_fxp.npz", 'wb') as f:
                np.save(f, fxp)

        self._L2_write_init(fxp)
        lver = int(np.floor(len(self.xa) / 2.0))
        print(f"xa_mean: {self.xa[:lver].mean()}, y_ar: {self.y_ar.mean()}")
        cf_p = num.cost_func(self.xa, self.xa, self.y_ar, fxp.flatten(), self.Se_inv, self.Sa_inv)
        logging.info(f"Initial misfit: {cf_p:.2e}")

        #  Main iteration
        for it in range(1, self.conf.LM_IT_MAX + 1):
            best_sol, sol = {}, {}
            if it > 1:
                del K  # Clear previous Jacobian
                K, fxp = self.fwdm.calc_fwdm_jac(self.fwdm.grid.vec2atm(xp), nproc)

            lms = [lm_par * float(self.conf.LM_FAC) ** x for x in range(-1, self.conf.LM_MAX_FACTS_PER_ITER + 1)]
            for j, lm in enumerate(lms):
                sol["lm"] = lm
                xhat = self._mkl_LM_iteration(xp, K, fxp.flatten(), lm)
                sol["sol"] = self.fwdm.grid.reset_invalid(xhat)
                sol["fx"] = self.fwdm.calc_fwdm(self.fwdm.grid.vec2atm(sol["sol"]), nproc)
                sol["cf"] = num.cost_func(sol["sol"], self.xa, self.y_ar, sol["fx"].flatten(), self.Se_inv,
                                          self.Sa_inv, debug_nan=self.fwdm.debug_nan)
                logging.info(f"Iteration {it}: derived solution with lambda={lm:.2e}, misfit {sol['cf']:.2e}.")
                self.contributions(sol["sol"], self.xa, self.y_ar, sol["fx"].flatten())

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
                        converged = (cf_ratio > self.conf.CONV_CRITERION)
                        final = converged or (it == self.conf.LM_IT_MAX)
                        if converged and (best_sol["lm"] == sol["lm"]):
                            continue
                        sol = {}
                        xp, fxp, lm_par, cf_p = best_sol["sol"], best_sol["fx"], best_sol["lm"], best_sol["cf"]
                        logging.info(f"Iteration {it}: accepted solution with lambda={lm_par:.2e}" +
                                     f", misfit {best_sol['cf']:.2e} ({cf_ratio:.2f} of previous)")
                        self._L2_write_iter(self.fwdm.grid.vec2atm(xp), fxp, -1 if final else it)
                        if not final:
                            break
                        logging.info("Convergence reached!" if converged else "Max. number of iterations reached!")
                        logging.info(f"Total LM run time: {time.time() - lm_start_time} s.")
                        return

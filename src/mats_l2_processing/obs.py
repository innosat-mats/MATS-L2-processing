from abc import ABC, abstractmethod
import numpy as np
# from mats_l1_processing.pointing import pix_deg
from mats_l2_processing.pointing import faster_heights, verify_channels
from mats_l2_processing.util import get_image, multiprocess, seconds2DT
from mats_l2_processing.io import append_gen_ncdf, read_L1_ncdf
# from mats_utils.geolocation.coordinates import col_heights
import logging


class Obs(ABC):
    def __init__(self, grid, metadata, conf, const, processes):
        verify_channels(conf, metadata)

        self.grid = grid
        self.metadata = metadata

        self.rows, self.columns = get_row_col(conf, metadata[0])
        self.channels = conf.CHANNELS
        self.sep_chn_los = conf.SEP_CHN_LOS

        self.TP_heights_vars = const.TP_VARS
        self.fwdm_vars = const.NEEDED_DATA
        # self.img_time = metadata["EXPDate_s"]
        self.num_simages = metadata[0]['size']

        self.TP_heights = self._calc_tp_heights(metadata, processes)
        self.valid_obs = np.logical_and(self.TP_heights > conf.TP_ALT_RANGE[0] * 1e3,
                                        self.TP_heights < conf.TP_ALT_RANGE[1] * 1e3)
        assert self.valid_obs.any(), "All observations flagged as invalid, nothing to retrieve!"

    def _calc_tp_heights_image(self, image, common_args):
        return faster_heights(image, self.grid.pointing, cols=self.columns, rows=self.rows).T

    def _calc_tp_heights(self, metadata, processes):
        heights = []
        idxs = range(len(metadata)) if self.sep_chn_los else [0]
        for i in idxs:
            heights.append(multiprocess(self._calc_tp_heights_image, metadata[i], self.TP_heights_vars,
                                        processes, [], stack=True))
        if self.sep_chn_los:
            return np.stack(heights, axis=0)
        else:
            shape = (len(self.channels),) + heights[0].shape
            return np.broadcast_to(heights[0][np.newaxis, ...], shape)

    def calc_los_image(self, num_simage, chn):
        return self.grid.calc_los_image(get_image(self.metadata[chn], num_simage, self.fwdm_vars), [])

    def prepare_obs_data(self, conf, ifiles):
        obs_data = np.empty((len(self.channels), len(self.grid.img_time), len(self.columns), len(self.rows)))
        idxs = np.meshgrid(self.rows, self.columns, indexing='ij')
        for i, chn in enumerate(self.channels):
            cvar = conf.OBS_SRC_VAR[chn]
            meta_idx = i if conf.SEP_CHN_LOS else 0
            tlim = [seconds2DT(self.metadata[meta_idx]["EXPDate_s"][idx] + offset)
                    for idx, offset in [(0, -0.1), (-1, 0.1)]]
            data = read_L1_ncdf(ifiles[meta_idx], start_time=tlim[0], stop_time=tlim[1], var=[cvar])
            obs_data[i, ...] = conf.NCDF_OBS_FACTOR * np.transpose(data[cvar][:, idxs[0], idxs[1]], axes=(0, 2, 1))
        return obs_data

    def write_obs_ncdf(self, fname, obs_data, obs_suffix="", obs_suffix_long="", attributes={}):
        ncvars = {}
        dims = ("img_time", "img_col", "img_row")
        for i, chn in enumerate(self.channels):
            ncvars[f"{chn}{obs_suffix}"] = (f"{self.grid.ncpar[chn][0]}{obs_suffix_long}", self.grid.ncpar[chn][1],
                                            obs_data[i, :, :, :], dims)
        append_gen_ncdf(fname, ncvars, attributes=attributes)

    def write_TPheights_ncdf(self, fname):
        ncvars = {}
        dims = ("img_time", "img_col", "img_row")
        TP_channels = self.channels if self.sep_chn_los else [self.channels[0]]
        for i, chn in enumerate(TP_channels):
            ncvars[f"TPheight_{chn}"] = (f"Tangent point height, {chn}", "meter", self.TP_heights[i, ...], dims)
        append_gen_ncdf(fname, ncvars)


def get_row_col(conf, metadata):
    row_range = (0, metadata["NROW"][0]) if conf.ROW_RANGE[0] < 0 else conf.ROW_RANGE
    columns, rows = [np.arange(r[0], r[1], 1) for r in [conf.COL_RANGE, row_range]]
    return rows, columns

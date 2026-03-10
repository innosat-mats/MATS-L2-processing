from abc import ABC, abstractmethod
import numpy as np
from itertools import product
from scipy.sparse import coo_matrix


class Interpolator(ABC):
    def __init__(self, grid):
        self.edges = grid.edges
        self.shape = [len(c) for c in grid.centers]
        self.numpoints = np.prod(self.shape)
        self.coord_factors = [np.prod(self.shape[j + 1:]) for j in range(len(self.shape) - 1)] + [1]
        # self.shape = grid.atm_shape[1:]

    @abstractmethod
    def interpolate(self, pos, data):
        pass

    def grad_path2grid_sp(self, pathGrad, pWeights):
        numw = len(pWeights[1])
        shape = (1, self.numpoints)

        cols = np.empty((pathGrad.shape[0], numw), dtype=int)
        vals = np.empty((pathGrad.shape[0], numw))
        rows = np.stack([np.full((numw,), i) for i in range(pathGrad.shape[0])], axis=0)

        it = np.nditer(pWeights[1], flags=["c_index", "multi_index"])
        for w in it:
            coord = pWeights[0][it.multi_index[0], it.multi_index[1], :]
            cols[:, it.index] = sum([coord[c] * self.coord_factors[c] for c in range(pWeights.shape[2])])
            vals[:, it.index] = w * pathGrad[:, it.multi_index[0]]

        res = []
        for c in range(pathGrad.shape[0]):
            res.append(coo_matrix((vals[c, :], (rows[c, :], cols[c, :])), shape=shape))
            res[-1].sum_duplicates()
        return res


class Trilinear_interpolator_3D(Interpolator):
    def __init__(self, grid):
        super().__init__(grid)

    def grad_path2grid(self, pathGrad, pWeights):
        numw = pWeights[1].shape[0] * pWeights[1].shape[1]
        shape = (1, self.numpoints)

        vals = np.empty((pathGrad.shape[0], numw))
        rows = np.zeros(numw, dtype=int)
        cols = np.empty(numw, dtype=int)

        it = np.nditer(pWeights[1], flags=["c_index", "multi_index"])
        for w in it:
            coord = pWeights[0][it.multi_index[0], it.multi_index[1], :]
            cols[it.index] = sum([coord[c] * self.coord_factors[c] for c in range(pWeights[0].shape[2])])
            vals[:, it.index] = w * pathGrad[:, it.multi_index[0]]

        res = []
        for c in range(pathGrad.shape[0]):
            res.append(coo_matrix((vals[c, :], (rows, cols)), shape=shape))
            # res[-1].sum_duplicates()
        return res

    def grad_path2grid_dense(self, pathGrad, pWeights):
        res = np.zeros((pathGrad.shape[0], *self.shape))
        it = np.nditer(pWeights[1], flags=["multi_index"])
        for w in it:
            coord = pWeights[0][it.multi_index[0], it.multi_index[1], :]
            res[:, *coord] += w * pathGrad[:, it.multi_index[0]]
        return res

    def interpolate(self, pos, data):
        num_pos = pos.shape[0]
        coords0 = np.stack([np.searchsorted(self.edges[i], pos[:, i], sorter=None) - 1 for i in range(3)], axis=-1)

        coords, dists, iw = np.zeros((num_pos, 8, 3), dtype=int), np.zeros((num_pos, 3, 2)), np.zeros((num_pos, 8))
        dists[:, :, 1] = np.stack([pos[:, i] - self.edges[i][coords0[:, i]] for i in range(3)], axis=-1)
        dists[:, :, 0] = np.stack([self.edges[i][coords0[:, i] + 1] - pos[:, i] for i in range(3)], axis=-1)

        idata = np.stack(data, axis=0)
        res = np.zeros((len(data), num_pos))

        norms = np.prod(dists[:, :, 0] + dists[:, :, 1], axis=1)
        for i, idx in enumerate(product([0, 1], repeat=3)):
            for j in range(3):
                coords[:, i, j] = coords0[:, j] + idx[j]
            iw[:, i] = dists[:, 0, idx[0]] * dists[:, 1, idx[1]] * dists[:, 2, idx[2]]
            res += iw[:, i] * idata[:, coords[:, i, 0], coords[:, i, 1], coords[:, i, 2]]
        res /= norms[np.newaxis, :]
        return *[res[i, :] for i in range(len(data))], (coords, iw / norms[:, np.newaxis])


class Linear_interpolator_1D(Interpolator):
    def __init__(self, grid):
        super().__init__(grid)

    def grad_path2grid(self, pathGrad, pWeights):
        res = np.zeros((pathGrad.shape[0], *self.shape))
        it = np.nditer(pWeights[1], flags=["multi_index"])
        for w in it:
            coord = pWeights[0][it.multi_index[0], it.multi_index[1]]
            res[:, coord] += w * pathGrad[:, it.multi_index[0]]
        return res

    def interpolate(self, pos, data):
        num_pos = len(pos)
        coords, iw, res = np.empty((num_pos, 2), dtype=int), np.zeros((num_pos, 2)), np.zeros((len(data), num_pos))
        coords[:, 1] = np.searchsorted(self.edges[0], pos, sorter=None)
        coords[:, 0] = coords[:, 1] - 1
        iw[:, 1] = pos - self.edges[0][coords[:, 0]]
        iw[:, 0] = self.edges[0][coords[:, 1]] - pos

        for j, dat in enumerate(data):
            res[j, :] += iw[:, 0] * data[j][coords[:, 0]] + iw[:, 1] * data[j][coords[:, 1]]
        dists = self.edges[0][coords[:, 1]] - self.edges[0][coords[:, 0]]
        res /= dists[np.newaxis, :]
        return *[res[i, :] for i in range(len(data))], (coords, iw / dists[:, np.newaxis])

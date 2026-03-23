from abc import ABC, abstractmethod
import numpy as np
from itertools import product, chain
from scipy.sparse import coo_matrix


class Interpolator(ABC):
    def __init__(self, grid):
        self.points = grid.points
        self.shape = [len(c) for c in self.points]
        self.numpoints = np.prod(self.shape)
        self.coord_factors = [np.prod(self.shape[j + 1:]) for j in range(len(self.shape) - 1)] + [1]
        # self.shape = grid.atm_shape[1:]

        self.grad_add = None
        self.postprocess_grads = None

    @abstractmethod
    def interpolate(self, pos, data):
        pass

    @abstractmethod
    def grad_path2grid(self, pathGrad, pWeights):
        pass

    @staticmethod
    def grad_add_dense(ml1, ml2, m2_factor=1.0):
        return ml1 + m2_factor * ml2

    @staticmethod
    def grad_add_sparse(ml1, ml2, m2_factor=1.0):
        return [coo_matrix((np.concatenate([m1.data, m2_factor * m2.data]),
                           (np.concatenate([m1.row, m2.row]), np.concatenate([m1.col, m2.col]))),
                           shape=m1.shape) for m1, m2 in zip(ml1, ml2)]

    @staticmethod
    def postprocess_grads_dense(grads):
        for grad in chain(*grads):
            grad.flatten()

    @staticmethod
    def postprocess_grads_sparse(grads):
        for grad in chain(*grads):
            grad.sum_duplicates()


class Trilinear_interpolator_3D(Interpolator):
    def __init__(self, grid):
        super().__init__(grid)
        self.grad_add = self.grad_add_sparse
        self.postprocess_grads = self.postprocess_grads_sparse

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

        return [coo_matrix((vals[c, :], (rows, cols)), shape=shape) for c in range(pathGrad.shape[0])]

    def grad_path2grid_dense(self, pathGrad, pWeights):
        res = np.zeros((pathGrad.shape[0], *self.shape))
        it = np.nditer(pWeights[1], flags=["multi_index"])
        for w in it:
            coord = pWeights[0][it.multi_index[0], it.multi_index[1], :]
            res[:, *coord] += w * pathGrad[:, it.multi_index[0]]
        return res

    def interpolate(self, pos, data):
        num_pos = pos.shape[0]
        coords0 = np.stack([np.searchsorted(self.points[i], pos[:, i], sorter=None) - 1 for i in range(3)], axis=-1)

        coords, dists, iw = np.zeros((num_pos, 8, 3), dtype=int), np.zeros((num_pos, 3, 2)), np.zeros((num_pos, 8))
        dists[:, :, 1] = np.stack([pos[:, i] - self.points[i][coords0[:, i]] for i in range(3)], axis=-1)
        dists[:, :, 0] = np.stack([self.points[i][coords0[:, i] + 1] - pos[:, i] for i in range(3)], axis=-1)

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
        self.grad_add = self.grad_add_dense
        self.postprocess_grads = self.postprocess_grads_dense

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
        coords[:, 1] = np.searchsorted(self.points[0], pos, sorter=None)
        coords[:, 0] = coords[:, 1] - 1
        iw[:, 1] = pos - self.points[0][coords[:, 0]]
        iw[:, 0] = self.points[0][coords[:, 1]] - pos

        for j, dat in enumerate(data):
            res[j, :] += iw[:, 0] * data[j][coords[:, 0]] + iw[:, 1] * data[j][coords[:, 1]]
        dists = self.points[0][coords[:, 1]] - self.points[0][coords[:, 0]]
        res /= dists[np.newaxis, :]
        return *[res[i, :] for i in range(len(data))], (coords, iw / dists[:, np.newaxis])

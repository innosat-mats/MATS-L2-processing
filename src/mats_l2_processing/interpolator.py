from abc import ABC, abstractmethod
import numpy as np
from itertools import product


class Interpolator(ABC):
    def __init__(self, grid):
        self.edges = grid.edges
        self.shape = grid.atm_shape[1:]

    @abstractmethod
    def interpolate(self, pos, data):
        pass

    def grad_path2grid(self, pathGrad, pWeights):
        res = np.zeros((pathGrad.shape[0], *self.shape))
        it = np.nditer(pWeights[1], flags=["multi_index"])
        for w in it:
            coord = pWeights[0][it.multi_index[0], it.multi_index[1], :]
            res[:, *coord] += w * pathGrad[:, it.multi_index[0]]
        return res


class Trilinear_interpolator_3D(Interpolator):
    def __init__(self, grid):
        super().__init__(grid)

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

    def interpolate(self, pos, data):
        num_pos = len(pos)
        coords, iw, res = np.empty((num_pos, 2), dtype=int), np.zeros((num_pos, 2)), np.zeros((len(data), num_pos))
        coords[:, 1] = np.searchsorted(self.edges, pos, sorter=None)
        coords[:, 0] = coords[:, 1] - 1
        iw[:, 1] = pos - self.edges[coords[:, 0]]
        iw[:, 0] = self.edges[coords[:, 1]] - pos

        for j, dat in enumerate(data):
            res[j, :] += iw[:, 0] * data[j][coords[:, 0]] + iw[:, 1] * data[j][coords[:, 1]]
        dists = self.edges[coords[:, 1]] - self.edges[coords[:, 0]]
        res /= dists[np.newaxis, :]
        return *[res[i, :] for i in range(len(data))], (coords, iw / dists[:, np.newaxis])

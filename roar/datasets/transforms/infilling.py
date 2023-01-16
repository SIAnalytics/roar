import numpy as np
from mmcv.transforms import BaseTransform
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from mmcls.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class NoisyLinearImputer(BaseTransform):
    """Implementation of RemOve And Debias.

        <https://proceedings.mlr.press/v162/rong22a.html> _.

    It is modified from
        https://github.com/tleemann/road_evaluation/blob/main/road/imputations.py  # noqa

    Args:
        noise (float): magnitude of noise to add (absolute, set to 0 for no noise)
        weighting (list):
    """
    neighbors_weights = [((1, 1), 1 / 12), ((0, 1), 1 / 6), ((-1, 1), 1 / 12),
                         ((1, -1), 1 / 12), ((0, -1), 1 / 6),
                         ((-1, -1), 1 / 12), ((1, 0), 1 / 6), ((-1, 0), 1 / 6)]

    def __init__(self, noise=0.01):
        self.noise = noise

    def add_offset_to_indices(self, indices, offset, mask_shape):
        """Add the corresponding offset to the indices.

        Return new indices plus a valid bit-vector.
        """
        cord1 = indices % mask_shape[1]
        cord0 = indices // mask_shape[1]
        cord0 += offset[0]
        cord1 += offset[1]

        valid = ((cord0 < 0) | (cord1 < 0) | (cord0 >= mask_shape[0]) |
                 (cord1 >= mask_shape[1]))
        return ~valid, indices + offset[0] * mask_shape[1] + offset[1]

    def setup_sparse_system(self, img, mask, neighbors_weights):
        """Vectorized version to set up the equation system.

        mask: (H, W)-tensor of missing pixels.
        Image: (H, W, C)-tensor of all values.
        Return (N,N)-System matrix whti shape (N,C)
            Right hand side for each of the C channels.
        """
        maskflt = mask.flatten()
        imgflat = img.reshape((img.shape[0], -1))
        indices = np.argwhere(maskflt == 0).flatten()
        coords_to_vidx = np.zeros(len(maskflt), dtype=int)
        coords_to_vidx[indices] = np.arange(len(indices))  # lookup_indices =

        numEquations = len(indices)
        A = lil_matrix((numEquations, numEquations))  # System matrix
        b = np.zeros((numEquations, img.shape[0]))
        sum_neighbors = np.ones(numEquations)  # Sum of weights assigned

        for n in neighbors_weights:
            offset, weight = n[0], n[1]
            # Sum of the neighbors.
            # Take out outliers
            valid, new_coords = self.add_offset_to_indices(
                indices, offset, mask.shape)

            valid_coords = new_coords[valid]
            valid_ids = np.argwhere(valid == 1).flatten()

            # Add values to the right hand-side
            has_values_coords = valid_coords[maskflt[valid_coords] > 0.5]
            has_values_ids = valid_ids[maskflt[valid_coords] > 0.5]

            b[has_values_ids, :] -= weight * imgflat[:, has_values_coords].T

            # Add weights to the system (left hand side)
            has_no_values = valid_coords[
                maskflt[valid_coords] < 0.5]  # Find coordinates in the system.
            variable_ids = coords_to_vidx[has_no_values]
            has_no_values_ids = valid_ids[maskflt[valid_coords] < 0.5]

            A[has_no_values_ids, variable_ids] = weight

            # Reduce weight for invalid
            sum_neighbors[np.argwhere(
                valid == 0).flatten()] = sum_neighbors[np.argwhere(
                    valid == 0).flatten()] - weight

        A[np.arange(numEquations), np.arange(numEquations)] = -sum_neighbors
        return A, b

    def transform(self, results):
        img = results['img'].transpose(2, 0, 1)
        mask = 1 - results['mask']
        imgflt = img.reshape(img.shape[0], -1)
        maskflt = mask.reshape(-1)
        indices_linear = np.argwhere(
            maskflt == 0).flatten()  # Indices that need to be imputed.
        # Set up sparse equation system, solve system.
        A, b = self.setup_sparse_system(img.copy(), mask.copy(),
                                        self.neighbors_weights)
        res = spsolve(csc_matrix(A), b)

        # Fill the values with the solution of the system.
        img_infill = imgflt.copy()
        img_infill[:, indices_linear] = res.T + self.noise * np.random.random(
            res.T.shape)
        img_infill = img_infill.reshape(img.shape)
        results['img'] = img_infill.transpose(1, 2, 0)
        return results

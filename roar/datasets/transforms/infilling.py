import numpy as np
from mmcv.transforms import BaseTransform
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from mmpretrain.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LinearImputation(BaseTransform):
    """Implementation of RemOve And Debias.

    It is modified from
        https://github.com/tleemann/road_evaluation/blob/main/road/imputations.py  # noqa

    Args:
        noise (float): magnitude of noise to add.
    """
    neighbors_weights = [
        ((0, 1), 1 / 6),
        ((1, 0), 1 / 6),
        ((-1, 0), 1 / 6),
        ((0, -1), 1 / 6),
        ((1, 1), 1 / 12),
        ((-1, 1), 1 / 12),
        ((1, -1), 1 / 12),
        ((-1, -1), 1 / 12),
    ]

    def __init__(self, noise=0.01):
        self.noise = noise

    def add_offset_to_indices(self, indices, offset, mask_shape):
        """Add the corresponding offset to the indices.

        Returns:
            tuple: a pair of ndarray
                - valid flag
                - new indices added a valid bit-vector
        """
        coord0, coord1 = np.divmod(indices, mask_shape[1])
        coord0 += offset[0]
        coord1 += offset[1]

        valid = np.logical_and(
            np.logical_and(0 <= coord0, coord0 < mask_shape[0]),
            np.logical_and(0 <= coord1, coord1 < mask_shape[1]))
        return valid, indices + offset[0] * mask_shape[1] + offset[1]

    def setup_sparse_system(self, img, mask, neighbors_weights):
        """Vectorized version to set up the equation system.

        Returns:
            tuple: a pair of ndarray
                - System matrix with with shape (N, N)
                - Right hand side for each of the C channels with shape (N, C)
            where N is `num_equations`
        """
        maskflt = mask.flatten()
        indices = np.argwhere(maskflt == 0).flatten()
        coords_to_vidx = np.zeros(len(maskflt), dtype=int)
        coords_to_vidx[indices] = np.arange(len(indices))

        num_equations = len(indices)
        A = lil_matrix((num_equations, num_equations))  # System matrix
        b = np.zeros((num_equations, img.shape[0]))
        sum_neighbors = np.ones(num_equations)  # Sum of weights assigned

        for n in neighbors_weights:
            offset, weight = n[0], n[1]
            # Sum of the neighbors, and take out outliers.
            valid, new_coords = self.add_offset_to_indices(
                indices, offset, mask.shape)
            valid_coords = new_coords[valid]
            valid_ids = np.argwhere(valid == 1).flatten()

            # Add values to the right hand-side
            has_values_coords = valid_coords[maskflt[valid_coords] > 0.5]
            has_values_ids = valid_ids[maskflt[valid_coords] > 0.5]

            imgflat = img.reshape((img.shape[0], -1))
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

        A[np.arange(num_equations), np.arange(num_equations)] = -sum_neighbors
        return A, b

    def transform(self, results):
        img = results['img'].transpose(2, 0, 1)
        mask = 1 - results['mask']

        # Set up sparse equation system, solve system.
        A, b = self.setup_sparse_system(img, mask, self.neighbors_weights)
        res = spsolve(csc_matrix(A), b)

        # Fill the values with the solution of the system.
        infilled = img.reshape(img.shape[0], -1)
        indices = np.argwhere(mask.reshape(-1) == 0).reshape(-1)
        infilled[:,
                 indices] = res.T + self.noise * np.random.random(res.T.shape)
        infilled = infilled.reshape(img.shape)
        results['img'] = infilled.transpose(1, 2, 0)

        return results

import numpy as np
import cupy as cp
from plyfile import PlyData

from utils.quaternion import quaternion_to_matrix


class GaussianModel:
    def __init__(self, mu_w, scales, quaternions, sigma_w, opacity, sh_coeffs):
        self.mu_w = mu_w
        self.scales = scales
        self.quaternions = quaternions
        self.sigma_w = sigma_w
        self.opacity = opacity
        self.sh_coeffs = sh_coeffs

    @classmethod
    def from_ply(cls, ply_path):
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        n_points = len(vertex.data)

        mu_w = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
        scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
        quaternion = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
        opacity = np.array(vertex['opacity'])

        sh_coeffs = np.empty((n_points, 3, 16), dtype=np.float32)
        sh_coeffs[:, 0, 0] = vertex['f_dc_0']
        sh_coeffs[:, 1, 0] = vertex['f_dc_1']
        sh_coeffs[:, 2, 0] = vertex['f_dc_2']

        for i in range(48 - 3):
            sh_coeffs[:, i // 15, i % 15 + 1] = vertex[f'f_rest_{i}']

        # preprocess
        mu_w = cp.array(mu_w, dtype=cp.float32)

        scales = cp.array(scales, dtype=cp.float32)
        scales = cp.exp(scales)

        quaternion = cp.array(quaternion, dtype=cp.float32)
        rotation_matrix = quaternion_to_matrix(quaternion)

        opacity = cp.array(opacity, dtype=cp.float32)
        opacity = 1.0 / (1.0 + cp.exp(-opacity))

        sh_coeffs = cp.array(sh_coeffs, dtype=cp.float32)
        sh_coeffs = sh_coeffs.reshape(-1, 3, 16)

        scale_squared = scales ** 2
        RS = rotation_matrix * scale_squared[:, None, :]
        sigma_w = RS @ rotation_matrix.transpose(0, 2, 1)

        return cls(mu_w, scales, quaternion, sigma_w, opacity, sh_coeffs)

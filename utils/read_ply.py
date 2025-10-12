from plyfile import PlyData
import numpy as np
import cupy as cp

plydata = PlyData.read('data/train/point_cloud/iteration_30000/point_cloud.ply')

vertex = plydata['vertex']

mu_w = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
rotations = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
opacity = np.array(vertex['opacity'])

all_names = ["f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)]
sh_coeffs = np.stack([vertex[name] for name in all_names], axis=1).reshape(-1, 16, 3)

sigma_w = np.array([scales[i] * np.eye(3) for i in range(scales.shape[0])])

scales = cp.array(scales, dtype=cp.float32)
rotations = cp.array(rotations, dtype=cp.float32)
opacity = cp.array(opacity, dtype=cp.float32)
sh_coeffs = cp.array(sh_coeffs, dtype=cp.float32)
mu_w = cp.array(mu_w, dtype=cp.float32)
sigma_w = cp.array(sigma_w, dtype=cp.float32)

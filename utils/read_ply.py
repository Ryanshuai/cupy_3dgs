from plyfile import PlyData
import numpy as np
import cupy as cp

from utils.quaternion import quaternion_to_matrix

# plydata = PlyData.read('data/train/point_cloud/iteration_30000/point_cloud.ply')
# plydata = PlyData.read('data/train/point_cloud/iteration_30000/point_cloud.ply')
plydata = PlyData.read('test/cropped_center_1of5.ply')
# plydata = PlyData.read('test/test_minimal.ply')

vertex = plydata['vertex']

mu_w = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
rotations = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
opacity = np.array(vertex['opacity'])

f_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)  # (N, 3)
f_rest = np.stack([vertex[f"f_rest_{i}"] for i in range(45)], axis=1)  # (N, 45)
f_rest = f_rest.reshape(-1, 3, 15)  # (N, 3, 15)
sh_coeffs = np.concatenate([f_dc[:, :, None], f_rest], axis=2)  # (N, 3, 16)

scales = cp.array(scales, dtype=cp.float32)
scales = cp.exp(scales)
rotations = cp.array(rotations, dtype=cp.float32)

R = quaternion_to_matrix(rotations)  # (N, 3, 3)
S = scales[:, :, None] * cp.eye(3)  # (N, 3, 3)
sigma_w = R @ S @ S.transpose(0, 2, 1) @ R.transpose(0, 2, 1)

opacity = cp.array(opacity, dtype=cp.float32)
opacity = 1.0 / (1.0 + cp.exp(-opacity))
sh_coeffs = cp.array(sh_coeffs, dtype=cp.float32)
mu_w = cp.array(mu_w, dtype=cp.float32)
sigma_w = cp.array(sigma_w, dtype=cp.float32)

print("-" * 50)

print(f"DC coeffs stats: min={sh_coeffs[:, 0, :].min():.3f}, max={sh_coeffs[:, 0, :].max():.3f}")
print(f"Sample DC (RGB): {sh_coeffs[0, 0, :]}")
print(f"Opacity stats: min={opacity.min():.3f}, max={opacity.max():.3f}")

print(f"\nScales stats: min={scales.min():.6f}, max={scales.max():.6f}, mean={scales.mean():.6f}")
print(f"Zero scales: {(scales == 0).sum()}")
print(f"Negative scales: {(scales < 0).sum()}")

# 检查协方差矩阵
det_sigma = cp.linalg.det(sigma_w)
print(f"\nCovariance determinant: min={det_sigma.min():.6e}, max={det_sigma.max():.6e}")
print(f"Non-positive det: {(det_sigma <= 0).sum()} / {len(det_sigma)}")
print(f"Very small det (<1e-10): {(det_sigma < 1e-10).sum()}")

# 检查协方差矩阵的特征值
eigenvalues = cp.linalg.eigvalsh(sigma_w)
print(f"\nEigenvalues: min={eigenvalues.min():.6e}, max={eigenvalues.max():.6e}")
print(f"Negative eigenvalues: {(eigenvalues < 0).sum()}")

print("-" * 30)
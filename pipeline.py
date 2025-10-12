import cupy as cp
import cv2

from utils.read_ply import mu_w, sigma_w, opacity, sh_coeffs
from utils.camera_settings import intrinsics_from_fov
from gs_core.transforms_points import view_transform_point, calculate_projection_matrix_from_fov, project_to_ndc, \
    ndc_to_screen
from gs_core.transforms_covariances import view_transform_covariance, calculate_intrinsic_jacobian, project_to_screen
from gs_core.spherical_harmonecs import eval_sh
from gs_core.rasterization import render

R_view = cp.eye(3, dtype=cp.float32)
t_view = cp.array([0, 0, -5], dtype=cp.float32)

fov_y = cp.deg2rad(45)
screen_w = 320
screen_h = 192
aspect = screen_w / screen_h
near = 0.2
far = 10

fx, fy, cx, cy = intrinsics_from_fov(fov_y, screen_w, screen_h)

mu_c = view_transform_point(mu_w, R_view, t_view)

P = calculate_projection_matrix_from_fov(fov_y, aspect, near, far)
mu_ndc = project_to_ndc(mu_c, P)

margin = 1.3
# https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0/cuda_rasterizer/auxiliary.h
frustum_cull = ((mu_c[..., 2] < -near) &  # z_c is negative in front of the camera
                (mu_ndc[:, 0] >= -margin) & (mu_ndc[:, 0] <= margin) &
                (mu_ndc[:, 1] >= -margin) & (mu_ndc[:, 1] <= margin))

mu_ndc = mu_ndc[frustum_cull]
mu_c = mu_c[frustum_cull]
sigma_w = sigma_w[frustum_cull]
opacity = opacity[frustum_cull]
sh_coeffs = sh_coeffs[frustum_cull]

mu_screen = ndc_to_screen(mu_ndc, screen_w, screen_h)

sigma_c = view_transform_covariance(sigma_w, R_view)
J = calculate_intrinsic_jacobian(mu_c[:, 0], mu_c[:, 1], mu_c[:, 2], fx, fy)
sigma_screen = project_to_screen(sigma_c, J)

z_sorted_indices = cp.argsort(-mu_c[:, 2])  # Front-to-Back

mu_c = mu_c[z_sorted_indices]
mu_screen = mu_screen[z_sorted_indices]
sigma_screen = sigma_screen[z_sorted_indices]
opacity = opacity[z_sorted_indices]
sh_coeffs = sh_coeffs[z_sorted_indices]

colors = eval_sh(sh_coeffs, -mu_c)

print(f"Rendering {mu_screen.shape[0]} Gaussians...")
image = render(mu_screen, sigma_screen, opacity, colors, screen_w, screen_h, tile_size=16)
image = cp.clip(image, 0, 1)
image = (image * 255).astype(cp.uint8)
image = cp.asnumpy(image)
cv2.imshow("image", image)
cv2.waitKey(0)

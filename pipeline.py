import cupy as cp

from utils.read_ply import mu_w, sigma_w
from utils.camera_settings import intrinsics_from_fov
from gs_core.transforms_points import view_transform_point, calculate_projection_matrix_from_fov, project_to_ndc, \
    ndc_to_screen
from gs_core.transforms_covariances import view_transform_covariance, calculate_intrinsic_jacobian, project_to_screen

R_view = cp.eye(3, dtype=cp.float32)
t_view = cp.array([0, 0, -5], dtype=cp.float32)

fov_y = cp.deg2rad(60)
screen_w = 1920
screen_h = 1080
aspect = screen_w / screen_h
near = 0.2
far = 1000

fx, fy, cx, cy = intrinsics_from_fov(fov_y, screen_w, screen_h)

mu_c = view_transform_point(mu_w, R_view, t_view)

P = calculate_projection_matrix_from_fov(fov_y, aspect, near, far)
mu_ndc = project_to_ndc(mu_c, P)

margin = 1.3
# https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0/cuda_rasterizer/auxiliary.h
frustum_cull = ((mu_c[..., 2] < -near) &  # z_c is negative in front of the camera
                (-margin <= mu_ndc[:, 0] <= margin) &
                (-margin <= mu_ndc[:, 1] <= margin))

mu_ndc = mu_ndc[frustum_cull]
mu_screen = ndc_to_screen(mu_ndc, screen_w, screen_h)

mu_c = mu_c[frustum_cull]
sigma_w = sigma_w[frustum_cull]
sigma_c = view_transform_covariance(sigma_w, R_view)
J = calculate_intrinsic_jacobian(mu_c[:, 0], mu_c[:, 1], mu_c[:, 2], fx, fy)
sigma_screen = project_to_screen(sigma_c, J)

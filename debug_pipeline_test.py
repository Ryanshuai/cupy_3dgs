import cupy as cp
from scipy.spatial.transform import Rotation
import cv2

from utils.read_ply import mu_w, sigma_w, opacity, sh_coeffs
from utils.camera_settings import intrinsics_from_fov
from gs_core.transforms_points import view_transform_point, calculate_projection_matrix_from_fov, project_to_ndc, \
    ndc_to_screen
from gs_core.transforms_covariances import view_transform_covariance, calculate_intrinsic_jacobian, project_to_screen
from gs_core.spherical_harmonecs import eval_sh
from gs_core.rasterization import render

# 1. 检查点云数据
print(f"Total Gaussians: {mu_w.shape[0]}")
print(f"Point cloud range: X[{mu_w[:, 0].min():.2f}, {mu_w[:, 0].max():.2f}], "
      f"Y[{mu_w[:, 1].min():.2f}, {mu_w[:, 1].max():.2f}], "
      f"Z[{mu_w[:, 2].min():.2f}, {mu_w[:, 2].max():.2f}]")

camera_position = cp.array([0, 0, -5], dtype=cp.float32)
camera_rotation = [0, 0, 0]

rx, ry, rz = 0, 0, 0
R_view = cp.asarray(Rotation.from_euler('zyx', [rz, ry, rx], degrees=True).as_matrix(), dtype=cp.float32)

t_view = cp.array([0, 0, 5], dtype=cp.float32)

# 在 pipeline.py 开头添加
print(f"Point cloud center: {mu_w.mean(axis=0)}")
print(f"Point cloud extent: {mu_w.max(axis=0) - mu_w.min(axis=0)}")
print(f"Camera position (world): {-t_view @ R_view}")  # 相机世界坐标

# 计算点云与相机的距离
cam_pos_world = -t_view @ R_view
distances = cp.linalg.norm(mu_w - cam_pos_world, axis=1)
print(f"Distance range: [{distances.min():.2f}, {distances.max():.2f}]")

fov_y = cp.deg2rad(60)
screen_w = 640
screen_h = 320

# screen_w = 1920
# screen_h = 1080
aspect = screen_w / screen_h
near = 0.2
far = 1000.0

fx, fy, cx, cy = intrinsics_from_fov(fov_y, screen_w, screen_h)

mu_c = view_transform_point(mu_w, R_view, t_view)

# 2. 检查相机空间深度
print(f"\nCamera space Z range: [{mu_c[:, 2].min():.2f}, {mu_c[:, 2].max():.2f}]")
print(f"Points in front of camera (z < 0): {(mu_c[:, 2] < 0).sum()}")

P = calculate_projection_matrix_from_fov(fov_y, aspect, near, far)
mu_ndc = project_to_ndc(mu_c, P)

margin = 1.3
frustum_cull = ((mu_c[..., 2] > near) &
                (mu_ndc[:, 0] >= -margin) & (mu_ndc[:, 0] <= margin) &
                (mu_ndc[:, 1] >= -margin) & (mu_ndc[:, 1] <= margin))

# 3. 检查视锥剔除后的点数
print(f"\nAfter frustum culling: {frustum_cull.sum()} / {len(frustum_cull)} Gaussians")

if frustum_cull.sum() == 0:
    print("\n❌ No Gaussians passed frustum culling!")
    print("Suggestions:")
    print("  - Adjust camera position (t_view)")
    print("  - Check if point cloud is centered at origin")
    print("  - Try: t_view = cp.array([0, 0, -(mu_w[:, 2].mean() + 5)])")
    exit()

mu_ndc = mu_ndc[frustum_cull]
mu_c = mu_c[frustum_cull]
sigma_w = sigma_w[frustum_cull]
opacity = opacity[frustum_cull]
sh_coeffs = sh_coeffs[frustum_cull]

mu_screen = ndc_to_screen(mu_ndc, screen_w, screen_h)

sigma_c = view_transform_covariance(sigma_w, R_view)
J = calculate_intrinsic_jacobian(mu_c[:, 0], mu_c[:, 1], mu_c[:, 2], fx, fy)
sigma_screen = project_to_screen(sigma_c, J)

z_sorted_indices = cp.argsort(-mu_c[:, 2])

mu_w_filtered = mu_w[frustum_cull]

mu_c = mu_c[z_sorted_indices]
mu_screen = mu_screen[z_sorted_indices]
sigma_screen = sigma_screen[z_sorted_indices]
opacity = opacity[z_sorted_indices]
sh_coeffs = sh_coeffs[z_sorted_indices]
mu_w_sorted = mu_w_filtered[z_sorted_indices]

print(f"Cov screen samples:\n{sigma_screen}")
# print(f"Cov screen determinants: {(sigma_screen[:, 0] * sigma_screen[:, 2] - sigma_screen[:, 1]**2)[:3]}")

print(f"Z depth after sort: min={mu_c[:, 2].min():.3f}, max={mu_c[:, 2].max():.3f}")
print(f"First 10 depths: {mu_c[:10, 2]}")  # 应该从小(远)到大(近)

camera_center_w = -t_view  # 相机在世界坐标系中的位置

directions = mu_w_sorted - camera_center_w
directions = directions / cp.linalg.norm(directions, axis=1, keepdims=True)
colors = eval_sh(sh_coeffs, directions)
colors = cp.clip(colors, 0, 1)

print(f"Raw SH output (before clip): min={colors.min():.3f}, max={colors.max():.3f}")
print(f"Sample colors:\n{colors[:5]}")

# 4. 检查颜色值
print(f"\nColor range: [{colors.min():.3f}, {colors.max():.3f}]")
print(f"Opacity range: [{opacity.min():.3f}, {opacity.max():.3f}]")
print(f"Activated opacity range: [{opacity.min():.3f}, {opacity.max():.3f}]")

print(
    f"Screen coords: X[{mu_screen[:, 0].min():.1f}, {mu_screen[:, 0].max():.1f}], Y[{mu_screen[:, 1].min():.1f}, {mu_screen[:, 1].max():.1f}]")
print(
    f"Out of bounds: {((mu_screen[:, 0] < 0) | (mu_screen[:, 0] > screen_w) | (mu_screen[:, 1] < 0) | (mu_screen[:, 1] > screen_h)).sum()}")

print(f"\nRendering {mu_screen.shape[0]} Gaussians...")
image = render(mu_screen, sigma_screen, opacity, colors, screen_w, screen_h, tile_size=16)

# 7. 检查渲染结果
print(f"\nRendered image range: [{image.min():.3f}, {image.max():.3f}]")
print(f"Non-zero pixels: {(image > 0).any(axis=2).sum()} / {screen_w * screen_h}")

image = cp.clip(image, 0, 1)
image = (image * 255).astype(cp.uint8)
image = cp.asnumpy(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow("image", image)
cv2.imwrite("output.png", image)
cv2.waitKey(0)

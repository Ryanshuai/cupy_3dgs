import cupy as cp
import cv2

from utils.camera_settings import Camera
from gs_core.gaussian_model import GaussianModel
from gs_core.transforms_points import view_transform_point, calculate_projection_matrix_from_fov, project_to_ndc, \
    ndc_to_screen
from gs_core.transforms_covariances import view_transform_covariance, calculate_intrinsic_jacobian, project_to_screen
from gs_core.spherical_harmonecs import eval_sh
from gs_core.rasterization import render

gm = GaussianModel.from_ply("test/test_diagonal.ply")
gm = GaussianModel.from_ply("test/cropped_center_1of5.ply")

camera = Camera(
    position=[0, 0, 3],  # 相机在 z=3
    lookat=[0, 0, 0],  # 看向原点
    up=[0, 1, 0],  # y轴向上
    fov_y=45,  # 垂直视场角60度
    near=0.1,  # 近裁剪面(根据场景调整)
    far=100,  # 远裁剪面(根据场景调整)
    screen_w=640,  # 屏幕宽度(示例)
    screen_h=360  # 屏幕高度(示例)
)

print(f"Total Gaussians: {gm.mu_w.shape[0]}")
print(f"Point cloud center: {gm.mu_w.mean(axis=0)}")

# 视图变换
mu_c = view_transform_point(gm.mu_w, camera.R_view, camera.t_view)
print(f"Camera space Z: [{mu_c[:, 2].min():.2f}, {mu_c[:, 2].max():.2f}]")

# 投影
P = calculate_projection_matrix_from_fov(camera.fov_y, camera.aspect, camera.near, camera.far)
mu_ndc = project_to_ndc(mu_c, P)
mu_screen = ndc_to_screen(mu_ndc, camera.screen_w, camera.screen_h)

# 协方差变换
sigma_c = view_transform_covariance(gm.sigma_w, camera.R_view)
J = calculate_intrinsic_jacobian(mu_c[:, 0], mu_c[:, 1], mu_c[:, 2], camera.fx, camera.fy)
sigma_screen = project_to_screen(sigma_c, J)

# 从远到近排序
z_sorted_indices = cp.argsort(-mu_c[:, 2])
mu_screen = mu_screen[z_sorted_indices]
sigma_screen = sigma_screen[z_sorted_indices]
opacity = gm.opacity[z_sorted_indices]
sh_coeffs = gm.sh_coeffs[z_sorted_indices]
mu_w_sorted = gm.mu_w[z_sorted_indices]

# 计算颜色
camera_center_w = -camera.t_view @ camera.R_view
directions = mu_w_sorted - camera_center_w
directions = directions / cp.linalg.norm(directions, axis=1, keepdims=True)
colors = eval_sh(sh_coeffs, directions)
colors = cp.clip(colors, 0, 1)

print(
    f"Screen coords: X[{mu_screen[:, 0].min():.0f}, {mu_screen[:, 0].max():.0f}], Y[{mu_screen[:, 1].min():.0f}, {mu_screen[:, 1].max():.0f}]")

# 渲染
image = render(mu_screen, sigma_screen, opacity, colors, camera.screen_w, camera.screen_h, tile_size=16)
image = cp.clip(image, 0, 1)
image = (image * 255).astype(cp.uint8)
image = cp.asnumpy(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
cv2.imshow("image", image)
cv2.imwrite("output.png", image)
cv2.waitKey(0)

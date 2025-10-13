# test/test_simple_scene.py
import cupy as cp
import cv2
from gs_core.rasterization import render

# 三个高斯球：左上到右下对角线
mu_screen = cp.array([
    [80.0, 80.0],     # 左上
    [160.0, 160.0],    # 中间
    [240.0, 240.0]    # 右下
], dtype=cp.float32)

# 所有球都是 \ 方向的椭圆（左上-右下倾斜）
angle = cp.deg2rad(45)
c, s = cp.cos(angle), cp.sin(angle)
R = cp.array([[c, -s], [s, c]])
D = cp.diag(cp.array([150.0, 40.0]))  # 长轴150，短轴40
sigma_base = R @ D @ R.T

sigma = cp.stack([sigma_base, sigma_base, sigma_base], axis=0).astype(cp.float32)

opacity = cp.array([0.8, 0.8, 0.8], dtype=cp.float32)
color = cp.array([
    [1.0, 0.0, 0.0],  # 红
    [0.0, 1.0, 0.0],  # 绿
    [0.0, 0.0, 1.0]   # 蓝
], dtype=cp.float32)

screen_w, screen_h = 320, 320
image = render(mu_screen, sigma, opacity, color, screen_w, screen_h, tile_size=16)

image = cp.clip(image, 0, 1)
image = (image * 255).astype(cp.uint8)
image = cp.asnumpy(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.imshow("Test", image)
cv2.imwrite("test_simple.png", image)
cv2.waitKey(0)
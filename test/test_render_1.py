# test/test_simple_scene.py
import cupy as cp
import cv2
from gs_core.rasterization import render

# Single red gaussian at screen center
mu_screen = cp.array([[160.0, 96.0]], dtype=cp.float32)  # center of 320x192 screen

# Isotropic covariance (circular gaussian)
sigma = cp.array([
    [[100.0, 0.0],
     [0.0, 100.0]]
], dtype=cp.float32)

opacity = cp.array([0.9], dtype=cp.float32)
color = cp.array([[1.0, 0.0, 0.0]], dtype=cp.float32)  # red

screen_w, screen_h = 320, 192

image = render(mu_screen, sigma, opacity, color, screen_w, screen_h, tile_size=16)
image = cp.clip(image, 0, 1)
image = (image * 255).astype(cp.uint8)
image = cp.asnumpy(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.imshow("Test", image)
cv2.imwrite("test_simple.png", image)
cv2.waitKey(0)

# Expected: Red blurred circle at screen center
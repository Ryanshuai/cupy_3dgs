import cupy as cp
import cv2

from gs_core.rasterization import render

# Three gaussians in a row (RGB)
mu_screen = cp.array([
    [80.0, 96.0],  # left - red
    [160.0, 96.0],  # center - green
    [240.0, 96.0]  # right - blue
], dtype=cp.float32)

sigma = cp.array([[[50.0, 0.0], [0.0, 50.0]]] * 3, dtype=cp.float32)
opacity = cp.array([0.8, 0.8, 0.8], dtype=cp.float32)
color = cp.array([
    [1.0, 0.0, 0.0],  # red
    [0.0, 1.0, 0.0],  # green
    [0.0, 0.0, 1.0]  # blue
], dtype=cp.float32)

screen_w, screen_h = 320, 192

image = render(mu_screen, sigma, opacity, color, screen_w, screen_h, tile_size=16)
image = cp.clip(image, 0, 1)
image = (image * 255).astype(cp.uint8)
image = cp.asnumpy(image)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.imshow("RGB Test", image)
cv2.imwrite("test_three_gaussians.png", image)
cv2.waitKey(0)

# Expected: Three blurred circles - red, green, blue from left to right.

import cupy as cp


def intrinsics_from_fov(fov_y_rad, W, H, xp=cp):
    fy = (H * 0.5) / xp.tan(0.5 * fov_y_rad)
    fx = fy
    cx, cy = W * 0.5, H * 0.5
    return fx, fy, cx, cy



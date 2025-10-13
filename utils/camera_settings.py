import cupy as cp


def intrinsics_from_fov(fov_y_rad, W, H, xp=cp):
    fy = (H * 0.5) / xp.tan(0.5 * fov_y_rad)
    fx = fy
    cx, cy = W * 0.5, H * 0.5
    return fx, fy, cx, cy


class Camera:
    def __init__(self, position, lookat, up, fov_y, near, far, screen_w, screen_h):
        self.position = cp.array(position, dtype=cp.float32)
        self.lookat = cp.array(lookat, dtype=cp.float32)
        self.up = cp.array(up, dtype=cp.float32)
        self.fov_y = fov_y
        self.aspect = screen_w / screen_h
        self.near = near
        self.far = far
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.fx, self.fy, self.cx, self.cy = intrinsics_from_fov(fov_y, screen_w, screen_h)

        self.R_view, self.t_view = self.compute_view_matrix()

    def compute_view_matrix(self):
        forward = self.lookat - self.position
        forward = forward / cp.linalg.norm(forward)

        right = cp.cross(forward, self.up)
        right = right / cp.linalg.norm(right)

        up = cp.cross(right, forward)

        R_view = cp.stack([right, up, -forward], axis=0)
        t_view = -R_view @ self.position

        return R_view, t_view

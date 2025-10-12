import cupy as cp


def view_transform_point(mu_w, R, t):
    mu_c = mu_w @ R.T + t
    return mu_c


def calculate_projection_matrix_from_fov(fov_y, aspect, near, far, xp=cp):
    """
    OpenGL-style 4x4 perspective matrix (right-handed, NDC z∈[-1,1]).
    fov_y: vertical field of view in radians; aspect=W/H; near/far>0
    """
    f = 1.0 / xp.tan(0.5 * fov_y)
    n, fa = near, far
    P = xp.zeros((4, 4), dtype=xp.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (fa + n) / (n - fa)
    P[2, 3] = (2 * fa * n) / (n - fa)
    P[3, 2] = -1.0  # w -> -z_c, clip -> NDC
    return P


def project_to_ndc(mu_c, P):  # P: 4x4 projection matrix
    mu_c_homo = cp.concatenate([mu_c, cp.ones((mu_c.shape[0], 1))], axis=-1)
    mu_p_homo = (P @ mu_c_homo.T).T
    mu_p = mu_p_homo[:, :3] / mu_p_homo[:, 3:4]  # clip -> NDC
    return mu_p


def ndc_to_screen(mu_ndc, width, height):
    u = (mu_ndc[:, 0] + 1) * 0.5 * width
    v = (1 - mu_ndc[:, 1]) * 0.5 * height
    return cp.stack([u, v], axis=-1)



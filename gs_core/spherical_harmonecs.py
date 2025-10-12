import cupy as cp


def eval_sh(sh_coeffs, directions):  # direction: gaussians to camera: camera_pos - mu_w = -mu_c
    N = sh_coeffs.shape[0]
    result = cp.zeros((N, 3))

    dirs = directions / cp.linalg.norm(directions, axis=1, keepdims=True)
    x, y, z = dirs[:, [0]], dirs[:, [1]], dirs[:, [2]]

    C0 = 0.28209479177387814
    result += C0 * sh_coeffs[:, 0, :]

    # C1 = 0.4886025119029199
    # result += C1 * (-y) * sh_coeffs[:, 1, :]
    # result += C1 * z * sh_coeffs[:, 2, :]
    # result += C1 * (-x) * sh_coeffs[:, 3, :]
    #
    # C2_0 = 1.0925484305920792
    # C2_1 = -1.0925484305920792
    # C2_2 = 0.31539156525252005
    # C2_3 = -1.0925484305920792
    # C2_4 = 0.5462742152960396
    #
    # xx, yy, zz = x * x, y * y, z * z
    # xy, yz, xz = x * y, y * z, x * z
    #
    # result += C2_0 * xy * sh_coeffs[:, 4, :]
    # result += C2_1 * yz * sh_coeffs[:, 5, :]
    # result += C2_2 * (2.0 * zz - xx - yy) * sh_coeffs[:, 6, :]
    # result += C2_3 * xz * sh_coeffs[:, 7, :]
    # result += C2_4 * (xx - yy) * sh_coeffs[:, 8, :]
    #
    # C3_0 = -0.5900435899266435
    # C3_1 = 2.890611442640554
    # C3_2 = -0.4570457994644658
    # C3_3 = 0.3731763325901154
    # C3_4 = -0.4570457994644658
    # C3_5 = 1.445305721320277
    # C3_6 = -0.5900435899266435
    #
    # result += C3_0 * y * (3 * xx - yy) * sh_coeffs[:, 9, :]
    # result += C3_1 * xy * z * sh_coeffs[:, 10, :]
    # result += C3_2 * y * (4 * zz - xx - yy) * sh_coeffs[:, 11, :]
    # result += C3_3 * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeffs[:, 12, :]
    # result += C3_4 * x * (4 * zz - xx - yy) * sh_coeffs[:, 13, :]
    # result += C3_5 * z * (xx - yy) * sh_coeffs[:, 14, :]
    # result += C3_6 * x * (xx - 3 * yy) * sh_coeffs[:, 15, :]

    return cp.clip(result, 0, 1)

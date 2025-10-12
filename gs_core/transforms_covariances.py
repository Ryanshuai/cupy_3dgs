# the author refer to https://www.cs.umd.edu/~zwicker/publications/EWAVolumeSplatting-VIS01.pdf,
# but seems implementation is not exactly the same as the paper

import cupy as cp


def view_transform_covariance(Sigma_w, R):
    Sigma_c = R @ Sigma_w @ R.T
    return Sigma_c


def calculate_intrinsic_jacobian(x_c, y_c, z_c, fx, fy):
    """
    Compute the Jacobian of the perspective projection (intrinsics).

    Args:
        x_c, y_c, z_c : float or ndarray
            3D point(s) in camera coordinates.
        fx, fy : float
            Focal lengths (camera intrinsics).

    Returns:
        J : ndarray of shape  (N, 2, 3)
            The Jacobian matrix:
                [ [ fx/z,   0,      -fx * x / z^2 ],
                  [  0,    fy/z,    -fy * y / z^2 ] ]
    """
    # Avoid division by zero
    z_c_safe = cp.maximum(z_c, 1e-8)

    # Compute partial derivatives
    J11 = fx / z_c_safe
    J12 = cp.zeros_like(J11)
    J13 = -fx * x_c / (z_c_safe ** 2)

    J21 = cp.zeros_like(J11)
    J22 = fy / z_c_safe
    J23 = -fy * y_c / (z_c_safe ** 2)

    J = cp.stack([
        cp.stack([J11, J12, J13], axis=-1),
        cp.stack([J21, J22, J23], axis=-1)
    ], axis=-2)

    return J


def project_to_screen(Sigma_c, J):
    Sigma_p = J @ Sigma_c @ J.transpose(0, 2, 1)
    return Sigma_p

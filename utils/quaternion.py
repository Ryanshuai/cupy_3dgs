import cupy as cp


def quaternion_to_matrix(wxyz: cp.ndarray) -> cp.ndarray:
    wxyz_norm = cp.linalg.norm(wxyz, axis=1, keepdims=True)
    wxyz = wxyz / wxyz_norm

    w = wxyz[..., 0]
    x = wxyz[..., 1]
    y = wxyz[..., 2]
    z = wxyz[..., 3]

    rotation_matrix = cp.empty(wxyz.shape[:-1] + (3, 3), dtype=wxyz.dtype)

    rotation_matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    rotation_matrix[..., 0, 1] = 2 * (x * y - z * w)
    rotation_matrix[..., 0, 2] = 2 * (x * z + y * w)
    rotation_matrix[..., 1, 0] = 2 * (x * y + z * w)
    rotation_matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    rotation_matrix[..., 1, 2] = 2 * (y * z - x * w)
    rotation_matrix[..., 2, 0] = 2 * (x * z - y * w)
    rotation_matrix[..., 2, 1] = 2 * (y * z + x * w)
    rotation_matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return rotation_matrix

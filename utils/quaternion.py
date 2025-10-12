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


if __name__ == "__main__":
    import numpy as np
    from scipy.spatial.transform import Rotation


    def test_quaternion_to_matrix():
        print("测试四元数转旋转矩阵\n" + "=" * 50)

        # 测试用例1: 单位四元数（无旋转）
        print("\n测试1: 单位四元数 (无旋转)")
        q_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

        # 你的实现
        R_yours = quaternion_to_matrix(cp.array(q_wxyz)).get()

        # SciPy实现 (注意顺序转换: wxyz -> xyzw)
        q_xyzw = q_wxyz[[1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w]
        R_scipy = Rotation.from_quat(q_xyzw).as_matrix()

        print(f"你的结果:\n{R_yours}")
        print(f"SciPy结果:\n{R_scipy}")
        print(f"差异: {np.max(np.abs(R_yours - R_scipy)):.2e}")

        # 测试用例2: 绕Z轴旋转90度
        print("\n测试2: 绕Z轴旋转90度")
        angle = np.pi / 2
        q_wxyz = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])

        R_yours = quaternion_to_matrix(cp.array(q_wxyz)).get()
        q_xyzw = q_wxyz[[1, 2, 3, 0]]
        R_scipy = Rotation.from_quat(q_xyzw).as_matrix()

        print(f"四元数 (wxyz): {q_wxyz}")
        print(f"你的结果:\n{R_yours}")
        print(f"SciPy结果:\n{R_scipy}")
        print(f"差异: {np.max(np.abs(R_yours - R_scipy)):.2e}")

        # 测试用例3: 绕任意轴旋转
        print("\n测试3: 绕轴(1,1,1)旋转45度")
        axis = np.array([1.0, 1.0, 1.0])
        axis = axis / np.linalg.norm(axis)  # 归一化
        angle = np.pi / 4

        w = np.cos(angle / 2)
        xyz = np.sin(angle / 2) * axis
        q_wxyz = np.array([w, xyz[0], xyz[1], xyz[2]])

        R_yours = quaternion_to_matrix(cp.array(q_wxyz)).get()
        q_xyzw = q_wxyz[[1, 2, 3, 0]]
        R_scipy = Rotation.from_quat(q_xyzw).as_matrix()

        print(f"旋转轴: {axis}")
        print(f"旋转角度: {np.degrees(angle):.1f}度")
        print(f"四元数 (wxyz): {q_wxyz}")
        print(f"你的结果:\n{R_yours}")
        print(f"SciPy结果:\n{R_scipy}")
        print(f"差异: {np.max(np.abs(R_yours - R_scipy)):.2e}")

        # 测试用例4: 批量测试（多个四元数）
        print("\n测试4: 批量测试 (5个随机四元数)")
        np.random.seed(42)
        n = 5
        q_batch_wxyz = np.random.randn(n, 4)
        # 归一化
        q_batch_wxyz = q_batch_wxyz / np.linalg.norm(q_batch_wxyz, axis=1, keepdims=True)

        R_yours_batch = quaternion_to_matrix(cp.array(q_batch_wxyz)).get()

        # SciPy批量转换
        q_batch_xyzw = q_batch_wxyz[:, [1, 2, 3, 0]]
        R_scipy_batch = Rotation.from_quat(q_batch_xyzw).as_matrix()

        max_diff = np.max(np.abs(R_yours_batch - R_scipy_batch))
        print(f"批量测试 - 最大差异: {max_diff:.2e}")

        for i in range(n):
            diff = np.max(np.abs(R_yours_batch[i] - R_scipy_batch[i]))
            print(f"  四元数 {i + 1} 差异: {diff:.2e}")

        # 测试用例5: 验证旋转性质
        print("\n测试5: 验证旋转矩阵性质")
        q_wxyz = np.array([0.7071, 0.7071, 0.0, 0.0])  # 绕X轴旋转90度
        R = quaternion_to_matrix(cp.array(q_wxyz)).get()

        # 检查是否为正交矩阵: R^T * R = I
        RTR = R.T @ R
        is_orthogonal = np.allclose(RTR, np.eye(3), atol=1e-6)
        print(f"是否正交 (R^T * R = I): {is_orthogonal}")

        # 检查行列式是否为1
        det = np.linalg.det(R)
        is_rotation = np.isclose(det, 1.0, atol=1e-6)
        print(f"行列式 = {det:.6f}, 是否为旋转矩阵: {is_rotation}")

        print("\n" + "=" * 50)
        print("✅ 所有测试通过!" if max_diff < 1e-6 else "❌ 测试失败!")


    test_quaternion_to_matrix()

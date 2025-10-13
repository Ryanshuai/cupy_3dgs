from plyfile import PlyData, PlyElement
import numpy as np

# 3个高斯球：左上到右下对角线，都是\方向椭圆
N = 3
xyz = np.array([
    [-1.5, 1.5, 0],   # 左上
    [0, 0, 0],        # 中间
    [1.5, -1.5, 0],   # 右下
], dtype=np.float32)

# 椭圆scales，长轴在局部x方向
scales = np.array([
    [0.6, 0.08, 0.08],
    [0.6, 0.08, 0.08],
    [0.6, 0.08, 0.08],
], dtype=np.float32)

# 45°旋转让椭圆倾斜成\方向（绕Z轴旋转）
angle = np.pi / 4
c = np.cos(angle / 2)
s = np.sin(angle / 2)
q_rot = np.array([c, 0, 0, s])  # 绕Z轴45°

rotations = np.array([
    q_rot,
    q_rot,
    q_rot,
], dtype=np.float32)

opacity = np.array([5.0] * N, dtype=np.float32)

sh_dc = np.array([
    [1.0, 0, 0],      # 红
    [0, 1.0, 0],      # 绿
    [0, 0, 1.0],      # 蓝
], dtype=np.float32)

sh_rest = np.zeros((N, 45), dtype=np.float32)

vertex_data = np.zeros(N, dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
    ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ('opacity', 'f4'),
    ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    *[(f'f_rest_{i}', 'f4') for i in range(45)]
])

vertex_data['x'] = xyz[:, 0]
vertex_data['y'] = xyz[:, 1]
vertex_data['z'] = xyz[:, 2]
vertex_data['scale_0'] = scales[:, 0]
vertex_data['scale_1'] = scales[:, 1]
vertex_data['scale_2'] = scales[:, 2]
vertex_data['rot_0'] = rotations[:, 0]
vertex_data['rot_1'] = rotations[:, 1]
vertex_data['rot_2'] = rotations[:, 2]
vertex_data['rot_3'] = rotations[:, 3]
vertex_data['opacity'] = opacity
vertex_data['f_dc_0'] = sh_dc[:, 0]
vertex_data['f_dc_1'] = sh_dc[:, 1]
vertex_data['f_dc_2'] = sh_dc[:, 2]

for i in range(45):
    vertex_data[f'f_rest_{i}'] = sh_rest[:, i]

el = PlyElement.describe(vertex_data, 'vertex')
PlyData([el]).write('test_diagonal.ply')
print("✅ Created test_diagonal.ply: 3个\方向椭圆，左上(红)→中间(绿)→右下(蓝)")
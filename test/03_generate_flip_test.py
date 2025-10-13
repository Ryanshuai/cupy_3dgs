from plyfile import PlyData, PlyElement
import numpy as np

# 3 Gaussian spheres: diagonal from top-left to bottom-right, all oriented in \ direction
N = 3

xyz = np.array([
    [-1.5, 1.5, 0],   # top-left
    [0, 0, 0],        # center
    [1.5, -1.5, 0],   # bottom-right
], dtype=np.float32)

# Ellipsoid scales, major axis in local x direction
scales = np.array([
    [0.6, 0.08, 0.08],
    [0.6, 0.08, 0.08],
    [0.6, 0.08, 0.08],
], dtype=np.float32)

# 45° rotation to tilt ellipsoid into \ direction (rotation around Z axis)
angle = np.pi / 4
c = np.cos(angle / 2)
s = np.sin(angle / 2)
q_rot = np.array([c, 0, 0, s])  # 45° around Z axis

rotations = np.array([
    q_rot,
    q_rot,
    q_rot,
], dtype=np.float32)

opacity = np.array([5.0] * N, dtype=np.float32)

sh_dc = np.array([
    [0, 1.0, 0],      # green
    [1.0, 0, 0],      # red
    [0, 0, 1.0],      # blue
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
PlyData([el]).write('test_flip_test.ply')

print("✅ Created test_flip_test.ply: 3 ellipsoids in \ direction, top-left(green)→center(red)→bottom-right(blue)")
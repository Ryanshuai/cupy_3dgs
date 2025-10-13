from plyfile import PlyData, PlyElement
import numpy as np

# 2 Gaussian spheres
N = 2

# Positions
xyz = np.array([
    [0, 0, 0],    # red sphere at z=0
    [0, 0, -2],   # green sphere at z=-2
], dtype=np.float32)

# Ellipsoid scales, major axis in local x direction
scales = np.array([
    [0.7, 0.05, 0.05],  # red sphere scale
    [0.7, 0.05, 0.05],  # green sphere scale
], dtype=np.float32)

# Rotations:
# Red sphere: rotate 45° around Z axis (top-left→bottom-right \ direction)
# Green sphere: rotate -45° around Z axis (bottom-left→top-right / direction)
angle1 = np.pi / 4   # 45°
angle2 = -np.pi / 4  # -45°

c1 = np.cos(angle1 / 2)
s1 = np.sin(angle1 / 2)
q_rot1 = np.array([c1, 0, 0, s1])  # 45° around Z axis

c2 = np.cos(angle2 / 2)
s2 = np.sin(angle2 / 2)
q_rot2 = np.array([c2, 0, 0, s2])  # -45° around Z axis

rotations = np.array([
    q_rot1,  # red sphere: \ direction
    q_rot2,  # green sphere: / direction
], dtype=np.float32)

# Opacity
presig_opacity = np.array([20.0, 20.0], dtype=np.float32)

# Color: SH DC component
sh_dc = np.array([
    [1.0, 0, 0],  # red
    [0, 1.0, 0],  # green
], dtype=np.float32)

# SH rest components (45 higher-order coefficients)
sh_rest = np.zeros((N, 45), dtype=np.float32)

# Create PLY vertex data structure
vertex_data = np.zeros(N, dtype=[
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
    ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ('opacity', 'f4'),
    ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    *[(f'f_rest_{i}', 'f4') for i in range(45)]
])

# Fill data
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
vertex_data['opacity'] = presig_opacity
vertex_data['f_dc_0'] = sh_dc[:, 0]
vertex_data['f_dc_1'] = sh_dc[:, 1]
vertex_data['f_dc_2'] = sh_dc[:, 2]
for i in range(45):
    vertex_data[f'f_rest_{i}'] = sh_rest[:, i]

# Save as PLY file
el = PlyElement.describe(vertex_data, 'vertex')
PlyData([el]).write('test_z_sort.ply')

print("✅ Created test_z_sort.ply:")
print("   - z=0:  red ellipsoid, \\ direction (top-left→bottom-right)")
print("   - z=-2: green ellipsoid, / direction (bottom-left→top-right)")
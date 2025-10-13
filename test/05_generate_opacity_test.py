from plyfile import PlyData, PlyElement
import numpy as np

# 3 Gaussian spheres
N = 3

# Positions: three spheres at same xy position, different z depths
xyz = np.array([
    [0, 0, 0],      # front: red, opaque
    [0, 0, -1],     # middle: green, semi-transparent
    [0, 0, -2],     # back: blue, semi-transparent
], dtype=np.float32)

# Ellipsoid scales, all made spherical for easy observation
scales = np.array([
    [0.5, 0.5, 0.5],  # red sphere
    [0.5, 0.5, 0.5],  # green sphere
    [0.5, 0.5, 0.5],  # blue sphere
], dtype=np.float32)

# Rotations: no rotation (identity quaternion)
rotations = np.array([
    [1, 0, 0, 0],  # red sphere
    [1, 0, 0, 0],  # green sphere
    [1, 0, 0, 0],  # blue sphere
], dtype=np.float32)

# Opacity: red fully opaque, green and blue semi-transparent
presig_opacity = np.array([
    20.0,   # red: sigmoid(20) ≈ 1.0 (fully opaque)
    0.0,    # green: sigmoid(0) = 0.5 (semi-transparent)
    0.0,    # blue: sigmoid(0) = 0.5 (semi-transparent)
], dtype=np.float32)

# Color: SH DC component
sh_dc = np.array([
    [1.0, 0, 0],    # red
    [0, 1.0, 0],    # green
    [0, 0, 1.0],    # blue
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
PlyData([el]).write('test_opacity.ply')

print("✅ Created test_opacity.ply: 3 overlapping spheres")
print("   Positions: all at (0, 0, z)")
print("   - z=0:  red, fully opaque (opacity=20.0)")
print("   - z=-1: green, semi-transparent (opacity=0.0)")
print("   - z=-2: blue, semi-transparent (opacity=0.0)")
print("\nExpected result (viewing from z>0 toward origin):")
print("   ✓ If depth sorting is correct: should only see red (frontmost and opaque)")
print("   ✗ If depth sorting is incorrect: may see green or blue")

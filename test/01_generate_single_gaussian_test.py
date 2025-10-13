from plyfile import PlyData, PlyElement
import numpy as np

# 1 Gaussian sphere
N = 1

# Position: at origin
xyz = np.array([
    [0, 0, 0],      # Red sphere at origin
], dtype=np.float32)

# Scale: medium-sized sphere
scales = np.array([
    [0.5, 0.5, 0.5],  # Spherical
], dtype=np.float32)

# Rotation: no rotation (identity quaternion)
rotations = np.array([
    [1, 0, 0, 0],
], dtype=np.float32)

# Opacity: fully opaque
presig_opacity = np.array([
    20.0,   # sigmoid(20) ≈ 1.0 (fully opaque)
], dtype=np.float32)

# Color: pure red
sh_dc = np.array([
    [1.0, 0, 0],    # Red
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
PlyData([el]).write('test_single_gaussian.ply')

print("✅ Created test_single_gaussian.ply: 1 red sphere")
print("   Position: (0, 0, 0)")
print("   Scale: 0.5 (spherical)")
print("   Color: pure red")
print("   Opacity: fully opaque")
print("\nExpected result (viewing from z>0 toward origin):")
print("   ✓ Should see a red circle at center")
print("   ✓ Background should be black (0, 0, 0)")
print("   ✓ Edges should have Gaussian blur gradient effect")
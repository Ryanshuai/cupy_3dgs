from plyfile import PlyData, PlyElement
import numpy as np

# Create 4 Gaussians
N = 4
xyz = np.array([
    [0, 0, 0],    # center - red sphere
    [-2, 0, 0],   # left - green ellipsoid (z elongated)
    [2, 0, 0],    # right - blue ellipsoid (x elongated)
    [0, 1.5, 0],  # top - red rotated sphere
], dtype=np.float32)

# Scale: [sx, sy, sz]
scales = np.array([
    [0.16, 0.08, 0.08],
    [0.08, 0.16, 0.08],
    [0.08, 0.08, 0.16],
    [0.16, 0.08, 0.08],  # same as center
], dtype=np.float32)

# Quaternion for 45° rotation around each axis
# First rotate around X, then Y, then Z
angle = np.pi / 4  # 45 degrees
c = np.cos(angle / 2)
s = np.sin(angle / 2)

# Individual rotations (wxyz format)
qx = np.array([c, s, 0, 0])  # X axis
qy = np.array([c, 0, s, 0])  # Y axis
qz = np.array([c, 0, 0, s])  # Z axis

# Quaternion multiplication: qz * qy * qx
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

q_combined = quat_mult(quat_mult(qz, qy), qx)

rotations = np.array([
    [1, 0, 0, 0],  # identity
    [1, 0, 0, 0],  # identity
    [1, 0, 0, 0],  # identity
    q_combined,    # 45° on all axes
], dtype=np.float32)

# High opacity
opacity = np.array([5.0] * N, dtype=np.float32)

# Spherical harmonics DC component (RGB colors)
sh_dc = np.array([
    [1.0, 0, 0],    # red
    [0, 1.0, 0],    # green
    [0, 0, 1.0],    # blue
    [1.0, 1.0, 1.0],    # red (top)
], dtype=np.float32)

# Higher order SH coefficients (all zeros)
sh_rest = np.zeros((N, 45), dtype=np.float32)

# Build vertex array
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

# Save PLY file
el = PlyElement.describe(vertex_data, 'vertex')
PlyData([el]).write('test_minimal.ply')

print("✅ Created test_minimal.ply with 4 Gaussians:")
print("   - Center: red sphere")
print("   - Left: green ellipsoid (z elongated)")
print("   - Right: blue ellipsoid (x elongated)")
print("   - Top: red ellipsoid rotated 45° on X, Y, Z axes")
print(f"   - Top rotation quaternion: {q_combined}")
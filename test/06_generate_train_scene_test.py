from plyfile import PlyData, PlyElement
import numpy as np

# Read point cloud
plydata = PlyData.read('../data/train/point_cloud/iteration_30000/point_cloud.ply')
vertex = plydata['vertex'].data
points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)

# Calculate overall range
mins, maxs = points.min(axis=0), points.max(axis=0)
full_range = maxs - mins
max_len = full_range.max()  # Take the length of the maximum axis

# Define center region range (±1/10 max_len)
half_len = max_len / 100.0
mask = (
    (points[:, 0] > -half_len) & (points[:, 0] < half_len) &
    (points[:, 1] > -half_len) & (points[:, 1] < half_len) &
    (points[:, 2] > -half_len) & (points[:, 2] < half_len)
)

# Filter points
cropped_vertex = vertex[mask]

# Write new PLY
PlyData([PlyElement.describe(cropped_vertex, 'vertex')]).write('test_train_scene.ply')
print(f"Cropping complete, retained {len(cropped_vertex)} points, output file: test_train_scene.ply")
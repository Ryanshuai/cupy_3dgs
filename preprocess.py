from plyfile import PlyData
import numpy as np

plydata = PlyData.read('data/train/point_cloud/iteration_30000/point_cloud.ply')

vertex = plydata['vertex']

xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
rotations = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)
opacity = np.array(vertex['opacity'])

sh_dc = np.stack([
    vertex['f_dc_0'],  # R
    vertex['f_dc_1'],  # G
    vertex['f_dc_2']  # B
], axis=1)  # shape: (N, 3)

rest_names = ["f_rest_" + str(i) for i in range(45)]

sh_rest = np.stack([vertex[name] for name in rest_names], axis=1)

num_coeffs = len(rest_names) // 3
sh_rest = sh_rest.reshape(-1, num_coeffs, 3)


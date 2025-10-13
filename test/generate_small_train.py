from plyfile import PlyData, PlyElement
import numpy as np

# 读取点云
plydata = PlyData.read('../data/train/point_cloud/iteration_30000/point_cloud.ply')
vertex = plydata['vertex'].data
points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)

# 计算总体范围
mins, maxs = points.min(axis=0), points.max(axis=0)
full_range = maxs - mins
max_len = full_range.max()  # 取最大轴的长度

# 定义中心区域范围（±1/10 max_len）
half_len = max_len / 100.0
mask = (
    (points[:, 0] > -half_len) & (points[:, 0] < half_len) &
    (points[:, 1] > -half_len) & (points[:, 1] < half_len) &
    (points[:, 2] > -half_len) & (points[:, 2] < half_len)
)

# 筛选点
cropped_vertex = vertex[mask]

# 写出新的 PLY
PlyData([PlyElement.describe(cropped_vertex, 'vertex')]).write('cropped_center_1of5.ply')
print(f"裁剪完成，共保留 {len(cropped_vertex)} 个点，输出文件：cropped_center_1of5.ply")
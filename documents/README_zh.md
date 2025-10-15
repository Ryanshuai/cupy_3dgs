# 3D Gaussian Splatting - CuPy 实现

中文文档 | [English](../README.md)

<p align="center">
  <i>一个简洁、易懂的 3D Gaussian Splatting 推理实现</i>
</p>

## 🎯 项目目的

这是一个基于 **CuPy** 的 3D Gaussian Splatting **推理实现**，旨在提供一个简洁、易于理解的代码库，帮助开发者深入理解 3DGS 的核心渲染原理。

<p align="center">
  <img src="../output.png" alt="Rendered Train Scene" width="800"/>
  <br>
  <i>渲染效果：使用本项目渲染的火车场景</i>
</p>

## 📊 与原始实现的对比

| 特性 | 本实现 | 原始 CUDA 实现 |
|------|--------|----------------|
| 语言 | Python + CuPy | C++ + CUDA |
| 功能 | 仅推理 | 训练 + 推理 |
| 代码量 | ~800 行 | ~5000+ 行 |
| 依赖 | cupy, numpy, opencv-python, plyfile | 复杂的编译环境 |
| 测试用例 | 6 个渐进式测试场景 | 无单元测试 |
| 可读性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 适用场景 | 学习、理解算法 | 生产、训练模型 |

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 下载训练好的场景数据

从官方 3DGS 项目下载预训练的火车场景（包含训练好的 PLY 文件）：
```bash
# 下载预训练模型（约 14GB）
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip
unzip models.zip

# 解压后，找到 train 场景的点云文件：
# train/point_cloud/iteration_30000/point_cloud.ply
# 将其放置到项目的 data/train/point_cloud/iteration_30000/ 目录下
```

### 3. 准备测试数据
```bash
cd test
python 06_generate_train_scene_test.py
```

### 4. 运行渲染
```bash
python pipeline.py
```

渲染完成后，会在项目根目录生成 **`output.png`** 文件。

## 📚 参考资料

本实现基于以下论文和代码：

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [官方 GitHub 实现](https://github.com/graphdeco-inria/gaussian-splatting)
- [官方 CUDA 光栅化实现](https://github.com/graphdeco-inria/diff-gaussian-rasterization)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！特别欢迎：
- 性能优化建议
- 更多测试场景
- 文档完善

## 📝 License

本项目遵循 MIT License。

---

<p align="center">
  <b>如果这个项目帮助你理解了 3D Gaussian Splatting，请给个 ⭐️ Star 支持一下！</b>
</p>
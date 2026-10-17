---
layout: home

hero:
  name: statgpu
  text: 让统计计算更快
  tagline: 以 sklearn 风格 API 提供 GPU 加速统计方法、新手友好的方法指南与可复现的基准证据。
  image:
    light: /images/statgpu-compute-hero-light.jpg
    dark: /images/statgpu-compute-hero.webp
    alt: 连接统计曲线与数据路径的 GPU 计算网格
  actions:
    - theme: brand
      text: 快速上手
      link: /cn/getting-started/quickstart
    - theme: alt
      text: 浏览模型
      link: /cn/models/README
    - theme: alt
      text: 打开基准面板
      link: /dashboard/
      target: _self

features:
  - title: 熟悉的接口，丰富的统计方法
    details: 提供 sklearn 风格的建模接口，多种回归模型支持 R 风格的公式输入。配合原理说明与示例，帮助你选择方法、理解结果。
  - title: 按任务选择 CPU 或 GPU
    details: 支持 NumPy、CuPy 与 PyTorch 后端，根据数据规模、计算任务和方法支持范围，选择合适的计算设备。
  - title: 探索大规模统计计算的加速潜力
    details: 通过不同数据规模下的性能对比，了解 GPU 在大样本、高维和重复计算任务中的潜在优势，找到适合你的加速场景。
---

## 使用指南

先选择适合统计问题的方法，再判断计算需求，逐步完成拟合、验证与结果解读。

| 使用步骤 | 从这里开始 |
|---|---|
| 1. 选择适合问题的统计方法 | [模型与方法目录](models/README.md) |
| 2. 判断是否需要 GPU | [GPU 入门指南](guides/when-to-use-gpu.md)<br>进阶：[设备与显存配置](guides/device-and-memory.md) |
| 3. 准备数据并拟合模型 | [数组输入与首次拟合](getting-started/quickstart.md) · [R 风格公式输入](reference/linear-model-api.md#formula-inputs) |
| 4. 调参与交叉验证 | [交叉验证指南](guides/cross-validation.md) |
| 5. 理解估计结果与统计推断 | [系数推断方法](guides/inference-modes.md) · [统计检验与结果对象](guides/inference-api.md) |

## 方法与参考

- 方法分类：[回归](models/regression/README.md) · [生存分析](models/survival/README.md) · [统计检验](models/testing/README.md) · [非参数方法](models/nonparametric/README.md) · [面板数据](panel/README.md) · [无监督学习](unsupervised/README.md)
- 接口与支持范围：[完整 API 参考](README.md#完整-api-参考) · [已实现方法](guides/implemented-methods.md)
- 求解器：[算法说明](guides/solver-algorithms.md) · [兼容性矩阵](guides/solver-penalty-matrix.md)
- 性能参考：[基准结果解读](benchmarks.md) · [交互式基准面板](../en/guides/statgpu_benchmark_dashboard.md#open-dashboard)

## 联系与参与

联系邮箱：[linziqian@fzu.edu.cn](mailto:linziqian@fzu.edu.cn)。

遇到问题或有功能建议，欢迎通过 [GitHub Issues](https://github.com/TheHiddenObserver/statgpu/issues) 反馈。参与代码、文档或测试工作前，请阅读[贡献指南](https://github.com/TheHiddenObserver/statgpu/blob/master/CONTRIBUTING.md)。

[致谢](acknowledgements.md) ·
[更新日志](changelog.md) ·
[发布指南](https://github.com/TheHiddenObserver/statgpu/blob/master/RELEASING.md) ·
[GitHub](https://github.com/TheHiddenObserver/statgpu)

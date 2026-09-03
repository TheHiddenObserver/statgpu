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
      link: /cn/models/
    - theme: alt
      text: 打开基准面板
      link: /dashboard/
      target: _self

features:
  - title: 不只介绍 API，也讲清方法
    details: 模型文档说明适用场景、直观原理与假设，并提供可运行示例、结果解读和常见误区。
  - title: 明确的 CPU 与 GPU 行为
    details: 可在 CPU 上使用 NumPy，或在 CUDA 上使用 CuPy 与 PyTorch；显式请求 GPU 时不会静默回退到 CPU。
  - title: 可以核查的性能证据
    details: 在基准面板中查看版本化测试结果、硬件环境、验证状态和数据来源。
---

## 探索 statgpu

| 我想要…… | 从这里开始 |
|---|---|
| 运行第一个模型 | [快速上手](getting-started/quickstart.md) |
| 选择合适的统计方法 | [模型目录](models/) |
| 查看当前已实现能力 | [已实现方法](guides/implemented-methods.md) |
| 配置 CPU、CUDA 或 Torch | [设备与显存指南](guides/device-and-memory.md) |
| 理解标准误与统计检验 | [推断 API](guides/inference-api.md) |
| 比较实测性能 | [交互式基准面板](/dashboard/) |

## 主要内容

- [回归与统计模型](models/)
- [无监督学习](unsupervised/)
- [面板数据模型](panel/)
- [交叉验证](guides/cross-validation.md)
- [更新日志](changelog.md)

[参与贡献](https://github.com/TheHiddenObserver/statgpu/blob/master/CONTRIBUTING.md) ·
[发布指南](https://github.com/TheHiddenObserver/statgpu/blob/master/RELEASING.md) ·
[GitHub](https://github.com/TheHiddenObserver/statgpu)

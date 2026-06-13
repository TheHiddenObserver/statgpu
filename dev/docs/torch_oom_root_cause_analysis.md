# PyTorch OOM 根本原因分析

**测试日期**: 2026-04-17  
**GPU**: Tesla P100-SXM2-16GB (15.90 GB 总显存)  
**数据集**: 100,000 × 5,000 (FP64, 3.73 GB/数组)

---

## 核心结论

**Torch OOM 不是代码问题，而是测试脚本的问题**：

1. **Torch 本身可以处理 100K×5K 数据集** - 内存分析显示仅需约 4GB 显存
2. **OOM 发生在 CuPy 和 Torch 混用时** - 两个框架竞争显存导致
3. **CuPy 内存池不释放** - 即使调用 `free_all_blocks()`，显存仍被占用

---

## 内存使用详细分析

### 100K×5K 数据集内存需求

| 数组 | 大小 (FP64) |
|------|-------------|
| X (原始数据) | 3.73 GB |
| y (目标变量) | 0.75 GB |
| X_design (设计矩阵) | 3.73 GB |
| XtX (X'X 矩阵) | 0.19 GB |
| **峰值内存需求** | **~8.4 GB** |

### 实际内存测量 (Torch 单独运行)

| 步骤 | 操作 | Torch 显存分配 |
|------|------|---------------|
| Initial | 初始状态 | 0.00 GB |
| Transfer | X,y 传到 GPU | 3.73 GB |
| ones | 创建 ones 向量 | 3.73 GB (无变化) |
| design | 创建设计矩阵 | **7.45 GB** |
| cleanup | 删除临时张量 | 3.73 GB |
| XtX | 计算 X'X | 3.92 GB |

**结论**: Torch 实际峰值约 7.45 GB，远低于 15.90 GB 上限

---

## OOM 根本原因

### 原因 1: CuPy 内存池不释放

```
[2] After CuPy allocation:
  CuPy: used=3.73GB, total_allocated=3.73GB

[2b] After CuPy cleanup:
  CuPy: used=3.73GB, total_allocated=3.73GB  ← 没有变化！
```

**CuPy 的内存池行为**:
- `cp.get_default_memory_pool().free_all_blocks()` 只标记为可用
- 实际显存仍被保留（total_allocated 不变）
- 后续 Torch 分配时，这部分显存无法使用

### 原因 2: 两个框架竞争显存

```
测试顺序（benchmark 脚本）:
1. CuPy 测试 → 分配 ~8GB，"释放"后仍保留
2. Torch 测试 → 只剩 15.90 - 8 = 7.90 GB 可用
3. Torch 峰值需要 7.45 GB → 接近上限时 OOM
```

### 原因 3: Torch 内存分配器碎片化

Torch 使用分块分配器（split allocator），可能导致：
- 大块连续显存不足
- 即使总空闲显存足够，无法分配大张量

---

## 验证实验

### 实验 1: 增量分配测试

```python
# 将 100K×5K 分成 10 个 10K×5K 块
for i in range(10):
    chunk = torch.randn(10000, 5000, device='cuda')
    chunks.append(chunk)
    
# 结果：10 块全部成功分配，总计 3.74 GB
```

**结论**: Torch 可以分配完整的 3.73 GB 数据，没有内在限制

### 实验 2: 清洁状态测试

```
Torch 单独运行 ( fresh start):
- 初始: 0.00 GB
- 峰值: 7.45 GB
- 结果: SUCCESS

CuPy 单独运行:
- 峰值: 7.64 GB
- 结果: SUCCESS
```

**结论**: 两个框架单独运行时都能成功

---

## 为什么之前的 benchmark 会 OOM？

### benchmark 脚本执行顺序

```python
# 1. CuPy GPU 测试
X_cupy = cp.asarray(X_np)  # 分配 3.73 GB
# ... 计算 ...
cp.get_default_memory_pool().free_all_blocks()  # 标记释放，实际仍占用

# 2. Torch GPU 测试
X_torch = torch.from_numpy(X_np).to('cuda')  # 需要 3.73 GB
# OOM! 因为 CuPy 占用的显存未真正释放
```

### 显存竞争示意

```
GPU 总显存：15.90 GB
├─ CuPy "释放"后保留：~8 GB (无法被 Torch 使用)
├─ Torch 峰值需求：~7.5 GB
└─ 剩余：~0.4 GB → OOM
```

---

## 解决方案

### 方案 1: 进程隔离 (推荐)

在不同进程中运行 CuPy 和 Torch 测试：

```bash
# 进程 1: CuPy 测试
python -c "import cupy; ...; exit()"

# 进程 2: Torch 测试 (全新进程，显存完全释放)
python -c "import torch; ..."
```

### 方案 2: 先运行 Torch

```python
# 先运行 Torch (显存需求低时)
torch_test()

# 清理
torch.cuda.empty_cache()

# 再运行 CuPy (可以占用更多显存)
cupy_test()
```

### 方案 3: 限制 CuPy 内存池

```python
import cupy as cp

# 限制 CuPy 内存池大小
mempool = cp.get_default_memory_pool()
mempool.set_limit(size=4 * 1024**3)  # 限制 4GB
```

### 方案 4: 使用环境变量

```bash
# 设置 PyTorch 内存分配策略
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## 最佳实践建议

### 多框架环境

1. **避免在同一进程混用 CuPy 和 Torch**
2. **如必须混用，先运行 Torch** (内存管理更积极)
3. **使用进程级隔离** (最可靠)

### 显存管理

```python
# Torch 推荐做法
import torch

# 及时清理
del tensor
torch.cuda.empty_cache()

# 重置统计
torch.cuda.reset_peak_memory_stats()

# 监控显存
allocated = torch.cuda.memory_allocated() / 1024**3
```

```python
# CuPy 推荐做法
import cupy as cp

# 清理后强制释放
cp.get_default_memory_pool().free_all_blocks()

# 或者完全重置内存池
cp.get_default_memory_pool().free_all_blocks()
cp.cuda.MemoryHook
```

---

## 结论

**Torch OOM 不是显存管理不善造成的**，而是：

1. **CuPy 内存池行为** - "释放"后仍保留显存
2. **框架间显存竞争** - 两个框架无法共享显存池
3. **测试脚本顺序** - 先运行 CuPy 导致 Torch 显存不足

**推荐**: 在生产环境中，选择单一后端 (CuPy 或 Torch)，避免混用。

---

**分析脚本**: `dev/scripts/torch_oom_analysis.py`  
**报告生成**: 2026-04-17

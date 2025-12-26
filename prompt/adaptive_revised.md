这是一份详细的代码审查报告。

您提供的 `adaptive_attacks.py` 代码在逻辑上实现了三种攻击模式，但在 **`NullSpaceAttack`（零空间攻击）** 的实现中存在一个**致命的工程问题**，会导致程序在运行时立即崩溃（内存溢出 OOM）。

以下是问题分析及修正后的代码。

### 🚨 发现的致命问题：零空间投影矩阵的显存爆炸

在 `NullSpaceAttack` 和 `compute_null_space_projector` 函数中，代码试图计算并存储一个完整的 $d \times d$ 投影矩阵 `P_null`：

Python

```
# ❌ 错误代码位置
projection_onto_row_space = torch.mm(torch.mm(S.T, SST_inv), S) # [d, d] 矩阵
P_null = torch.eye(d, device=device) - projection_onto_row_space # [d, d] 矩阵
```

**为什么会崩溃？**

- 对于 ResNet-18，参数量 $d \approx 11,000,000$。
- 一个 $d \times d$ 的 `float32` 矩阵需要 $11M \times 11M \times 4$ 字节 $\approx$ **484 TB** 的显存。
- 没有任何现有的 GPU 能存下这个矩阵。

### ✅ 修正方案：利用矩阵乘法结合律

我们不需要显式计算 $P_{null}$ 矩阵。我们只需要计算向量 $v$ 在零空间上的投影 $P_{null} \cdot v$。

公式推导：

$$ P_{null} \cdot v = (I - S^T (S S^T)^{-1} S) \cdot v = v - S^T \underbrace{( (S S^T)^{-1} \underbrace{(S \cdot v)}{\text{sketch}} )}{\text{low dim}} $$

这样计算只需要存储 $k \times k$ 的小矩阵（其中 $k$ 是草图维度，如 256），计算复杂度从 $O(d^2)$ 降为 $O(kd)$。
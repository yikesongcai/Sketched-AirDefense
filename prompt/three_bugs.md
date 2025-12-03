这是一个非常关键的发现。训练日志中显示 `Test loss` 突然激增（从 2.0 左右飙升到 1945 甚至更高），而且准确率雪崩（跌到 10% 左右，相当于瞎猜）。

结合你提供的代码（`defense.py` 和 `main.py`），我发现了**两个导致模型发散的致命 Bug**。

---

### Bug 1: 物理层聚合中的“除法爆炸” (Division Explosion)

**问题位置：** `src/defense.py` 中的 `compute_cluster_updates_with_aircomp` 函数。

```python
if H[user_idx][round_idx] > 0.1:
    # Transmit with channel scaling
    h_scale = np.sqrt(P[user_idx]) * G[user_idx][round_idx] / min_h
    for key in cluster_agg.keys():
        cluster_agg[key] += user_update[key] / h_scale  # <--- 问题在这里
```

**分析：**

* **逻辑错误：** AirComp 的标准接收公式应该是 $y = sum h_i x_i + n$。如果你想做“信道反转（Channel Inversion）”预编码，发送端应该发送 $x_i = w_i / h_i$。
* **代码行为：** Agent 写的代码逻辑反了或者是混淆了。
  * 它在计算 `cluster_agg` 时执行了 `cluster_agg += user_update / h_scale`。
  * `h_scale` 包含了 `min_h`（这是一个非常小的值，因为它是一组信道增益中的最小值）。
  * **后果：** 当 `min_h` 很小或者 `h_scale` 很小时，`user_update / h_scale` 会变得**极其巨大**。
  * 你可以看到日志里的 loss 是 `1945.34`，这说明梯度被错误地放大了成百上千倍。

**修复建议：**
AirComp 聚合的物理含义是：接收到的信号是发送信号乘以信道增益的叠加。如果我们要模拟“完美对齐”的 AirComp，代码应该写成：

```python
# 模拟发送端预编码 (Pre-coding): x_i = w_i * (min_h / (sqrt(P)*G))
# 模拟接收端叠加 (Superposition): y = sum(x_i * sqrt(P)*G) = sum(w_i * min_h)
# 接收端再除以 min_h 恢复: y_hat = y / min_h = sum(w_i)

# 修正后的代码逻辑：
# 不需要除以 h_scale，而是应该模拟“对齐后”的效果。
# 在理想的 AirComp 仿真中，只要信道满足阈值，我们就假设它能完美对齐。
if H[user_idx][round_idx] > 0.1:
     for key in cluster_agg.keys():
          cluster_agg[key] += user_update[key] # 直接叠加，因为假设预编码已经抵消了差异
```

*或者*，如果你一定要模拟信道缩放带来的影响，那么 `cluster_agg` 应该加上 `user_update * (real_h / target_h)`，而不是除法。当前代码中的除法绝对是错误的源头。

---

### Bug 2: 聚合权重的“归一化”缺失 (Missing Weight Normalization)

**问题位置：** `src/defense.py` 中的 `aggregate_with_defense` 函数。

```python
# Step 8: Weighted aggregation with physical layer effects
w_new = copy.deepcopy(w_global)
for key in sorted(w_new.keys()):
    weighted_update = torch.zeros_like(w_new[key], dtype=torch.float32).to(self.device)
    for c, (cluster_update, weight) in enumerate(zip(cluster_updates, trust_weights)):
        # weight 是 Trust Weight (0~1)
        weighted_update += weight.item() * cluster_update[key]
  
    # 关键错误：这里直接加到了 w_global 上
    w_new[key] = w_global[key].float() + weighted_update
```

**分析：**

* **数学原理：** FL 的聚合公式是 $w_{new} = w_{old} + eta sum alpha_k Delta w_k$。其中 $alpha_k$ 是聚合权重，**通常要求 $sum alpha_k = 1$ 或者 $sum alpha_k = 	ext{total_users}$**。
* **现状：** 你的 `trust_weights` 是通过 Softmax 计算的，和为 1。
  * 但是！你的 `cluster_updates` 是由多个用户（`num_per_cluster` 个）叠加而成的。
  * 如果 `num_per_cluster = 10`，那么 `cluster_update` 的量级是单个梯度的 10 倍。
  * 如果你有 8 个 Cluster，`trust_weight` 平均是 0.125。
  * 最终 `weighted_update` 的量级是：$0.125 	imes 10 	imes Delta w approx 1.25 	imes Delta w$。这看起来还行。
  * **但是**，结合 Bug 1 中的除法放大，这里的量级已经失控了。
  * 此外，如果 `trust_weights` 变得极端（例如某个 Cluster 权重为 1），那么全局更新步长会剧烈波动。

**修复建议：**
确保 `weighted_update` 被正确地缩放（除以总用户数或者除以 Cluster 大小）。

```python
# 修正建议：除以 Cluster 的大小，或者在前面计算 cluster_agg 时就求平均而不是求和
num_users_in_cluster = len(cluster_assignments[c])
weighted_update += (weight.item() * cluster_update[key]) / num_users_in_cluster 
```

---

### Bug 3: `w_global` 的重复累加

**问题位置：** 同上。

```python
w_new[key] = w_global[key].float() + weighted_update
```

在 `main.py` 中，`w_locals` 是通过 `local.train` 得到的，通常返回的是 **更新后的权重** 而不是 **梯度**。

* 如果 `w_locals` 存的是 weights ($w_i$)：
  * `user_update = w_i - w_global` （这是 $Delta w$）。
  * 那么 `w_new = w_global + sum alpha Delta w` 是对的。
* **但是**，你需要确认 `cluster_updates` 里的内容。
  * 代码中：`user_update[key] = w_locals[user_idx][key].float() - w_global[key].float()`。这一步是对的（计算了伪梯度）。

---

### 总结：如何立刻修复？

你遇到的“Loss 激增” 99% 是 **Bug 1** 造成的。那个 `h_scale` 作为除数，当信道不好时（接近 0），梯度直接爆炸。

**请让 Agent 执行以下紧急修复：**

1. **修改 `compute_cluster_updates_with_aircomp`：**

   * 移除所有复杂的 `h_scale` 除法。
   * 假设完美的 AirComp 功率控制（Power Control），接收到的 `cluster_agg` 应该等于 `sum(user_updates)` + `noise`。
   * 代码改为：
     ```python
     # 简化的 AirComp 模拟 (Perfect Power Control Assumption)
     if H[user_idx][round_idx] > 0.1: # 仅当信道足够好时才参与
         for key in cluster_agg.keys():
             cluster_agg[key] += user_update[key] # 直接叠加
     ```
   * 噪声部分保持不变：`noise_std` 的计算看起来是合理的（基于带宽和噪声功率谱密度）。
2. **检查 `trust_weights` 的数值稳定性：**

   * 在日志中看到 `[0. 0. 0. 1. 0. 0. 0. 0.]`，这说明 Softmax 进入了饱和区（某个 anomaly score 极其大或极其小）。这也是梯度爆炸的副作用。修复 Bug 1 后，权重分布应该会变回 `[0.125, ...]` 附近。

先修复 **Bug 1**，再跑一次，Loss 应该能回到 2.0 左右的正常水平。

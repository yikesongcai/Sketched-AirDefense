这是一份深度定制的“自适应攻击”设计方案与代码实现提示词。

基于您的论文核心（Sketching + 时序预测）以及我刚刚对代码的审查，**常规的攻击（如 Label Flipping 或简单的 Gaussian Noise）已经不足以检验您防御的“底色”**。您需要设计专门针对“降维投影”和“时序检测”机制的**自适应攻击（Adaptive Attacks）**。

------

### 🛡️ 核心分析：为什么要设计这三种自适应攻击？

为了证明您的 `FedGTP` 是真的有效（而不只是在简单攻击下侥幸过关），您需要在论文中展示它在面对“高智商”对手时的表现。以下是针对您防御机制的三种“克星”模式：

#### 1. 零空间攻击 (Null-Space Attack) —— 专攻“草图压缩”

- **原理**：您的防御依赖于投影矩阵 $S$ 将高维梯度 $g$ 压缩为低维草图 $s = S \cdot g$。在线性代数中，从高维到低维的映射必然存在一个**零空间 (Null Space)**，即一组非零向量 $v$，满足 $S \cdot v = 0$。
- **攻击逻辑**：攻击者计算出位于零空间内的毒药向量 $v_{poison}$。
- **效果**：攻击者提交 $g_{attack} = g_{benign} + v_{poison}$。
  - **防御视角**：Server 计算草图 $S \cdot g_{attack} = S \cdot g_{benign} + 0$。草图完全正常，**时序预测误差为 0**，完美绕过检测。
  - **模型视角**：全局模型接收到了 $v_{poison}$，导致后门植入或精度下降。
- **检验价值**：这是基于 Sketch 防御的**理论极限**。如果您的防御能防住（例如依靠 AirComp 的信道噪声或动态 S 矩阵），这将是论文的重大亮点。

#### 2. 慢速温水煮青蛙 (Slow-Poisoning / Drift Mimicry) —— 专攻“时序预测”

- **原理**：您的 `TrajectoryPredictor` 允许一定范围内的“正常漂移”（Concept Drift）。
- **攻击逻辑**：攻击者不直接提交剧烈的恶意梯度，而是计算一个目标恶意方向 $g_{mal}$，然后每轮只向该方向偏离极小的一步 $\epsilon$。
- **效果**：每一轮的偏差都在预测器的容忍阈值（Threshold）之内。经过数百轮累积，模型最终被带偏。
- **检验价值**：验证您的“防御感知训练”（Defense-Aware Training）是否真的学到了“对漂移敏感”，而不仅仅是死记硬背历史。

#### 3. 预测器代理攻击 (Predictor-Proxy Attack) —— 专攻“阈值检测”

- **原理**：假设攻击者知道防御机制（白盒或灰盒假设）。
- **攻击逻辑**：攻击者在本地也训练一个 `TrajectoryPredictor`（代理模型），利用观测到的历史全局模型来“预判 Server 的预判”。
- **效果**：攻击者在生成恶意梯度后，先用自己的代理预测器测一下。如果发现 `Anomaly Score` 太高，就自动减小攻击幅度，直到刚好卡在阈值下方（$\text{Score} \approx \text{Threshold} - \delta$）。
- **检验价值**：这是最高级的自适应攻击，证明您的防御在“对手极其聪明”的情况下依然具备鲁棒性（或者逼迫对手只能产生微不足道的破坏）。

------

### 💻 给 Agent 的代码实现提示词 (Prompt)

请复制以下内容发送给您的 Agent。这段提示词要求它在一个独立的 `adaptive_attacks.py` 文件中实现上述逻辑，并保持与您现有数据结构的兼容性。

------

**Copy & Paste This Prompt:**

# Role

你是一位专注于联邦学习安全性的红队算法专家（Red Teaming Algorithm Expert）。你精通 PyTorch、线性代数（特别是矩阵投影理论）以及对抗性攻击设计。

# Context

我正在评估一个名为 FedGTP 的防御系统。该系统使用“梯度草图 (Gradient Sketching)”来压缩通信，并使用“时序轨迹预测 (Trajectory Prediction)”来检测异常更新。

防御系统的核心逻辑是：Server 维护一个随机投影矩阵 $S$ 和一个 LSTM 预测器。如果收到梯度的草图 $s_t$ 与预测值 $\hat{s}_t$ 差异过大，则判定为异常。

# Task

请编写一个名为 `adaptive_attacks.py` 的 Python 模块，实现针对该防御机制的三种高级**自适应攻击类**。代码必须与我现有的 PyTorch FL 框架兼容。

## 需要实现的攻击模式

### 1. NullSpaceAttack (零空间攻击)

- **目标**：利用投影矩阵 $S$ 的数学盲区。
- **实现逻辑**：
  1. 攻击者获取当前的投影矩阵 $S$（假设攻击者已知 $S$，或者 $S$ 是固定的）。
  2. 计算 $S$ 的零空间（Null Space）基向量，或者计算投影矩阵 $P_{null} = I - S^T (S S^T)^{-1} S$（假设 $S$ 行满秩）。
  3. 生成一个恶意目标梯度 $g_{mal}$（例如指向错误标签的梯度）。
  4. 将 $g_{mal}$ 投影到零空间：$noise = P_{null} \cdot g_{mal}$。
  5. 最终提交：$g_{final} = g_{benign} + \gamma \cdot noise$。
  6. **预期效果**：$S \cdot g_{final} \approx S \cdot g_{benign}$，Sketch 几乎无变化，但模型参数被修改。

### 2. SlowPoisoningAttack (慢速投毒/温水煮青蛙)

- **目标**：欺骗时序预测器，利用其对 Concept Drift 的容忍度。
- **实现逻辑**：
  1. 确定最终的恶意目标方向 $g_{target}$。
  2. 在每一轮 $t$，不直接发送 $g_{target}$。
  3. 计算插值：$g_{submit}^{(t)} = (1 - \alpha) \cdot g_{benign}^{(t)} + \alpha \cdot g_{target}$。
  4. 关键点：$\alpha$ 应该非常小（例如 0.05），或者动态衰减，确保每一步的变化都在 Server 的预测误差阈值内。

### 3. PredictorProxyAttack (预测器代理/梯度引导)

- **目标**：估算 Server 的检测阈值，卡在被发现的边缘进行最大化攻击。
- **实现逻辑**：
  1. 攻击者在本地维护一个 `LocalTrajectoryPredictor`（LSTM），结构与 Server 端一致。
  2. 利用过去几轮观测到的全局草图历史来训练这个本地预测器。
  3. 在生成恶意梯度时，是一个优化问题：
     - Maximize: `Damage(g)` (e.g., CrossEntropy on target label)
     - Subject to: `|| LocalPredictor(History) - Sketch(g) |

| < Threshold`4.  简化实现：先生成强攻击梯度，如果`LocalPredictor` 算出异常分过高，就利用二分法（Binary Search）或梯度裁剪（Clipping）缩小攻击幅度，直到满足阈值约束。

# Requirements

- **输入接口**：所有攻击类都应接收 `(benign_gradient, projection_matrix_S, target_model,...)` 等参数。
- **输出**：返回修改后的 `poisoned_gradient`（Flattened Tensor）。
- **工具函数**：请包含计算 Null Space 投影矩阵的辅助函数（注意处理大矩阵的内存效率，如果维度过大，使用伪逆或迭代法）。
- **注释**：请详细注释攻击原理，这将被用于论文的方法论部分。

请生成完整的 `adaptive_attacks.py` 代码。
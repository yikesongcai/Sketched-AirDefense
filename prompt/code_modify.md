这是一个非常明智的选择。直接在现有的代码库上“动大手术”确实容易引入 bug，而且 FedSAC 的原版代码结构（特别是 `strategy.py` 中的优化求解部分）比较复杂。将“核心逻辑”封装好，交给 AI Agent 去做具体的“填空”和“拼接”工作，效率会高得多。

为了让你的本地 AI Agent 能够一次性写出高质量、可运行的代码，我为你准备了 **一套完整的“指令包”**。

这套指令包包含：

1.  **核心逻辑代码（Core Logic Snippets）**：这是论文算法的“灵魂”，你需要直接提供给 Agent，防止它瞎写数学原理。
2.  **系统架构图（Architecture Prompts）**：告诉 Agent 修改哪里，怎么修改。
3.  **最终提示词（The Master Prompt）**：你可以直接复制这段话给你的 Agent。

-----

### 第一部分：核心逻辑代码 (Core Logic Snippets)

*请将这部分代码保存或直接贴给 Agent，告诉它：“这是 `src/defense.py` 的核心实现，请基于此进行封装。”*

这段代码实现了我们确定的 **“草图压缩 (Sketching)”** 和 **“时序预测 (Trajectory Prediction)”** 功能。

```python
import torch
import torch.nn as nn
import numpy as np

class GradientSketcher:
    """
    负责生成随机投影矩阵 S，并将高维梯度压缩为低维草图。
    物理含义：模拟 AirComp 传输中的压缩感知或特征提取。
    """
    def __init__(self, input_dim, sketch_dim, device):
        self.device = device
        self.input_dim = input_dim
        self.sketch_dim = sketch_dim
        # 固定随机矩阵 S，保证每一轮的投影方式一致
        # 使用正态分布初始化，满足 JL Lemma 的等距保线性质
        self.S = torch.randn(input_dim, sketch_dim).to(device) / (sketch_dim ** 0.5)

    def sketch(self, update_vector):
        """
        Input: update_vector (Tensor, Flattened) [d]
        Output: sketched_vector (Tensor) [sketch_dim]
        """
        # 简单的线性投影: s = x * S
        return torch.matmul(update_vector, self.S)

class TrajectoryPredictor(nn.Module):
    """
    自监督时序预测器：输入过去 W 轮的草图，预测下一轮草图。
    """
    def __init__(self, sketch_dim, hidden_dim, num_layers=1):
        super(TrajectoryPredictor, self).__init__()
        # 使用 GRU 或 LSTM 处理时序数据
        self.rnn = nn.GRU(input_size=sketch_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=num_layers, 
                          batch_first=True)
        # 输出层预测下一轮的草图
        self.fc = nn.Linear(hidden_dim, sketch_dim)

    def forward(self, history):
        """
        history: [batch_size, window_size, sketch_dim] (即每个 Cluster 的历史轨迹)
        """
        out, _ = self.rnn(history)
        # 取最后一个时间步的输出
        last_step = out[:, -1, :] 
        prediction = self.fc(last_step)
        return prediction

def compute_anomaly_score(predicted_sketch, actual_sketch):
    """
    计算预测误差作为异常分数
    """
    # MSE Loss: ||s_pred - s_real||^2
    error = torch.norm(predicted_sketch - actual_sketch, dim=1) ** 2
    return error
```

-----

### 第二部分：给本地 AI Agent 的完整提示词 (Master Prompt)

你可以直接复制以下内容（包含中文指令和英文关键词，以便 Agent 更好理解代码上下文）发送给你的 AI Agent：

**[角色设定]**
你是一个精通联邦学习（Federated Learning）、空中计算（AirComp）和 PyTorch 开发的资深算法工程师。

**[任务目标]**
我现有一个基于 **FedSAC** 的代码库（包括 `main.py`, `src/update.py`, `src/strategy.py` 等）。
请帮助我修改这个代码库，实现一种名为 **"Sketched-AirDefense"** 的新防御机制。
该机制不依赖于传统的分类器，而是利用 **梯度草图（Gradient Sketching）** 和 **自监督时序轨迹预测（Self-Supervised Trajectory Prediction）** 来检测拜占庭攻击。

**[已有代码结构分析]**

  * `src/strategy.py`: 包含了聚合策略。目前有 `FedAvg`, `FedAvg_Byzantine`, `Secure_aggregation`。
  * `main.py`: 主训练循环。
  * `utils/options.py`: 参数配置。

**[具体修改步骤要求]**

**1. 新增参数 (`utils/options.py`)**
请添加以下参数：

  * `--sketch_dim`: int, default=128, 草图压缩后的维度。
  * `--pred_hidden`: int, default=64, 预测器的隐藏层维度。
  * `--window_size`: int, default=5, 时序预测的历史窗口大小。
  * `--defense_method`: str, default='sketched', 选择防御模式。

**2. 创建新模块 (`src/defense.py`)**
请基于我提供的“核心逻辑代码”，创建 `src/defense.py` 文件。

  * 包含 `GradientSketcher` 类。
  * 包含 `TrajectoryPredictor` 类。
  * 包含一个辅助函数 `get_model_flattened_dim(net)` 用于自动计算模型参数总数。

**3. 修改主循环 (`main.py`)**
这是最关键的部分。请重构训练循环逻辑，具体逻辑如下：

  * **初始化阶段**：
      * 在 `net_glob` 初始化后，实例化 `GradientSketcher` 和 `TrajectoryPredictor`。
      * 定义 `predictor_optimizer` (e.g., Adam)。
      * 初始化一个 `history_buffer` 队列，用于存储过去 `window_size` 轮每个 Cluster 的聚合草图。
  * **训练循环 (`for iter in trange(args.rounds)`)**：
      * **Step 1: 模拟 AirComp 分组聚合**
          * 使用 `args.num_cluster` 将用户随机或按规则分成 K 组。
          * 对每一组 $k$，计算组内用户的模型更新之和：$\Delta w_{cluster\_k} = \sum_{i \in Cluster_k} \Delta w_i$。
          * *(注意：在代码中我们要先算出这个 Cluster Aggregate，模拟服务器收到的 AirComp 信号)*。
      * **Step 2: 草图提取与预测**
          * 将每个 Cluster 的 $\Delta w_{cluster\_k}$ 展平，并通过 `GradientSketcher` 得到草图 $s_k$。
          * 如果 `history_buffer` 长度不足 `window_size`，则跳过防御，默认全信任。
          * 如果长度足够，将历史草图输入 `TrajectoryPredictor`，得到预测值 $\hat{s}_k$。
      * **Step 3: 异常检测与权重调整**
          * 计算误差 $err_k = ||s_k - \hat{s}_k||^2$。
          * 基于 $err_k$ 计算信任权重 $\alpha_k$。简单逻辑：`scores = softmax(-err_k)`, 或者将误差最大的 20% Cluster 权重置 0。
      * **Step 4: 自监督更新**
          * 计算预测器的 Loss (MSE between $\hat{s}_k$ and $s_k$)。
          * 执行 `predictor_optimizer.step()` 更新预测器。
          * 将当前真实的 $s_k$ 加入 `history_buffer`，移除最旧的。
      * **Step 5: 全局聚合**
          * 利用 Step 3 计算出的安全权重 $\alpha_k$，对各组的更新进行加权平均：$w_{global}^{t+1} = w_{global}^t + \sum \alpha_k \Delta w_{cluster\_k}$。

**4. 兼容性提示**

  * 请保留原有的 `generate_clients` 和数据加载逻辑。
  * 不要删除 `src/strategy.py` 中的原有代码，可以通过导入或新增函数的方式扩展。

**[输出格式]**
请直接给出修改后的 `src/defense.py` 完整代码，以及 `main.py` 中**训练循环部分**的完整代码块。

-----

### 第三部分：给 Agent 的补充“思维链”（可选）

*如果 Agent 写出的代码逻辑有问题，可以将这段逻辑发给它进行修正。*

> **关于 `main.py` 逻辑的补充说明：**
> 在 `main.py` 中，我们现在的逻辑是：
>
> 1.  先收集所有 `w_locals`（模拟所有设备本地训练完成）。
> 2.  **模拟 AirComp 过程**：不要直接把所有 `w_locals` 扔进 `FedAvg`。必须先按 Cluster 聚合。
>
> <!-- end list -->

> ```python
> # 伪代码逻辑
> cluster_updates = []
> for k in range(num_clusters):
>     cluster_users = get_users_in_cluster(k)
>     # 模拟物理层叠加
>     cluster_agg_update = sum([w_locals[u] - w_global for u in cluster_users]) 
>     cluster_updates.append(cluster_agg_update)
> ```
>
> 3.  然后对 `cluster_updates` 应用防御。
> 4.  最后再更新全局模型。这样才符合 AirComp 的物理特性。

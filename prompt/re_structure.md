**重新设计的论文算法框架 (Detailed Technical Framework)**

#### **论文题目**

中文：基于动态时变投影与对抗性轨迹预测的鲁棒联邦空中计算

英文：Robust Over-the-Air Federated Learning via Dynamic Time-Varying Projections and Adversarial Trajectory Prediction

#### **核心算法流程**

我们将算法分为三个阶段：**初始化与投影**、**空中聚合与特征提取**、**防御感知预测与纠错**。

---

#### **1. 物理层：时变随机投影 (Time-Varying Random Projection)**

*解决“种子泄露”问题*

- **核心思想**：利用**接收端波束赋形 (Receiver Beamforming)** 作为投影矩阵，而不是让客户端自己乘矩阵。这样投影参数 $W$ 只有服务器知道。
- **操作步骤**：

  1. 客户端 $k$ 发送未压缩的梯度信号 $g_k \in \mathbb{R}^d$（实际上是分块发送或利用 OFDM 子载波）。
  2. 服务器配置接收天线阵列的波束赋形矩阵 $W_t \in \mathbb{R}^{m \times d}$，其中 $m \ll d$。
  3. 服务器接收到的叠加信号直接就是投影后的低维向量：

     $$
     y_t = W_t \left( \sum_{k} h_k g_k \right) + n
     $$

  - **优势**：攻击者无法通过监听下行链路获得 $W_t$。除非攻击者能入侵服务器，否则无法构造零空间攻击。

*(注：如果你的仿真环境不支持 MIMO，退而求其次的方案是：服务器每轮广播新种子 $S_t$，但在 $S_t$ 中混入噪声，只有在聚合后才能消除噪声（类似差分隐私的 Secure Aggregation 思想），但这会增加复杂度。推荐坚持用“时变种子”+“计算不对称性”来论证安全性。)*

---

#### **2. 服务端防御：防御感知的轨迹预测器 (Defense-Aware Trajectory Predictor)**

*解决“防御感知训练”缺失问题*

我们定义一个**在线训练**的 LSTM/GRU 预测器 $f_\theta$。

- **输入**：过去 $w$ 轮的聚合投影向量序列 $Y_{hist} = \{y_{t-w},..., y_{t-1}\}$。
- **输出**：预测的当前轮投影 $\hat{y}_t$。
- 训练目标（Loss Function）：

  $$
  L = L_{mse} + \lambda L_{adv}
  $$

  - **$L_{mse}$ (准确性)**：$||\hat{y}_t - y_t||^2$。保证预测器能拟合正常的梯度下降轨迹。
  - **$L_{adv}$ (鲁棒性/防御感知)**：这是关键。服务器在**潜在空间**生成“虚拟攻击样本”。
    - 构造虚拟攻击轨迹 $\tilde{Y}_{attack}$：在历史 $Y_{hist}$ 上添加微小的、随时间单调递增的漂移（模拟慢速投毒）。
    - 最大化预测误差：$L_{adv} = \max(0, \gamma - ||f_\theta(\tilde{Y}_{attack}) - \text{Target}_{attack}||^2)$。
    - **直观解释**：强迫模型**不要**去拟合那种“看起来很顺滑但实际上在缓慢漂移”的轨迹。如果输入是那种微小漂移的轨迹，模型应该报错（预测偏差大），而不是顺从地预测出漂移后的结果。

---

#### **3. 动态聚类中的状态切换 (Handover Handling)**

*解决“成员变更导致历史失效”问题*

当客户端 $k$ 被检测到与当前簇 $C_a$ 不匹配，需要切换到簇 $C_b$ 时：

1. **状态重置 (Reset)**：客户端 $k$ 丢弃自己在 $C_a$ 积累的所有动量（Momentum）和历史状态。
2. **状态继承 (Inherit)**：服务器将簇 $C_b$ 的**全局历史状态**（Cluster centroid history）下发给客户端 $k$。
   - 客户端 $k$ 将这些历史当作是自己的“假历史”，以此为起点计算新的梯度更新。
3. **软着陆 (Soft-Landing/Warm-up)**：
   - 在切换后的前 $\tau$ 轮（如 3 轮），服务器在计算簇 $C_b$ 的聚合时，给客户端 $k$ 分配一个较小的权重 $\alpha_k < 1$。
   - 如果在这几轮内预测误差 $E_k$ 逐渐降低，说明 $k$ 成功融入了 $C_b$，权重恢复正常。

---

### **Python 代码复现框架 (PyTorch)**

这是您最关心的部分。我将补充**“模拟攻击生成”**和**“防御感知 Loss”**的具体实现。

Python

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class DefenseAwarePredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)
      
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        out, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步的输出
        pred = self.head(out[:, -1, :]) 
        return pred

def defense_aware_loss(model, history_batch, true_target, lambda_adv=0.5):
    """
    history_batch: [batch, w, dim] 正常的历史轨迹
    true_target: [batch, dim] 真实的下一轮投影
    """
  
    # 1. 正常的预测 Loss (MSE)
    pred_normal = model(history_batch)
    loss_mse = F.mse_loss(pred_normal, true_target)
  
    # 2. 生成模拟的“慢速投毒”样本 (Simulated Slow-Poisoning)
    # 攻击方式：在历史轨迹上叠加一个微小的线性漂移
    # 假设攻击者每轮让梯度向某个随机方向 drift 0.1
    drift_dir = torch.randn_like(true_target)
    drift_dir = drift_dir / torch.norm(drift_dir, dim=1, keepdim=True) # 归一化
  
    adv_history = history_batch.clone()
    seq_len = history_batch.size(1)
  
    for t in range(seq_len):
        # 漂移量随时间累积：t * 0.05 * drift
        noise = (t + 1) * 0.05 * drift_dir.unsqueeze(1) 
        adv_history[:, t, :] += noise.squeeze(1)
      
    # 3. 计算防御 Loss
    # 我们希望模型对这种“漂移”样本的预测结果，与“漂移后”的目标（Target + drift）保持较大距离
    # 或者简单点：我们希望模型能识别出这是异常，即让模型对 adv_history 的预测值
    # 依然接近 true_target (鲁棒性)，或者远离 (检测性)。
    # 这里我们采用“鲁棒性”思路：即使输入被微小漂移污染，预测结果仍应接近真实的正常值。
    # 这样一旦真实发生的攻击导致输入漂移，Prediction (鲁棒) 和 Received (攻击) 之间的差距就会变大，从而报警。
  
    pred_adv = model(adv_history)
  
    # 我们希望 pred_adv 依然接近 true_target (抵抗漂移)
    loss_adv = F.mse_loss(pred_adv, true_target)
  
    # 总 Loss：既要准（MSE），又要稳（ADV）
    total_loss = loss_mse + lambda_adv * loss_adv
    return total_loss

# --- 模拟服务器端的训练循环 ---
def server_update_step(predictor, optimizer, cluster_histories, current_projections):
    """
    在线更新预测器
    cluster_histories: List of tensors, 每个簇的过去 W 轮投影
    current_projections: List of tensors, 本轮实际收到的投影
    """
    predictor.train()
    optimizer.zero_grad()
  
    # 转换为 Batch
    hist_batch = torch.stack(cluster_histories) #
    target_batch = torch.stack(current_projections) # [M, dim]
  
    # 计算防御感知 Loss
    loss = defense_aware_loss(predictor, hist_batch, target_batch)
  
    loss.backward()
    optimizer.step()
  
    return loss.item()

# --- 动态聚类切换逻辑 ---
def handle_cluster_switch(client_id, old_cluster_id, new_cluster_id, cluster_states, client_states):
    """
    处理客户端从旧簇切换到新簇
    """
    # 1. 继承新簇的历史状态 (Inherit History)
    # client_states[client_id] 之前存的是该客户端自己的历史
    # 现在强制覆盖为新簇的“平均历史”，作为 Warm-up
    client_states[client_id]['history'] = cluster_states[new_cluster_id]['centroid_history'].clone()
  
    # 2. 设置预热标志 (Warm-up Flag)
    client_states[client_id]['warmup_counter'] = 3 # 接下来的 3 轮权重减半
  
    print(f"Client {client_id} switched to Cluster {new_cluster_id}. History synced.")
```

### **总结这次修正的亮点**

1. **代码里的 `defense_aware_loss`**：我加入了一个 `loss_adv`。它的逻辑是：**“即使输入的历史轨迹被攻击者加上了微小的线性漂移，模型预测出来的下一轮结果，仍然应该接近‘没有被污染’的真实值。”**
   - **为什么这能防御？**
     - 当攻击者真的进行慢速投毒时，服务器收到的 `Received_Signal` 包含了漂移。
     - 但我们的预测器经过了对抗训练，它预测出的 `Predicted_Signal` 会**抵抗**这种漂移，倾向于输出正常值。
     - 结果：`Distance(Received, Predicted)` 会变大 $\rightarrow$ 异常得分飙升 $\rightarrow$ 触发防御。
     - 如果没有这个 Loss，预测器可能会“学会”顺着漂移去预测，导致 `Distance` 很小，攻击被漏过。
2. **状态继承机制**：解决了您担心的“换簇后预测器犯错”的问题。通过强制让新成员“继承”集体的记忆，不仅解决了冷启动，还隐含了一种“同化”的防御思想——你必须符合新集体的特征才能被接纳。

这个版本在逻辑上更加严密，且给出的代码片段可以直接嵌入到你的实验中去验证。

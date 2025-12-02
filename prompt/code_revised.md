-----

### 一、 核心逻辑漏洞 (Critical Logic Issues)

#### 1\. 严重的“数据泄露”问题 (Data Leakage in Aggregation)

  * **问题位置：** `src/defense.py` 中的 `aggregate_with_defense` 函数。
  * **现象：**
    ```python
    # Step 7: Weighted aggregation
    w_new = copy.deepcopy(w_global)
    # ...
    w_new[key] = w_global[key].float() + weighted_update
    ```
    这里 Agent 直接拿 `w_global` 加上了 `weighted_update`。
  * **致命错误：** AirComp 的核心是**模拟信号叠加**。在 FedSAC 原版（`Secure_aggregation`）中，引入了**信道增益 (G)**、**发射功率 (P)** 和 **噪声 (Noise)**。Agent 的实现完全忽略了物理层信道模型，直接做了数字加权平均。这变成了一个普通的 FL 算法，而不是 AirComp FL。
  * **后果：** 你的论文题目是 "Over-the-Air"，但代码里没有 Air（信道）。这会被审稿人直接拒稿。
  * **修复方案：** 必须把 FedSAC 原版 `Secure_aggregation` 中的信道加噪逻辑搬过来，或者调用原版的聚合函数，只用 defense 模块计算出的 `trust_weights` 去替换原版中的 `alpha`。

#### 2\. 预测器训练的“自循环”陷阱 (Self-Loop in Training)

  * **问题位置：** `update_predictor` 在 `aggregate_with_defense` 中被调用。
  * **现象：**
    ```python
    # Step 5: Update predictor
    self.update_predictor(predictor_loss)
    ```
  * **逻辑漏洞：** `predictor_loss` 是基于 `current_sketches` 计算的。如果我们在**同一轮**中，先用 predictor 预测当前轮，算出 loss，然后立刻更新 predictor，这就相当于在测试集上训练。虽然这里是自监督，但在异常检测中，如果你立刻学会了当前的“异常模式”，下一轮它就被当成正常了。
  * **后果：** 预测器会迅速“适应”攻击，导致漏检（False Negative）。
  * **修复方案：** 应该**先**预测并检测（Detect），**后**更新历史（Update History），且 Predictor 的更新最好有一定的**延迟**或者只在“判定为正常”的样本上更新（避免学习到攻击者的模式）。

#### 3\. 维度不匹配风险 (Dimension Mismatch)

  * **问题位置：** `flatten_model_updates`。
  * **现象：** 代码直接 `cat` 了所有参数。
  * **风险：** 必须确保 `flatten` 的顺序和 `sketch` 的输入维度完全一致。虽然代码里看起来是按 keys 顺序遍历的，但在不同 Python 版本或不同设备上，`state_dict.keys()` 的顺序有时不保证。
  * **修复方案：** 强制对 keys 进行排序：`for key in sorted(w_update.keys()): ...`。

-----

### 二、 与 FedSAC 原版代码的兼容性问题

#### 1\. 参数冗余与冲突

  * **现象：** `utils/options.py` 中，Agent 添加了 `--num_cluster`。但 FedSAC 原版里可能已经有了或者逻辑不同。
  * **检查：** FedSAC 原版确实有 `--num_cluster`。Agent 再次添加没有问题，但要确保 `main.py` 传递参数时不要覆盖了原版逻辑中对 `num_cluster` 的特殊处理（比如原版可能有动态调整簇数量的逻辑）。

#### 2\. `main.py` 中的聚合调用

  * **问题位置：** `main.py`。
  * **现象：**
    ```python
    if args.defense_method == 'sketched':
        w_glob, defense_info = Sketched_Defense_Aggregation(...)
    elif args.trans == 'proposed':
        w_glob, ... = Secure_aggregation(...)
    ```
  * **缺失：** `Secure_aggregation` 返回了 `reputation` 和 `q`（用于下一轮的聚类）。但 `Sketched_Defense_Aggregation` 只返回了 `w_new`。
  * **后果：** 如果你不返回 `reputation`，下一轮 `generate_clients` 或聚类逻辑就会因为缺少状态变量而报错，或者退化为随机聚类。
  * **修复方案：** `Sketched_Defense_Aggregation` 必须维护并返回 `reputation`（基于 `trust_weights` 更新），以保持与主循环的接口兼容。

-----

### 三、 具体的代码修复指令 (给 Agent 的反馈)

你可以直接把下面这段话发给你的 Agent，让它进行第二轮修改：

**[Code Refinement Instructions]**

Agent，你的代码框架很好，但为了符合 "Over-the-Air Computation" 的物理设定并修复逻辑漏洞，请执行以下关键修改：

1.  **物理层信道模拟 (Critical):**
    在 `src/defense.py` 的 `compute_cluster_updates` 函数中，不能简单地求和。你需要模拟 AirComp 信道：

      * 引入参数 `H` (Channel Gain), `G` (Small-scale fading), `P` (Transmit Power), `N0` (Noise Power)。
      * 模拟公式：`cluster_agg = (sum(h_i * w_i) + noise) / h_scaling`。
      * **简化方案：** 为了不重写太复杂的物理层，请修改 `aggregate_with_defense`，让它接受 `H, G, P` 等物理层参数（从 `main.py` 传入）。在计算 `weighted_update` 时，对每个 Cluster 的聚合结果添加高斯噪声 `torch.randn_like(cluster_agg) * noise_std`，其中 `noise_std` 根据 `args.power` 和 `args.num_users` 计算（参考 `src/strategy.py` 中的 `FedAvg_Byzantine` 噪声计算逻辑）。

2.  **防止攻击适应 (Security):**
    在 `aggregate_with_defense` 中，修改 `update_predictor` 的逻辑：

      * **仅当** `anomaly_scores` 低于某个阈值（例如中位数）时，才用该 Cluster 的数据去更新 Predictor。
      * 如果某个 Cluster 被判定为异常（Trust Weight 很低），**绝对不要**用它来训练 Predictor，否则 Predictor 会学会“攻击是正常的”。

3.  **保持接口兼容 (Interface):**
    修改 `Sketched_Defense_Aggregation` 的返回值。

      * 它需要接受并返回 `reputation` 和 `q`（与 `Secure_aggregation` 保持一致）。
      * 在函数内部，根据 `trust_weights` 更新 `reputation`：`reputation[user_in_cluster_k] += trust_weights[k]`。这样主循环中的逻辑才能延续。

4.  **确定性 Flatten:**
    在 `flatten_model_updates` 中，使用 `sorted(w_update.keys())` 确保参数拼接顺序永远固定。

5.  **Main.py 传参修正:**
    在 `main.py` 调用 `Sketched_Defense_Aggregation` 时，补全缺失的参数：`distance, P, H, G, reputation, q` 等，确保物理层模拟能跑通。

-----

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本项目实现了论文 **Byzantine-Resilient Over-the-Air Federated Learning under Zero-Trust Architecture** 的算法。这是一个基于空中计算（Over-the-Air Computation, AirComp）的联邦学习系统，具备拜占庭攻击防御能力。

核心防御机制 **Sketched-AirDefense** 结合了：
- 梯度草图（Gradient Sketching）：使用随机投影压缩高维梯度
- 轨迹预测（Trajectory Prediction）：使用 GRU/LSTM 预测正常梯度轨迹
- 异常检测：通过预测误差识别恶意客户端

## 常用命令

```bash
# 运行实验（使用 Sketched-AirDefense 防御）
python main.py --defense_method sketched --attack omni --num_byz 6 --debug

# 运行基准实验（无防御）
python main.py --defense_method none --trans byzantine --attack omni

# 运行自适应攻击测试
python main.py --defense_method sketched --attack null_space --attack_strength 1.0
python main.py --defense_method sketched --attack slow_poison --poison_alpha 0.05
python main.py --defense_method sketched --attack predictor_proxy

# 使用 TensorBoard 查看训练曲线
python main.py --tsboard --debug
tensorboard --logdir=runs
```

## 核心架构

### 主要模块关系

```
main.py                     # 训练入口，数据集加载，训练循环
    ├── src/defense.py      # Sketched-AirDefense 防御核心
    │   ├── GradientSketcher        # 梯度草图器（随机投影）
    │   ├── TrajectoryPredictor     # GRU 轨迹预测器
    │   ├── DefenseAwarePredictor   # LSTM 防御感知预测器（含对抗训练）
    │   └── SketchedAirDefense      # 主防御类（整合所有组件）
    ├── src/adaptive_attacks.py     # 三种自适应攻击实现
    │   ├── NullSpaceAttack         # 零空间攻击（利用投影盲区）
    │   ├── SlowPoisoningAttack     # 慢速投毒（温水煮青蛙）
    │   └── PredictorProxyAttack    # 预测器代理攻击
    ├── src/strategy.py     # 聚合策略
    │   ├── FedAvg                  # 标准联邦平均
    │   ├── FedAvg_Byzantine        # 带拜占庭攻击的聚合
    │   └── Secure_aggregation      # 安全聚合（proposed方法）
    ├── src/nets.py         # 神经网络模型（MLP, CNN_v1, CNN_v2）
    └── src/update.py       # 本地模型更新（ModelUpdate类）
```

### 关键参数说明

**防御相关**：
- `--defense_method`: 防御方法选择（`sketched` 或 `none`）
- `--sketch_dim`: 草图维度（默认128）
- `--window_size`: 轨迹预测窗口大小（默认5）
- `--num_cluster`: 聚类数量（默认8）
- `--anomaly_threshold`: 异常阈值（0-1）

**攻击相关**：
- `--attack`: 攻击类型（`omni`, `gaussian`, `null_space`, `slow_poison`, `predictor_proxy`）
- `--num_byz`: 拜占庭节点数量
- `--attack_strength`: 零空间攻击强度
- `--poison_alpha`: 慢速投毒插值因子

**聚合方式**：
- `--trans`: 传输/聚合方式（`proposed`, `byzantine`, 或其他触发标准 FedAvg）

## 数据流

1. `main.py` 加载数据集（MNIST/CIFAR），按 IID 或 Non-IID 分配给各客户端
2. 每轮训练中，所有客户端执行本地更新（`ModelUpdate.train()`）
3. 如果启用 `sketched` 防御：
   - `SketchedAirDefense.apply_adaptive_attacks()` 对拜占庭节点应用攻击
   - `compute_cluster_updates_with_aircomp()` 模拟物理层 AirComp 信号叠加
   - `extract_sketches()` 提取各聚类的梯度草图
   - `predict_and_detect()` 使用轨迹预测器检测异常
   - 根据信任权重加权聚合
4. 否则使用 `FedAvg_Byzantine` 或 `Secure_aggregation`

## 注意事项

- 数据集路径硬编码在 `main.py:55-78`，需根据本地环境修改
- `flatten_model_updates` 和 `unflatten_model_updates` 使用 `sorted(keys())` 保证跨环境一致性
- 物理层模拟中 `N0=0` 表示无噪声场景，可调整测试噪声鲁棒性
- 优化求解依赖 `cvxpy` 和 MOSEK 求解器（在 `strategy.py` 中）

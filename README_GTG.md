# PASS + GTG-Shapley 集成方案

## 概述

本项目将 GTG-Shapley 贡献评估方法集成到 PASS 联邦学习框架中，实现更公平、更准确的客户端贡献评估和搭便车者检测。

## 核心思想

### GTG-Shapley 算法

基于论文: [GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation in Federated Learning](https://dl.acm.org/doi/10.1145/3501811) (ACM TIST 2022)

**关键特性**：
1. **梯度重构**: 从梯度更新重构模型参数，避免重复训练
2. **Shapley 值计算**: 使用 Monte Carlo 采样估计每个客户端的边际贡献
3. **引导截断采样**: Within-round 和 Between-round 截断减少计算量

### 集成架构

```
流程：
1. 训练阶段：客户端训练并返回梯度更新（而非参数）
2. 服务器聚合：从梯度重构并聚合全局模型
3. 审计阶段：服务器广播所有梯度给客户端
4. 客户端互评：每个客户端使用 GTG-Shapley 评估其他客户端
5. 贡献更新：服务器聚合 Shapley 值并更新贡献分数
6. 搭便车者剔除：低于阈值的客户端被标记
```

## 文件结构

```
flwr-pass/
├── passexample/
│   ├── gtg_shapley.py          # GTG-Shapley 核心算法实现
│   ├── client_app_gtg.py       # 客户端（GTG-Shapley 版本）
│   ├── server_app_gtg.py       # 服务器端（GTG-Shapley 版本）
│   ├── task.py                 # 训练、测试、数据加载等工具函数
│   └── ...
├── pyproject_gtg.toml          # GTG-Shapley 配置文件
├── run_gtg.sh                  # 运行脚本
└── README_GTG.md               # 本文档
```

## 核心模块说明

### 1. GTG-Shapley 核心算法 (`gtg_shapley.py`)

**主要类**：
- `GTGShapley`: GTG-Shapley 计算器
  - `reconstruct_model_from_gradient()`: 从梯度重构模型参数
  - `aggregate_gradients()`: 聚合多个客户端的梯度
  - `evaluate_model()`: 评估模型性能
  - `compute_marginal_contribution()`: 计算边际贡献
  - `compute_shapley_value_with_truncation()`: 使用截断采样计算 Shapley 值
  - `evaluate_all_clients()`: 评估所有客户端的 Shapley 值

**主要函数**：
- `compute_gradient_from_params()`: 从参数更新计算梯度

### 2. 客户端 (`client_app_gtg.py`)

**训练阶段**：
- 正常客户端：训练并计算梯度
- AFR 客户端：生成随机高斯噪声作为梯度
- 返回梯度而非参数更新

**审计阶段**：
- 接收所有客户端的梯度
- 使用 GTG-Shapley 计算每个客户端的 Shapley 值
- 返回 Shapley 评分

### 3. 服务器端 (`server_app_gtg.py`)

**训练阶段**：
- 收集所有客户端的梯度
- 聚合梯度并重构全局模型

**审计阶段**：
- 广播所有梯度给客户端
- 收集客户端返回的 Shapley 值
- 聚合 Shapley 值并更新贡献分数

**贡献分数更新**：
```python
normalized_shapley = 1.0 / (1.0 + exp(-mean_shapley * 10))
new_score = α × old_score + (1-α) × normalized_shapley
```

## 配置参数

### 联邦学习参数
- `num-server-rounds`: 训练轮数（默认 50）
- `local-epochs`: 本地训练轮数（默认 5）
- `learning-rate`: 学习率（默认 0.01）
- `batch-size`: 批次大小（默认 64）
- `dataset`: 数据集（默认 cifar10）

### PASS 参数
- `alpha`: 贡献分数移动平均系数（默认 0.8）
- `beta`: 阈值系数（默认 1.75）
- 阈值计算：`threshold = 1/(β × N)`

### GTG-Shapley 参数
- `gtg-sampling-rounds`: Monte Carlo 采样轮数（默认 10）
- `gtg-within-threshold`: Within-round 截断阈值（默认 0.01）
- `gtg-between-threshold`: Between-round 截断阈值（默认 0.05）

## 运行方法

### 1. 使用运行脚本
```bash
cd /data/users/c407_2025/lj/flwr-pass
./run_gtg.sh
```

### 2. 直接运行
```bash
flwr run . --run-config pyproject_gtg.toml
```

### 3. 查看日志
```bash
tail -f pass_gtg_output.log
```

## 搭便车者设置

当前配置：
- **AFR 客户端**: Client 2（生成随机高斯噪声）
- **SFR 客户端**: 禁用（设置为 -1）

修改搭便车者：编辑 `passexample/client_app_gtg.py`
```python
AFR_CLIENT_ID = 2  # 修改为其他客户端ID
SFR_CLIENT_ID = -1  # 启用 SFR 设置为有效ID
```

## 预期效果

### 1. 准确率提升
- 初始准确率：~10%（随机猜测）
- 训练后准确率：50-70%（CIFAR-10 正常水平）

### 2. AFR 检测
- AFR 客户端的 Shapley 值应该显著低于正常客户端
- AFR 客户端的贡献分数应该逐渐降低
- 当贡献分数低于阈值时，AFR 被标记为搭便车者

### 3. 日志示例
```
[Round 5] GTG-Shapley Values:
  Client 0: mean = 0.025000, std = 0.003000
  Client 1: mean = 0.023000, std = 0.002500
  Client 2: mean = -0.010000, std = 0.005000  # AFR - 负贡献
  Client 3: mean = 0.024000, std = 0.002800
  ...

[Round 5] Contribution Scores (threshold=0.0571):
  Client 0: score = 0.8500
  Client 1: score = 0.8400
  Client 2: score = 0.3200 ⚠️ ELIMINATED  # AFR 被检测
  Client 3: score = 0.8450
  ...
```

## 与原始 PASS 的对比

| 特性 | 原始 PASS (AccDiv) | PASS + GTG-Shapley |
|------|-------------------|-------------------|
| 贡献评估方法 | 准确率差异 (AccDiv) | Shapley 值 |
| 计算位置 | 客户端互评 | 客户端互评 |
| 传输内容 | 参数更新 | 梯度更新 |
| 理论保障 | 启发式 | 博弈论公理 |
| 公平性 | 较好 | 更好 |
| 计算复杂度 | 低 | 中等（有截断优化） |

## 优势

1. **更公平**: Shapley 值满足对称性、虚拟性、可加性等公理
2. **更准确**: 能精确量化每个客户端的边际贡献
3. **理论保障**: 基于博弈论的公平分配方法
4. **高效计算**: 引导截断采样减少计算量

## 注意事项

1. **计算资源**: GTG-Shapley 需要更多计算资源（Monte Carlo 采样）
2. **采样轮数**: 增加采样轮数可提高精度但增加计算时间
3. **截断阈值**: 调整阈值可平衡精度和效率
4. **梯度传输**: 确保梯度正确计算和传输

## 故障排除

### 问题 1: 内存不足
**解决方案**: 减少 `gtg-sampling-rounds` 或 `batch-size`

### 问题 2: Shapley 值全为 0
**检查**: 
- 梯度是否正确计算
- 学习率是否合适
- 模型是否正常训练

### 问题 3: AFR 未被检测
**检查**:
- AFR 客户端ID是否正确设置
- 贡献分数更新逻辑是否正确
- 阈值是否合理

## 参考文献

1. Liu, Z., Chen, Y., Yu, H., Liu, Y., & Cui, L. (2022). GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation in Federated Learning. ACM Transactions on Intelligent Systems and Technology, 13(4), 1-21.

2. PASS 论文（请补充）

## 联系方式

如有问题，请联系项目维护者。

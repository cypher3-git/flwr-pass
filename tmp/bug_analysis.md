# 🐛 Client 2 (AFR) 无法被剔除的问题分析

## 问题描述

在训练过程中，Client 2 (AFR攻击者) 的贡献分数收敛到 **0.1111**，高于阈值 **0.05**，导致无法被剔除。

### 分数变化趋势

```
Round 1:  1.0000  (初始值)
Round 2:  0.5556  ↓
Round 3:  0.3333  ↓
Round 4:  0.2222  ↓
Round 5:  0.1667  ↓
Round 6:  0.1389  ↓
Round 7:  0.1250  ↓
Round 8:  0.1181  ↓
Round 9:  0.1146  ↓
Round 10: 0.1128  ↓
...
Round 16: 0.1111  → 收敛！
Round 17-50: 0.1111 (不再变化)
```

**阈值**: 0.05  
**收敛值**: 0.1111 = 1/9  
**结果**: 0.1111 > 0.05 → **无法剔除** ❌

## 🔍 根本原因

### 1. 排名分数的致命缺陷

**原代码逻辑**：
```python
# 基于累积 Shapley 值排序
sorted_clients = sorted(all_clients, key=lambda x: cumulative_shapley[x])

# 使用相对排名计算分数
for rank, cid in enumerate(sorted_clients):
    rank_score = rank / (n - 1)  # n=10, rank ∈ [0,1,2,...,9]
```

**问题**：
- 10个客户端，排名分数只有10个离散值：`{0, 1/9, 2/9, ..., 8/9, 1}`
- Client 2 的 Shapley 值是倒数第二低
- Client 7 (SFR) 是最低
- 因此 Client 2 的排名永远是 **1**
- `rank_score = 1/9 = 0.1111`

### 2. 移动平均收敛分析

移动平均公式：
```
new_score = α × old_score + (1-α) × rank_score
```

当 `α=0.5`, `rank_score=0.1111` 时：
```
new_score = 0.5 × old_score + 0.5 × 0.1111
```

**收敛值计算**：
```
设收敛值为 x，则：
x = 0.5x + 0.5 × 0.1111
0.5x = 0.5 × 0.1111
x = 0.1111
```

### 3. 为什么 Client 7 能被剔除

Client 7 的排名 = 0（最差）：
```
rank_score = 0/9 = 0.0
收敛值 = 0.0 < 0.05 → 成功剔除 ✅
```

## 📊 数学证明

### 排名分数的收敛性

对于排名为 `r` 的客户端，其分数收敛到：

```
lim(t→∞) score_t = r / (n-1)
```

**证明**：
```
score_{t+1} = α × score_t + (1-α) × r/(n-1)

设 lim(t→∞) score_t = x，则：
x = α × x + (1-α) × r/(n-1)
x - αx = (1-α) × r/(n-1)
(1-α)x = (1-α) × r/(n-1)
x = r/(n-1)
```

### 10个客户端的收敛值表

| 排名 | rank_score | 收敛值 | vs 阈值0.05 |
|------|-----------|--------|-------------|
| 0 (最差) | 0/9 = 0.0000 | 0.0000 | ✅ 可剔除 |
| 1 | 1/9 = 0.1111 | 0.1111 | ❌ 无法剔除 |
| 2 | 2/9 = 0.2222 | 0.2222 | ❌ 无法剔除 |
| ... | ... | ... | ... |
| 9 (最好) | 9/9 = 1.0000 | 1.0000 | ❌ 无法剔除 |

**结论**：只有排名最差的客户端能被剔除！

## 🔧 解决方案

### 方案1：归一化累积Shapley值（已实施）

**新逻辑**：
```python
# 获取累积Shapley值的范围
min_val = min(cumulative_shapley.values())
max_val = max(cumulative_shapley.values())

# 归一化到[0, 1]
normalized_score = (cumulative_shapley[cid] - min_val) / (max_val - min_val)

# 移动平均更新
new_score = α × old_score + (1-α) × normalized_score
```

**优点**：
- ✅ 分数连续变化，不受排名限制
- ✅ 负Shapley值的客户端分数趋向0
- ✅ 正Shapley值的客户端分数趋向1
- ✅ 能够区分不同程度的恶意行为

**示例**：
假设累积Shapley值：
```
Client 7: -0.05 (最差)
Client 2: -0.02 (次差)
Client 5:  0.10
Client 0:  0.30 (最好)
```

归一化后：
```
Client 7: (-0.05 - (-0.05)) / (0.30 - (-0.05)) = 0.0 / 0.35 = 0.00
Client 2: (-0.02 - (-0.05)) / (0.30 - (-0.05)) = 0.03 / 0.35 = 0.086
Client 5: (0.10 - (-0.05)) / (0.30 - (-0.05)) = 0.15 / 0.35 = 0.43
Client 0: (0.30 - (-0.05)) / (0.30 - (-0.05)) = 0.35 / 0.35 = 1.00
```

经过移动平均后：
- Client 7 → 收敛到 **0.00 < 0.05** ✅ 剔除
- Client 2 → 收敛到 **0.086 > 0.05** ⚠️ 仍需调整阈值

### 方案2：调整阈值（配合方案1）

由于归一化后，Client 2 的分数约为 0.086，需要调整阈值：

```toml
# pyproject.toml
beta = 1.5  # threshold = 1/(1.5×10) = 0.0667
```

或者使用更激进的：
```toml
beta = 1.2  # threshold = 1/(1.2×10) = 0.0833
```

### 方案3：使用Softmax归一化（备选）

```python
# 使用softmax将Shapley值转换为概率分布
exp_values = np.exp(cumulative_shapley_values)
softmax_scores = exp_values / np.sum(exp_values)
```

**优点**：
- 自动处理负值
- 强调差异性

**缺点**：
- 可能过度放大差异
- 不够直观

## 📈 预期效果

### 使用归一化方案后

假设 Client 2 的累积Shapley值约为 -0.02，Client 7 为 -0.05：

| 轮次 | Client 2 Shapley | 归一化分数 | 移动平均后 | vs 阈值 |
|------|------------------|-----------|-----------|---------|
| 2 | -0.01 | 0.10 | 0.55 | > 0.05 |
| 5 | -0.02 | 0.086 | 0.29 | > 0.05 |
| 10 | -0.03 | 0.057 | 0.14 | > 0.05 |
| 15 | -0.04 | 0.029 | 0.07 | > 0.05 |
| 20 | -0.05 | 0.0 | **0.035** | **< 0.05** ✅ |

**预期**：Client 2 应该在第 15-20 轮被剔除。

## 🧪 测试验证

### 测试步骤

1. 运行新代码：
```bash
flwr run . local-simulation
```

2. 观察指标：
   - Client 2 的分数变化趋势
   - Client 2 被剔除的轮次
   - 最终模型准确率

3. 对比旧版本：
   - 旧版：Client 2 收敛到 0.1111，永不剔除
   - 新版：Client 2 应该在 15-20 轮被剔除

### 预期日志输出

```
[Round 15] Contribution Scores (threshold=0.0500):
  Client 2: score = 0.0720
  ...

[Round 18] Contribution Scores (threshold=0.0500):
  Client 2: score = 0.0520
  ...

[Round 20] Contribution Scores (threshold=0.0500):
  Client 2: score = 0.0480 [ELIMINATED]
  ...

[FREE-RIDERS DETECTED]: ['2', '7']
```

## 📝 代码变更总结

### 修改文件
- `passexample/server_app_gtg.py`

### 修改内容
1. 将排名分数改为归一化累积Shapley值
2. 使用 Min-Max 归一化：`(x - min) / (max - min)`
3. 保留移动平均机制

### 关键代码
```python
# 归一化累积 Shapley 值
min_val = min(cumulative_values)
max_val = max(cumulative_values)

normalized_score = (cumulative_shapley[cid] - min_val) / (max_val - min_val)

# 移动平均更新
new_score = alpha * old_score + (1 - alpha) * normalized_score
```

## ⚠️ 潜在问题

### 1. 所有Shapley值相同
如果所有客户端的累积Shapley值相同（`max_val == min_val`），会导致除零错误。

**解决**：
```python
if max_val > min_val:
    normalized_score = (value - min_val) / (max_val - min_val)
else:
    normalized_score = 0.5  # 默认值
```

### 2. 早期轮次的不稳定性
前几轮累积Shapley值较小，归一化可能不稳定。

**解决**：
- 保持初始分数为1.0
- 使用移动平均平滑变化

### 3. 阈值可能需要调整
归一化后的分数分布可能与原来不同，阈值 0.05 可能需要微调。

**建议**：
- 先运行测试，观察分数分布
- 根据实际情况调整 beta 参数

## ✅ 总结

### 问题本质
使用**离散的排名分数**导致Client 2收敛到固定值0.1111，高于阈值无法剔除。

### 解决方案
使用**连续的归一化Shapley值**，让分数能够反映真实的贡献差异。

### 预期改进
- ✅ Client 2 能够在15-20轮被成功剔除
- ✅ 分数变化更平滑、更合理
- ✅ 能够区分不同程度的恶意行为

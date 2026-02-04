# 提示词：使用 Flower (flwr) 框架复现 PASS 算法,代码写在/pass这个目录下

**角色：** AI 研究工程师 / 联邦学习专家
**任务：** 使用 **Flower (flwr)** 框架和 **PyTorch** 复现 **PASS** 算法（基于参数审计的安全公平联邦学习方案）。

**背景：**
PASS 算法引入了一种独特的“参数审计（Parameter Audit）”机制，要求客户端必须使用自己的私有数据来验证其他客户端的更新。这需要一种通常在标准联邦学习流程中不存在的特定通信流。你必须利用 Flower 的自定义 `Strategy`（策略）和 `Config`（配置）传输功能来实现这一点。

---

## 1. 框架与架构要求 (Framework & Architecture Requirements)

* **框架：** 仿真使用 `flwr` (Flower)，模型构建使用 `torch` (PyTorch)。
* **运行模式：** 使用 `flwr.simulation.start_simulation` 启动仿真。
* **关键数据流实现：**
    1.  **服务端 -> 客户端 (分发)：** 实现一个继承自 `FedAvg` 的自定义 `Strategy`。在 `configure_fit` 方法中，你必须将“上一轮所有客户端的更新”进行序列化，并将其注入到发送给被选中客户端的 `config` 字典中。
    2.  **客户端 (审计与训练)：** 在 `Client.fit` 方法中：
        * 从 `config` 中反序列化更新数据。
        * 执行 **参数审计 (Parameter Audit)**（在本地私有数据上验证他人的更新）。
        * 执行 **本地训练 (Local Training)**。
    3.  **客户端 -> 服务端 (反馈)：** 客户端必须将审计结果（准确率差异，`AccDiv`）放入 `fit` 返回值的 `metrics` 字典中传回。
    4.  **服务端 (聚合与剔除)：** 在 `Strategy.aggregate_fit` 方法中，提取 `metrics`，计算贡献分数 ($c_i$)，过滤掉搭便车者 (Free-Riders)，仅聚合有效的更新。

---

## 2. 算法细节 (Algorithm Details)

### A. 客户端部分 (`PASSClient`)
客户端必须实现两个核心的 PASS 组件：

1.  **PASS-PPS (隐私保护策略)**：
    * 在本地训练结束后，应用 **弱差分隐私 (Weak Differential Privacy)**：向梯度/权重添加高斯噪声。
        * 公式：$\Delta\tilde{\theta}_{ldp} = \Delta\tilde{\theta} + \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$。
    * 应用 **参数剪枝 (Parameter Pruning)**：以比率 $\gamma$ 随机修剪（置零）参数。
        * 公式：$\Delta\tilde{\theta}_{final} = (1-\gamma) \cdot \Delta\tilde{\theta}_{ldp}$。

2.  **参数审计 (Parameter Audit)**：
    * 遍历从其他客户端接收到的更新（通过 `config` 获取）。
    * 在本地私有测试集上计算 **准确率差异 (Accuracy Divergence, AccDiv)**：
        $$AccDiv_{i}^{r} = Acc(\theta^r) - Acc(\Delta\tilde{\theta}_{j} + \theta^{r-1})$$
    * 在 `metrics` 中返回 `{other_client_id: acc_div}`。

### B. 服务端部分 (`PASSStrategy`)
策略必须实现 **PASS-CE (贡献评估)**：

1.  **贡献度计算：**
    * 收集所有客户端提交的 `AccDiv` 值。
    * 计算第 $r$ 轮中客户端 $i$ 的贡献分数 $c_i$：
        $$c_{i}^{r} = \alpha \times c_{i}^{r-1} + (1-\alpha) \times \tanh(\frac{1}{N-1}\sum AccDiv)$$
        *（注：目的是惩罚那些降低模型准确率的更新）。*
2.  **搭便车者剔除 (Free-Rider Elimination)：**
    * 阈值公式：$T = \frac{1}{\beta \times N}$。
    * 逻辑：如果 $c_i^r < T$，则丢弃客户端 $i$ 的更新，并将其排除在聚合之外。

---

## 3. 实验设置 (Experimental Settings)

请严格使用论文实验部分提供的超参数：

* **模型：**
    * **MNIST:** 2层 CNN。
    * **CIFAR-10:** 3层 CNN。
* **联邦学习设置：**
    * 轮数 (Rounds): **200**。
    * 优化器: SGD, 学习率 $\eta = 0.1$。
    * 客户端数量: 10 (用于仿真目的)。
* **PASS 超参数：**
    * 高斯噪声方差 ($\sigma_n^2$): **$10^{-2}$**。
    * 剪枝率 ($\gamma$): **90% (0.9)**。
    * 移动平均系数 ($\alpha$): **0.95**。
    * 阈值系数 ($\beta$): **1.75**。

---

## 4. 输出要求 (Output Requirements)

请提供完整的、可运行的 Python 代码，并分为以下几个部分：
1.  **`utils.py`**：模型定义以及用于序列化的辅助函数（例如使用 `pickle` 在 config 中传输数组列表）。
2.  **`client.py`**：继承自 `flwr.client.NumPyClient` 的 `PASSClient` 类实现。
3.  **`strategy.py`**：继承自 `flwr.server.strategy.FedAvg` 的 `PASSStrategy` 类实现。
4.  **`main.py`**：使用 `flwr.simulation` 的仿真入口代码。
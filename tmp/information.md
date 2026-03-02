# PASS: 基于参数审计的安全公平联邦学习方案解析

基于文献《PASS: A Parameter Audit-Based Secure and Fair Federated Learning Scheme Against Free-Rider Attack》，以下是对客户端审计机制、贡献值计算、服务器处理逻辑、实验参数及检测情况的详细解析。

## 1. 客户端的审计机制 (Client-Side Parameter Audit)

PASS 方案的核心思想是利用客户端的私有数据来“审计”其他参与者的模型更新，从而判断其质量。

* **审计流程**：
    1.  [cite_start]服务器在每一轮训练后，会将上一轮聚合后的全局模型参数 $\theta^r$，以及各个客户端上传的本地更新（经过隐私处理后的 $\Delta\tilde{\theta}_{i,ldp}^{r}$）分发给所有客户端 [cite: 207, 214]。
    2.  [cite_start]**验证计算**：客户端 $i$ 使用自己的**私有数据集**，对收到的来自其他客户端的更新进行测试 [cite: 216]。
    3.  [cite_start]**计算准确率差异 (AccDiv)**：客户端计算全局模型 $\theta^r$ 的准确率与使用了某特定客户端更新后的模型准确率之间的差值。公式如下 [cite: 216, 222]：
        $$AccDiv_{i}^{r}=Acc(\theta^{r})-Acc(\Delta\tilde{\theta}_{i,ldp}^{r}+\tilde{\theta}_{i,ldp}^{r-1})$$
        其中，$AccDiv_{i}^{r}$ 反映了该更新对模型性能的影响。
    4.  [cite_start]**上传结果**：客户端将计算出的审计结果（AccDiv）上传回服务器 [cite: 222]。

## 2. 贡献值的计算 (Contribution Evaluation)

贡献值的计算主要在**服务器端**进行，依据是收集到的各个客户端上报的 AccDiv 值。

* **计算步骤**：
    1.  [cite_start]**平均 AccDiv**：服务器收到所有客户端发来的针对客户端 $i$ 的审计结果后，计算平均值：$\frac{1}{N-1}\sum_{i}^{N-1}AccDiv_{i}^{r}$ [cite: 251]。
    2.  [cite_start]**计算贡献值 $c_i^r$**：服务器结合**历史贡献值**和**当前轮次的表现**来计算当前贡献值。使用了 Tanh 函数将结果映射到 [-1, 1] 区间，以区分正向和负向贡献。公式如下 [cite: 222, 252]：
        $$c_{i}^{r} = \frac{1}{\alpha \times c_{i}^{r-1} + (1-\alpha) \times \tanh(\frac{1}{N-1}\sum_{i}^{N-1}AccDiv_{i}^{r})}$$
        [cite_start]其中 $\alpha$ 是移动平均系数（moving average coefficient），用于平衡历史贡献和当前贡献的权重 [cite: 222]。

## 3. 服务器端的处理逻辑 (Server-Side Processing)

服务器根据计算出的贡献值 $c_i^r$ 来决定是否保留该客户端。

* [cite_start]**阈值判定**：服务器设定一个阈值（Threshold），阈值为 $\frac{1}{\beta \times N}$，其中 $\beta$ 是阈值系数，$N$ 是参与者总数 [cite: 224, 253]。
* **剔除机制**：
    * [cite_start]如果某客户端 $i$ 的贡献值 $c_{i}^{r} < \frac{1}{\beta \times N}$，则该客户端被判定为“搭便车者”（Free-Rider）[cite: 222, 224]。
    * [cite_start]服务器会将该客户端从联邦学习系统中**移除 (Eliminate/Remove)** [cite: 224]。

## 4. 实验参数设置 (Experimental Settings)

文中为了验证 PASS 的有效性，设置了详细的实验环境和超参数：

* **数据集与模型**：
    * [cite_start]**MNIST**：使用 2 层 CNN 模型 [cite: 299]。
    * [cite_start]**CIFAR-10**：使用 3 层 CNN 模型 [cite: 299]。
    * 数据分布包含 I.I.D. (独立同分布) 和 Non-I.I.D. [cite_start]两种情况 [cite: 301, 302]。
* **联邦学习基础设置**：
    * [cite_start]优化器：SGD [cite: 300]。
    * [cite_start]学习率 (Learning Rate)：$\eta = 0.1$ [cite: 300]。
    * [cite_start]总轮数 (Round)：$R = 200$ [cite: 300]。
    * [cite_start]聚合算法：FedAvg [cite: 300]。
* **PASS 特定参数**：
    * [cite_start]**隐私保护噪声**：高斯噪声方差 $\sigma_{n}^{2} = 10^{-2}$（用于平衡隐私与准确率）[cite: 312, 409, 494]。
    * [cite_start]**参数剪枝率 (Pruning Rate)**：$\gamma = 90\%$（用于减少通信开销且保持性能）[cite: 507, 508]。
    * [cite_start]**移动平均系数**：$\alpha = 0.95$ [cite: 313]。
    * [cite_start]**阈值系数**：$\beta = 1.75$（在此设置下实现了 DSR 和 FPR 的最佳权衡）[cite: 499, 500]。
* **攻击者设置**：
    * [cite_start]攻击者比例：设置了 9%, 33%, 60%, 67% 四种比例，分别对应 1, 5, 15, 20 个攻击者（总共客户端 N=25 或 N=30 视具体实验而定）[cite: 304]。
    * [cite_start]**SFR 攻击模拟**：SFR 攻击者使用 MNIST 数据集（调整维度为 3x32x32 以匹配 CIFAR-10）进行训练，以此模拟不贡献真实数据的情况 [cite: 305, 306, 307]。

## 5. 对搭便车者的检测情况 (Detection Performance)

PASS 方案在对抗匿名搭便车 (AFR) 和自私搭便车 (SFR) 攻击方面表现优异，具体指标如下：

* **评价指标**：
    * [cite_start]**DSR (Defense Success Rate)**：防御成功率，即成功剔除攻击者的比例 [cite: 367]。
    * [cite_start]**FPR (False Positive Rate)**：误报率，即错误剔除诚实客户端的比例 [cite: 368, 369]。
    * [cite_start]**F1-Score**：综合评价指标 [cite: 370]。

* **检测结果**：
    1.  [cite_start]**整体性能**：PASS 能够实现 **100% 的防御成功率 (DSR)**，同时误报率 (FPR) 维持在 **20%** 左右 [cite: 112, 561]。
    2.  **F1-Score**：
        * [cite_start]对抗 **AFR** 攻击的平均 F1-score 为 **89%** [cite: 112, 580]。
        * [cite_start]对抗 **SFR** 攻击的平均 F1-score 为 **88%** [cite: 112, 580]。
    3.  [cite_start]**高比例攻击者场景**：即使攻击者数量**超过 50%**（即攻击者占多数），PASS 依然有效 [cite: 103, 124]：
        * [cite_start]对抗 AFR 的 F1-score 为 89% [cite: 16]。
        * [cite_start]对抗 SFR 的 F1-score 为 87% [cite: 16, 124]。
    4.  [cite_start]**对比优势**：与其他防御模型（如 RFFL, Median, SignSGD 等）相比，PASS 在 SFR 和 AFR 攻击下均取得了最高的 F1-score 和最低的误报率 [cite: 112, 561, 575]。
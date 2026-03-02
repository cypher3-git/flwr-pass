"""GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation

基于论文: GTG-Shapley: Efficient and Accurate Participant Contribution Evaluation 
in Federated Learning (ACM TIST 2022)

核心思想:
1. 梯度重构: 从梯度更新重构模型参数，避免重复训练
2. Shapley 值计算: 使用 Monte Carlo 采样估计每个客户端的边际贡献
3. 引导截断采样: Within-round 和 Between-round 截断减少计算量
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools


class GTGShapley:
    """GTG-Shapley 贡献评估器
    
    严格按照论文 Algorithm 1 实现：
    - 轮间截断 (Between-round Truncation)
    - 轮内截断 (Within-round Truncation)  
    - 引导采样 (Guided Sampling)
    - 基于梯度的子模型重构
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader,
        device: torch.device,
        learning_rate: float = 0.01,
        num_sampling_rounds: int = 10,
        within_round_threshold: float = 0.01,
        between_round_threshold: float = 0.005,
        convergence_threshold: float = 0.05,
        guided_sampling_m: int = 1,
    ):
        """
        Args:
            model: 神经网络模型
            test_loader: 测试数据加载器
            device: 计算设备
            learning_rate: 学习率（用于梯度重构）
            num_sampling_rounds: Monte Carlo 最大采样轮数
            within_round_threshold: ε_i - 轮内截断阈值
            between_round_threshold: ε_b - 轮间截断阈值
            convergence_threshold: 蒙特卡洛收敛判定阈值 (std/mean < threshold)
            guided_sampling_m: 引导采样中前 m 个位置的循环占据数
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.lr = learning_rate
        self.num_sampling_rounds = num_sampling_rounds
        self.within_threshold = within_round_threshold
        self.between_threshold = between_round_threshold
        self.convergence_threshold = convergence_threshold
        self.guided_sampling_m = guided_sampling_m
        
        # 性能优化：缓存评估结果
        self._eval_cache: Dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def reconstruct_model_from_gradient(
        self,
        base_params: dict,
        gradient: dict,
    ) -> dict:
        """从梯度重构模型参数
        
        公式: θ_new = θ_base - lr * gradient
        
        Args:
            base_params: 基础模型参数
            gradient: 梯度更新
            
        Returns:
            重构的模型参数
        """
        reconstructed = {}
        for key in base_params:
            if key in gradient:
                reconstructed[key] = base_params[key] - self.lr * gradient[key]
            else:
                reconstructed[key] = base_params[key].clone()
        return reconstructed
    
    def aggregate_gradients(
        self,
        base_params: dict,
        gradients: List[dict],
        weights: Optional[List[float]] = None
    ) -> dict:
        """聚合多个客户端的梯度
        
        Args:
            base_params: 基础模型参数
            gradients: 客户端梯度列表
            weights: 聚合权重（默认平均）
            
        Returns:
            聚合后的模型参数
        """
        if not gradients:
            return base_params
        
        if weights is None:
            weights = [1.0 / len(gradients)] * len(gradients)
        
        # 聚合梯度
        aggregated_gradient = {}
        for key in base_params:
            aggregated_gradient[key] = torch.zeros_like(base_params[key])
            for grad, weight in zip(gradients, weights):
                if key in grad:
                    aggregated_gradient[key] += weight * grad[key]
        
        # 重构模型
        return self.reconstruct_model_from_gradient(base_params, aggregated_gradient)
    
    def evaluate_model(self, state_dict: dict) -> float:
        """评估模型性能
        
        Args:
            state_dict: 模型参数
            
        Returns:
            准确率
        """
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                # 处理字典格式的批次数据
                if isinstance(batch, dict):
                    # CIFAR-10: {"img": ..., "label": ...}
                    # MNIST: {"image": ..., "label": ...}
                    if "img" in batch:
                        data = batch["img"]
                    elif "image" in batch:
                        data = batch["image"]
                    else:
                        raise ValueError(f"Unknown batch format: {batch.keys()}")
                    target = batch["label"]
                else:
                    # 元组格式（备用）
                    if len(batch) == 3:
                        data, target, _ = batch
                    else:
                        data, target = batch
                
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def compute_marginal_contribution(
        self,
        base_params: dict,
        all_gradients: Dict[str, dict],
        permutation: List[str],
        target_client: str,
    ) -> float:
        """计算目标客户端在给定排列下的边际贡献
        
        边际贡献 = Acc(S ∪ {i}) - Acc(S)
        其中 S 是排列中目标客户端之前的所有客户端
        
        Args:
            base_params: 基础模型参数
            all_gradients: 所有客户端的梯度
            permutation: 客户端排列
            target_client: 目标客户端ID
            
        Returns:
            边际贡献值
        """
        # 找到目标客户端在排列中的位置
        if target_client not in permutation:
            return 0.0
        
        target_idx = permutation.index(target_client)
        
        # S: 目标客户端之前的所有客户端
        clients_before = permutation[:target_idx]
        
        # 计算 Acc(S)
        if clients_before:
            gradients_before = [all_gradients[cid] for cid in clients_before if cid in all_gradients]
            model_s = self.aggregate_gradients(base_params, gradients_before)
            acc_s = self.evaluate_model(model_s)
        else:
            # 空集，使用基础模型
            acc_s = self.evaluate_model(base_params)
        
        # 计算 Acc(S ∪ {i})
        clients_with_target = permutation[:target_idx + 1]
        gradients_with_target = [all_gradients[cid] for cid in clients_with_target if cid in all_gradients]
        model_s_union_i = self.aggregate_gradients(base_params, gradients_with_target)
        acc_s_union_i = self.evaluate_model(model_s_union_i)
        
        # 边际贡献
        marginal = acc_s_union_i - acc_s
        return marginal
    
    def compute_shapley_value_with_truncation(
        self,
        base_params: dict,
        all_gradients: Dict[str, dict],
        target_client: str,
    ) -> float:
        """使用引导截断采样计算 Shapley 值
        
        Args:
            base_params: 基础模型参数
            all_gradients: 所有客户端的梯度
            target_client: 目标客户端ID
            
        Returns:
            Shapley 值估计
        """
        client_ids = list(all_gradients.keys())
        if target_client not in client_ids:
            return 0.0
        
        marginal_contributions = []
        
        for round_idx in range(self.num_sampling_rounds):
            # 随机排列客户端
            permutation = client_ids.copy()
            random.shuffle(permutation)
            
            # 计算边际贡献
            marginal = self.compute_marginal_contribution(
                base_params, all_gradients, permutation, target_client
            )
            marginal_contributions.append(marginal)
            
            # Within-round 截断: 如果方差足够小，提前停止
            if len(marginal_contributions) >= 3:
                std = np.std(marginal_contributions)
                if std < self.within_threshold:
                    break
        
        # 返回平均边际贡献作为 Shapley 值估计
        shapley_value = np.mean(marginal_contributions)
        return shapley_value
    
    def evaluate_all_clients(
        self,
        base_params: dict,
        all_gradients: Dict[str, dict],
        auditor_id: str,
    ) -> Dict[str, float]:
        """评估所有其他客户端的 Shapley 值（基于梯度）
        
        Args:
            base_params: 基础模型参数
            all_gradients: 所有客户端的梯度
            auditor_id: 审计者ID（自己）
            
        Returns:
            其他客户端的 Shapley 值字典
        """
        shapley_values = {}
        
        for client_id in all_gradients:
            if client_id != auditor_id:
                sv = self.compute_shapley_value_with_truncation(
                    base_params, all_gradients, client_id
                )
                shapley_values[client_id] = sv
        
        return shapley_values
    
    def fedavg_aggregate_params(
        self,
        params_list: List[dict],
        weights: Optional[List[float]] = None
    ) -> dict:
        """FedAvg 加权平均聚合参数
        
        按照 GTG-Shapley 源码的方式进行参数聚合
        
        Args:
            params_list: 客户端模型参数列表
            weights: 聚合权重（默认平均）
            
        Returns:
            聚合后的模型参数
        """
        if not params_list:
            return {}
        
        if weights is None:
            weights = [1.0 / len(params_list)] * len(params_list)
        
        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        aggregated = {}
        for key in params_list[0]:
            aggregated[key] = torch.zeros_like(params_list[0][key], dtype=torch.float64)
            for params, weight in zip(params_list, weights):
                if key in params:
                    aggregated[key] += weight * params[key].to(torch.float64)
            aggregated[key] = aggregated[key].to(params_list[0][key].dtype)
        
        return aggregated
    
    def compute_marginal_contribution_fedavg(
        self,
        initial_params: dict,
        all_client_params: Dict[str, dict],
        permutation: List[str],
        target_client: str,
    ) -> float:
        """计算目标客户端在给定排列下的边际贡献（FedAvg 方式）
        
        按照 GTG-Shapley 源码：
        边际贡献 = Acc(FedAvg(S ∪ {i})) - Acc(FedAvg(S))
        
        Args:
            initial_params: 初始模型参数（用作空集的基准）
            all_client_params: 所有客户端的模型参数
            permutation: 客户端排列
            target_client: 目标客户端ID
            
        Returns:
            边际贡献值
        """
        if target_client not in permutation:
            return 0.0
        
        target_idx = permutation.index(target_client)
        
        # S: 目标客户端之前的所有客户端
        clients_before = permutation[:target_idx]
        
        # 计算 Acc(FedAvg(S))
        if clients_before:
            params_before = [all_client_params[cid] for cid in clients_before if cid in all_client_params]
            if params_before:
                model_s = self.fedavg_aggregate_params(params_before)
                acc_s = self.evaluate_model(model_s)
            else:
                acc_s = self.evaluate_model(initial_params)
        else:
            # 空集，使用初始模型
            acc_s = self.evaluate_model(initial_params)
        
        # 计算 Acc(FedAvg(S ∪ {i}))
        clients_with_target = permutation[:target_idx + 1]
        params_with_target = [all_client_params[cid] for cid in clients_with_target if cid in all_client_params]
        if params_with_target:
            model_s_union_i = self.fedavg_aggregate_params(params_with_target)
            acc_s_union_i = self.evaluate_model(model_s_union_i)
        else:
            acc_s_union_i = self.evaluate_model(initial_params)
        
        # 边际贡献
        marginal = acc_s_union_i - acc_s
        return marginal
    
    def compute_shapley_value_fedavg(
        self,
        initial_params: dict,
        all_client_params: Dict[str, dict],
        target_client: str,
    ) -> float:
        """使用引导截断采样计算 Shapley 值（FedAvg 方式）
        
        Args:
            initial_params: 初始模型参数
            all_client_params: 所有客户端的模型参数
            target_client: 目标客户端ID
            
        Returns:
            Shapley 值估计
        """
        client_ids = list(all_client_params.keys())
        if target_client not in client_ids:
            return 0.0
        
        marginal_contributions = []
        
        for round_idx in range(self.num_sampling_rounds):
            # 随机排列客户端
            permutation = client_ids.copy()
            random.shuffle(permutation)
            
            # 计算边际贡献
            marginal = self.compute_marginal_contribution_fedavg(
                initial_params, all_client_params, permutation, target_client
            )
            marginal_contributions.append(marginal)
            
            # Within-round 截断: 如果方差足够小，提前停止
            if len(marginal_contributions) >= 3:
                std = np.std(marginal_contributions)
                if std < self.within_threshold:
                    break
        
        # 返回平均边际贡献作为 Shapley 值估计
        shapley_value = np.mean(marginal_contributions)
        return shapley_value
    
    def evaluate_all_clients_fedavg(
        self,
        initial_params: dict,
        all_client_params: Dict[str, dict],
        auditor_id: str,
    ) -> Dict[str, float]:
        """评估所有其他客户端的 Shapley 值（FedAvg 方式）
        
        性能优化：
        - 使用共享排列减少重复计算
        - 缓存评估结果避免重复评估相同子集
        
        Args:
            initial_params: 初始模型参数（用作空集的基准）
            all_client_params: 所有客户端的模型参数
            auditor_id: 审计者ID（自己）
            
        Returns:
            其他客户端的 Shapley 值字典
        """
        # 清空缓存
        self._eval_cache.clear()
        
        # 使用共享排列优化：生成一次排列，计算所有客户端的边际贡献
        shapley_values = self._compute_all_shapley_values_optimized(
            initial_params, all_client_params, auditor_id
        )
        
        return shapley_values
    
    def evaluate_all_clients_with_deltas(
        self,
        base_model: dict,
        client_deltas: Dict[str, dict],
        data_sizes: Dict[str, int],
        auditor_id: str,
    ) -> Dict[str, float]:
        """评估所有其他客户端的 Shapley 值（基于梯度更新）
        
        严格按照论文公式实现：
        V(S) = V(M̃_S) = V(M^{(t)} + Σ_{i∈S} (|D_i|/|D_S|) * Δ_i)
        
        Args:
            base_model: 基础模型参数 M^{(t)}
            client_deltas: 客户端梯度更新字典 {客户端ID: Δ_i}
            data_sizes: 客户端数据集大小字典 {客户端ID: |D_i|}
            auditor_id: 审计者ID（自己）
            
        Returns:
            其他客户端的 Shapley 值字典
        """
        # 清空缓存
        self._eval_cache.clear()
        
        # 使用论文公式计算 Shapley 值
        shapley_values = self._compute_shapley_with_deltas(
            base_model, client_deltas, data_sizes, auditor_id
        )
        
        return shapley_values
    
    def _compute_shapley_with_deltas(
        self,
        base_model: dict,
        client_deltas: Dict[str, dict],
        data_sizes: Dict[str, int],
        auditor_id: str,
    ) -> Dict[str, float]:
        """GTG-Shapley Algorithm 1 实现（基于梯度更新）
        
        严格按照论文实现：
        1. 轮间截断 (Between-round Truncation)
        2. 引导采样 (Guided Sampling)
        3. 轮内截断 (Within-round Truncation)
        4. 基于梯度的子模型重构: M̃_S = M^{(t)} + Σ_{i∈S} (|D_i|/|D_S|) * Δ_i
        5. 增量式 Shapley 值更新
        6. 蒙特卡洛收敛判定
        """
        client_ids = [cid for cid in client_deltas.keys() if cid != auditor_id]
        n = len(client_ids)
        if n == 0:
            return {}
        
        # ============================================================
        # Step 1: 轮间截断 (Between-round Truncation)
        # ============================================================
        # 计算 v_0 = V(M^{(t)}) - 基础模型效用
        v_0 = self._evaluate_with_cache(base_model, "base_model")
        
        # 计算 v_N = V(M^{(t+1)}) - 完全聚合后的效用
        # 使用论文公式重构完整模型
        all_deltas = [client_deltas[cid] for cid in client_ids]
        all_data_sizes = [data_sizes.get(cid, 1) for cid in client_ids]
        full_model = self.reconstruct_model_from_deltas(base_model, all_deltas, all_data_sizes)
        v_N = self._evaluate_with_cache(full_model, "full_model")
        
        # 轮间截断判断：如果 |v_N - v_0| <= ε_b，跳过本轮
        if abs(v_N - v_0) <= self.between_threshold:
            print(f"[GTG-Shapley] Between-round truncation: |{v_N:.4f} - {v_0:.4f}| = {abs(v_N - v_0):.4f} <= {self.between_threshold}")
            return {cid: 0.0 for cid in client_ids}
        
        # ============================================================
        # Step 2: 初始化 Shapley 值
        # ============================================================
        shapley_values = {cid: 0.0 for cid in client_ids}
        
        # ============================================================
        # Step 3: 蒙特卡洛采样循环
        # ============================================================
        k = 0  # 采样轮数计数器
        m = min(self.guided_sampling_m, n)  # 引导采样的前 m 个位置
        
        while k < self.num_sampling_rounds:
            k += 1
            
            # ============================================================
            # Step 3.1: 引导采样生成排列 (Guided Sampling)
            # ============================================================
            permutation = self._generate_guided_permutation(client_ids, k, m)
            
            # ============================================================
            # Step 3.2: 遍历排列，计算边际贡献（含轮内截断）
            # ============================================================
            prev_acc = v_0  # v_{j-1}^k 初始为 v_0
            
            for j, client_id in enumerate(permutation):
                # ============================================================
                # Step 3.2.1: 轮内截断判断 (Within-round Truncation)
                # ============================================================
                # 如果剩余潜在增益 |v_N - v_{j-1}^k| < ε_i，触发截断
                if abs(v_N - prev_acc) < self.within_threshold:
                    # 截断：后续所有客户端的边际贡献为 0
                    marginal = 0.0
                else:
                    # ============================================================
                    # Step 3.2.2: 使用论文公式重构子模型并计算效用
                    # M̃_S = M^{(t)} + Σ_{i∈S} (|D_i|/|D_S|) * Δ_i
                    # ============================================================
                    clients_subset = permutation[:j + 1]
                    cache_key = "|".join(sorted(clients_subset))
                    
                    # 收集子集的梯度更新和数据集大小
                    subset_deltas = [client_deltas[cid] for cid in clients_subset]
                    subset_data_sizes = [data_sizes.get(cid, 1) for cid in clients_subset]
                    
                    # 使用论文公式重构子模型
                    subset_model = self.reconstruct_model_from_deltas(
                        base_model, subset_deltas, subset_data_sizes
                    )
                    curr_acc = self._evaluate_with_cache(subset_model, cache_key)
                    
                    # 边际贡献 = v_j^k - v_{j-1}^k
                    marginal = curr_acc - prev_acc
                    prev_acc = curr_acc
                
                # ============================================================
                # Step 3.2.3: 增量式 Shapley 值更新
                # φ_i^{(t+1)} = (k-1)/k * φ_i^{(t+1)} + 1/k * marginal
                # ============================================================
                shapley_values[client_id] = ((k - 1) / k) * shapley_values[client_id] + (1 / k) * marginal
            
            # ============================================================
            # Step 3.3: 蒙特卡洛收敛判定
            # ============================================================
            if k >= 3:  # 至少 3 轮后才检查收敛
                if self._check_convergence(shapley_values):
                    print(f"[GTG-Shapley] Converged after {k} sampling rounds")
                    break
        
        return shapley_values
    
    def _compute_all_shapley_values_optimized(
        self,
        initial_params: dict,
        all_client_params: Dict[str, dict],
        auditor_id: str,
    ) -> Dict[str, float]:
        """GTG-Shapley Algorithm 1 实现
        
        严格按照论文实现：
        1. 轮间截断 (Between-round Truncation)
        2. 引导采样 (Guided Sampling)
        3. 轮内截断 (Within-round Truncation)
        4. 增量式 Shapley 值更新
        5. 蒙特卡洛收敛判定
        """
        client_ids = [cid for cid in all_client_params.keys() if cid != auditor_id]
        n = len(client_ids)
        if n == 0:
            return {}
        
        # ============================================================
        # Step 1: 轮间截断 (Between-round Truncation)
        # ============================================================
        # 计算 v_0 = V(M^{(t)}) - 初始模型效用
        v_0 = self._evaluate_with_cache(initial_params, "initial")
        
        # 计算 v_N = V(M^{(t+1)}) - 完全聚合后的效用
        all_params_list = [all_client_params[cid] for cid in client_ids]
        full_aggregated = self.fedavg_aggregate_params(all_params_list)
        v_N = self._evaluate_with_cache(full_aggregated, "full_aggregated")
        
        # 轮间截断判断：如果 |v_N - v_0| <= ε_b，跳过本轮
        if abs(v_N - v_0) <= self.between_threshold:
            print(f"[GTG-Shapley] Between-round truncation triggered: |{v_N:.4f} - {v_0:.4f}| = {abs(v_N - v_0):.4f} <= {self.between_threshold}")
            return {cid: 0.0 for cid in client_ids}
        
        # ============================================================
        # Step 2: 初始化 Shapley 值
        # ============================================================
        shapley_values = {cid: 0.0 for cid in client_ids}
        
        # ============================================================
        # Step 3: 蒙特卡洛采样循环
        # ============================================================
        k = 0  # 采样轮数计数器
        m = min(self.guided_sampling_m, n)  # 引导采样的前 m 个位置
        
        while k < self.num_sampling_rounds:
            k += 1
            
            # ============================================================
            # Step 3.1: 引导采样生成排列 (Guided Sampling)
            # ============================================================
            permutation = self._generate_guided_permutation(client_ids, k, m)
            
            # ============================================================
            # Step 3.2: 遍历排列，计算边际贡献（含轮内截断）
            # ============================================================
            prev_acc = v_0  # v_{j-1}^k 初始为 v_0
            
            for j, client_id in enumerate(permutation):
                # ============================================================
                # Step 3.2.1: 轮内截断判断 (Within-round Truncation)
                # ============================================================
                # 如果剩余潜在增益 |v_N - v_{j-1}^k| < ε_i，触发截断
                if abs(v_N - prev_acc) < self.within_threshold:
                    # 截断：后续所有客户端的边际贡献为 0
                    # v_j^k = v_{j-1}^k，所以 marginal = 0
                    marginal = 0.0
                else:
                    # ============================================================
                    # Step 3.2.2: 重构子模型并计算效用
                    # ============================================================
                    clients_subset = permutation[:j + 1]
                    cache_key = "|".join(sorted(clients_subset))
                    
                    params_list = [all_client_params[cid] for cid in clients_subset]
                    aggregated = self.fedavg_aggregate_params(params_list)
                    curr_acc = self._evaluate_with_cache(aggregated, cache_key)
                    
                    # 边际贡献 = v_j^k - v_{j-1}^k
                    marginal = curr_acc - prev_acc
                    prev_acc = curr_acc
                
                # ============================================================
                # Step 3.2.3: 增量式 Shapley 值更新
                # φ_i^{(t+1)} = (k-1)/k * φ_i^{(t+1)} + 1/k * marginal
                # ============================================================
                shapley_values[client_id] = ((k - 1) / k) * shapley_values[client_id] + (1 / k) * marginal
            
            # ============================================================
            # Step 3.3: 蒙特卡洛收敛判定
            # ============================================================
            if k >= 3:  # 至少 3 轮后才检查收敛
                if self._check_convergence(shapley_values):
                    print(f"[GTG-Shapley] Converged after {k} sampling rounds")
                    break
        
        return shapley_values
    
    def _generate_guided_permutation(
        self,
        client_ids: List[str],
        k: int,
        m: int,
    ) -> List[str]:
        """引导采样生成排列 (Guided Sampling)
        
        论文 Section 3.2:
        - 排列的前 m 位由 n 个参与者固定循环轮替占据
        - 剩下的 n-m 位进行随机排列
        
        Args:
            client_ids: 客户端ID列表
            k: 当前采样轮数 (1-indexed)
            m: 前 m 个位置的循环占据数
            
        Returns:
            生成的排列
        """
        n = len(client_ids)
        if m <= 0 or m > n:
            # 退化为纯随机排列
            permutation = client_ids.copy()
            random.shuffle(permutation)
            return permutation
        
        # 确定前 m 个位置的客户端（循环轮替）
        # 第 k 轮，前 m 个位置由 client_ids[(k-1)*m : (k-1)*m + m] 循环占据
        front_indices = [(k - 1 + i) % n for i in range(m)]
        front_clients = [client_ids[idx] for idx in front_indices]
        
        # 剩余客户端随机排列
        remaining_clients = [cid for cid in client_ids if cid not in front_clients]
        random.shuffle(remaining_clients)
        
        # 组合排列
        permutation = front_clients + remaining_clients
        return permutation
    
    def _check_convergence(self, shapley_values: Dict[str, float]) -> bool:
        """蒙特卡洛收敛判定
        
        论文 Section 3.3:
        收敛条件：估算的方差与均值的比例低于阈值
        
        这里使用简化的判定：所有 Shapley 值的变异系数 < threshold
        """
        values = list(shapley_values.values())
        if not values:
            return True
        
        mean_val = np.mean(np.abs(values))
        if mean_val < 1e-10:
            return True
        
        std_val = np.std(values)
        cv = std_val / mean_val  # 变异系数
        
        return cv < self.convergence_threshold
    
    def reconstruct_model_from_deltas(
        self,
        base_params: dict,
        deltas: List[dict],
        data_sizes: List[int],
    ) -> dict:
        """论文公式：基于基础模型和梯度更新重构子模型
        
        公式: M̃_S = M^{(t)} + Σ_{i∈S} (|D_i|/|D_S|) * Δ_i
        
        Args:
            base_params: 基础模型参数 M^{(t)}
            deltas: 客户端梯度更新列表 [Δ_i]
            data_sizes: 客户端数据集大小列表 [|D_i|]
            
        Returns:
            重构的子模型参数
        """
        if not deltas:
            return base_params
        
        # 计算子集总数据量 |D_S|
        total_data_size = sum(data_sizes)
        if total_data_size == 0:
            total_data_size = len(deltas)  # 避免除零
        
        # 计算加权梯度聚合
        aggregated_delta = {}
        for key in base_params:
            aggregated_delta[key] = torch.zeros_like(base_params[key], dtype=torch.float64)
            for delta, data_size in zip(deltas, data_sizes):
                if key in delta:
                    weight = data_size / total_data_size
                    aggregated_delta[key] += weight * delta[key].to(torch.float64)
            aggregated_delta[key] = aggregated_delta[key].to(base_params[key].dtype)
        
        # 重构模型: M̃_S = M^{(t)} + aggregated_delta
        reconstructed = {}
        for key in base_params:
            reconstructed[key] = base_params[key] + aggregated_delta[key]
        
        return reconstructed
    
    def _evaluate_with_cache(self, params: dict, cache_key: str) -> float:
        """带缓存的模型评估"""
        if cache_key in self._eval_cache:
            self._cache_hits += 1
            return self._eval_cache[cache_key]
        
        self._cache_misses += 1
        acc = self.evaluate_model(params)
        self._eval_cache[cache_key] = acc
        return acc


def compute_gradient_from_params(
    old_params: dict,
    new_params: dict,
    learning_rate: float = 0.01
) -> dict:
    """从参数更新计算梯度
    
    公式: gradient = (θ_old - θ_new) / lr
    
    Args:
        old_params: 训练前的参数
        new_params: 训练后的参数
        learning_rate: 学习率
        
    Returns:
        梯度字典
    """
    gradient = {}
    for key in old_params:
        if key in new_params:
            gradient[key] = (old_params[key] - new_params[key]) / learning_rate
        else:
            gradient[key] = torch.zeros_like(old_params[key])
    return gradient

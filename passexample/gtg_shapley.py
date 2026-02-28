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
    """GTG-Shapley 贡献评估器"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader,
        device: torch.device,
        learning_rate: float = 0.01,
        num_sampling_rounds: int = 10,
        within_round_threshold: float = 0.01,
        between_round_threshold: float = 0.05,
    ):
        """
        Args:
            model: 神经网络模型
            test_loader: 测试数据加载器
            device: 计算设备
            learning_rate: 学习率（用于梯度重构）
            num_sampling_rounds: Monte Carlo 采样轮数
            within_round_threshold: Within-round 截断阈值
            between_round_threshold: Between-round 截断阈值
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.lr = learning_rate
        self.num_sampling_rounds = num_sampling_rounds
        self.within_threshold = within_round_threshold
        self.between_threshold = between_round_threshold
        
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
    
    def _compute_all_shapley_values_optimized(
        self,
        initial_params: dict,
        all_client_params: Dict[str, dict],
        auditor_id: str,
    ) -> Dict[str, float]:
        """优化的 Shapley 值计算：使用共享排列减少计算量
        
        核心优化：每个排列可以同时计算所有客户端的边际贡献
        """
        client_ids = [cid for cid in all_client_params.keys() if cid != auditor_id]
        if not client_ids:
            return {}
        
        # 初始化边际贡献累加器
        marginal_sums = {cid: [] for cid in client_ids}
        
        # 生成共享排列并计算所有客户端的边际贡献
        for round_idx in range(self.num_sampling_rounds):
            # 随机排列客户端
            permutation = client_ids.copy()
            random.shuffle(permutation)
            
            # 计算该排列下所有客户端的边际贡献
            marginals = self._compute_all_marginals_for_permutation(
                initial_params, all_client_params, permutation
            )
            
            # 累加边际贡献
            for cid, marginal in marginals.items():
                marginal_sums[cid].append(marginal)
            
            # 早停检查：如果所有客户端的方差都足够小，提前停止
            if round_idx >= 2:
                all_converged = True
                for cid in client_ids:
                    if len(marginal_sums[cid]) >= 3:
                        std = np.std(marginal_sums[cid])
                        if std >= self.within_threshold:
                            all_converged = False
                            break
                if all_converged:
                    break
        
        # 计算平均 Shapley 值
        shapley_values = {}
        for cid in client_ids:
            if marginal_sums[cid]:
                shapley_values[cid] = np.mean(marginal_sums[cid])
            else:
                shapley_values[cid] = 0.0
        
        return shapley_values
    
    def _compute_all_marginals_for_permutation(
        self,
        initial_params: dict,
        all_client_params: Dict[str, dict],
        permutation: List[str],
    ) -> Dict[str, float]:
        """计算一个排列下所有客户端的边际贡献
        
        优化：增量计算，复用中间结果
        """
        marginals = {}
        prev_acc = None
        
        for idx, client_id in enumerate(permutation):
            # 计算 S = permutation[:idx] 的准确率
            if prev_acc is None:
                if idx == 0:
                    # 空集，使用初始模型
                    prev_acc = self._evaluate_with_cache(initial_params, "initial")
                else:
                    # 这不应该发生
                    prev_acc = self._evaluate_with_cache(initial_params, "initial")
            
            # 计算 S ∪ {client_id} 的准确率
            clients_with_target = permutation[:idx + 1]
            cache_key = "|".join(sorted(clients_with_target))
            
            params_list = [all_client_params[cid] for cid in clients_with_target]
            aggregated = self.fedavg_aggregate_params(params_list)
            curr_acc = self._evaluate_with_cache(aggregated, cache_key)
            
            # 边际贡献
            marginals[client_id] = curr_acc - prev_acc
            prev_acc = curr_acc
        
        return marginals
    
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

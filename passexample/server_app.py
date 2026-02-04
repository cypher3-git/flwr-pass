"""PASS Algorithm - Server Application
使用新版 Flower ServerApp API 实现 PASS 服务端策略
"""

from collections import defaultdict
from typing import Dict, List, Optional, Callable
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from passexample.task import (
    get_model,
    load_centralized_dataset,
    test,
    serialize_updates_dict,
    serialize_state_dict,
    compute_update,
)

# Create ServerApp
app = ServerApp()


class PASSState:
    """PASS 算法状态管理"""
    def __init__(self, alpha: float = 0.95, beta: float = 1.75):
        self.alpha = alpha
        self.beta = beta
        self.contribution_scores: Dict[str, float] = defaultdict(lambda: 0.5)
        self.previous_updates: Dict[str, dict] = {}
        self.previous_global_params: Optional[dict] = None
        self.current_round: int = 0


pass_state = PASSState()


class PASSStrategy(FedAvg):
    """自定义 PASS 策略，继承 FedAvg 并重写聚合逻辑"""
    
    def __init__(self, dataset: str, lr: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.lr = lr
    
    def configure_fit(self, server_round: int, parameters, client_manager):
        """配置训练轮次，动态注入参数审计配置"""
        # 构建配置字典
        config = {"lr": self.lr}
        
        # 注入参数审计配置
        if pass_state.previous_updates:
            config["other_updates"] = serialize_updates_dict(pass_state.previous_updates)
        else:
            config["other_updates"] = ""
        
        if pass_state.previous_global_params is not None:
            config["previous_global_params"] = serialize_state_dict(pass_state.previous_global_params)
        else:
            config["previous_global_params"] = ""
        
        # 采样客户端
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # 返回配置
        from flwr.common import FitIns
        return [(client, FitIns(parameters, config)) for client in clients]
    
    def aggregate_fit(self, server_round: int, results, failures):
        """重写聚合方法，使用 PASS 聚合逻辑"""
        if not results:
            return None, {}
        
        # 更新当前轮次
        pass_state.current_round = server_round
        
        # 使用 PASS 聚合函数
        aggregated_arrays = pass_aggregate(results, self.dataset)
        
        # 计算聚合的 metrics
        aggregated_metrics = {}
        if results:
            # 聚合训练损失
            total_examples = sum([r.metrics.get("num-examples", 0) for r in results])
            if total_examples > 0:
                weighted_loss = sum([
                    r.metrics.get("train_loss", 0) * r.metrics.get("num-examples", 0)
                    for r in results
                ])
                aggregated_metrics["train_loss"] = weighted_loss / total_examples
            
            # 聚合 partition_id (用于调试)
            partition_ids = [r.metrics.get("partition_id", 0) for r in results]
            aggregated_metrics["partition_id"] = sum(partition_ids) / len(partition_ids)
        
        return aggregated_arrays, aggregated_metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    """ServerApp 主入口"""
    
    # 读取配置（从 pyproject.toml 的 [tool.flwr.app.config] 读取）
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    dataset: str = context.run_config["dataset"]
    batch_size: int = context.run_config["batch-size"]
    alpha: float = context.run_config["alpha"]
    beta: float = context.run_config["beta"]
    
    # 初始化 PASS 状态
    global pass_state
    pass_state = PASSState(alpha=alpha, beta=beta)
    
    # 加载全局模型
    global_model = get_model(dataset)
    arrays = ArrayRecord(global_model.state_dict())
    
    print("=" * 60)
    print("PASS Algorithm - Federated Learning")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Learning rate: {lr}")
    print(f"PASS Parameters: α={alpha}, β={beta}")
    print("=" * 60)
    
    # 创建评估函数
    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        return global_evaluate(server_round, arrays, dataset, batch_size)
    
    # 初始化 PASS 策略
    strategy = PASSStrategy(dataset=dataset, lr=lr, fraction_evaluate=fraction_evaluate)
    
    # 使用 FedAvg.start() 运行训练
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )
    
    # 保存最终模型
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    print("Training complete!")


def pass_aggregate(results, dataset: str) -> ArrayRecord:
    """PASS 聚合函数，实现贡献评估和搭便车者剔除"""
    
    if not results:
        return None
    
    # 收集 AccDiv 和客户端更新
    acc_div_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
    client_updates: Dict[str, tuple] = {}
    
    for result in results:
        metrics = result.metrics
        arrays = result.arrays
        
        partition_id = str(metrics.get("partition_id", "unknown"))
        num_examples = metrics.get("num-examples", 1)
        
        state_dict = arrays.to_torch_state_dict()
        client_updates[partition_id] = (state_dict, num_examples)
        
        for key, value in metrics.items():
            if key.startswith("acc_div_"):
                other_cid = key[len("acc_div_"):]
                acc_div_matrix[other_cid][partition_id] = float(value)
    
    # 更新贡献分数
    update_contribution_scores(acc_div_matrix, list(client_updates.keys()))
    
    # 计算阈值
    num_clients = len(client_updates)
    threshold = 1.0 / (pass_state.beta * num_clients) if num_clients > 0 else 0.0
    
    # 过滤搭便车者
    valid_updates = []
    eliminated_clients = []
    
    for cid, (state_dict, num_examples) in client_updates.items():
        score = pass_state.contribution_scores[cid]
        if score >= threshold:
            valid_updates.append((state_dict, num_examples))
        else:
            eliminated_clients.append(cid)
    
    if eliminated_clients:
        print(f"\n{'='*60}")
        print(f"🚨 FREE-RIDER DETECTED AND ELIMINATED 🚨")
        print(f"Round: {pass_state.current_round}")
        print(f"Eliminated clients: {eliminated_clients}")
        print(f"Threshold: {threshold:.4f}")
        for cid in eliminated_clients:
            print(f"  - Client {cid}: score = {pass_state.contribution_scores[cid]:.4f}")
        print(f"{'='*60}\n")
    
    if not valid_updates:
        print("Warning: All clients eliminated, using all updates")
        valid_updates = [(sd, ne) for sd, ne in client_updates.values()]
    
    # 聚合参数
    aggregated_state_dict = aggregate_parameters(valid_updates)
    
    # 更新 PASS 状态
    pass_state.previous_updates = {}
    for cid, (state_dict, _) in client_updates.items():
        if pass_state.previous_global_params is not None:
            update = compute_update(pass_state.previous_global_params, state_dict)
        else:
            update = {k: torch.zeros_like(v) for k, v in state_dict.items()}
        pass_state.previous_updates[cid] = update
    
    pass_state.previous_global_params = aggregated_state_dict
    
    return ArrayRecord(aggregated_state_dict)


def update_contribution_scores(
    acc_div_matrix: Dict[str, Dict[str, float]],
    all_clients: List[str]
) -> None:
    """更新贡献分数
    
    c_i^r = α × c_i^(r-1) + (1-α) × tanh(mean(AccDiv))
    """
    for cid in all_clients:
        if cid in acc_div_matrix and acc_div_matrix[cid]:
            acc_divs = list(acc_div_matrix[cid].values())
            mean_acc_div = np.mean(acc_divs)
            new_contribution = np.tanh(mean_acc_div)
            
            old_score = pass_state.contribution_scores[cid]
            new_score = pass_state.alpha * old_score + (1 - pass_state.alpha) * new_contribution
            pass_state.contribution_scores[cid] = new_score


def aggregate_parameters(updates: List[tuple]) -> dict:
    """加权平均聚合参数"""
    total_examples = sum(num_examples for _, num_examples in updates)
    
    aggregated = None
    for state_dict, num_examples in updates:
        weight = num_examples / total_examples
        if aggregated is None:
            aggregated = {k: v.float() * weight for k, v in state_dict.items()}
        else:
            for k, v in state_dict.items():
                aggregated[k] += v.float() * weight
    
    return aggregated


def global_evaluate(server_round: int, arrays: ArrayRecord, dataset: str, batch_size: int) -> MetricRecord:
    """基于中央数据评估模型"""
    
    model = get_model(dataset)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_dataloader, img_key = load_centralized_dataset(dataset, batch_size)
    test_loss, test_acc = test(model, test_dataloader, device, img_key)
    
    print(f"[Round {server_round}] Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
    
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})

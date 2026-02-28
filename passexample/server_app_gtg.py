"""PASS + GTG-Shapley - Server Application
集成 GTG-Shapley 贡献评估的服务器端实现
"""

from collections import defaultdict
from typing import Dict, List
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp

from passexample.task import (
    get_model,
    load_centralized_dataset,
    test,
    serialize_updates_dict,
    serialize_state_dict,
)

# Create ServerApp
app = ServerApp()


class PASSGTGState:
    """PASS + GTG-Shapley 算法状态管理"""
    def __init__(self, alpha: float = 0.8, beta: float = 1.75):
        self.alpha = alpha  # 贡献分数移动平均系数
        self.beta = beta    # 阈值系数
        # 初始化贡献分数为 1.0
        self.contribution_scores: Dict[str, float] = defaultdict(lambda: 1.0)
        self.previous_client_params: Dict[str, dict] = {}  # 上一轮客户端参数（用于 GTG-Shapley）
        self.initial_global_params: dict = {}  # 初始模型（第0轮）
        self.cumulative_shapley: Dict[str, float] = defaultdict(float)  # 累积 Shapley 值


pass_gtg_state = PASSGTGState()


@app.main()
def main(grid: Grid, context: Context):
    """PASS + GTG-Shapley 主函数"""
    
    # 获取配置
    num_rounds = context.run_config["num-server-rounds"]
    dataset = context.run_config["dataset"]
    lr = context.run_config["learning-rate"]
    batch_size = context.run_config["batch-size"]
    alpha = context.run_config.get("alpha", 0.8)
    beta = context.run_config.get("beta", 1.75)
    
    # 初始化 PASS 状态
    pass_gtg_state.alpha = alpha
    pass_gtg_state.beta = beta
    
    # 初始化全局模型
    model = get_model(dataset)
    global_params = model.state_dict()
    
    # 保存初始模型（用于 GTG-Shapley 基准）
    pass_gtg_state.initial_global_params = {k: v.clone().cpu() for k, v in global_params.items()}
    
    # 加载中心化测试数据
    testloader, img_key = load_centralized_dataset(dataset, batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def global_evaluate(round_num: int, arrays: ArrayRecord, dataset: str, batch_size: int):
        """全局评估函数"""
        model = get_model(dataset)
        model.load_state_dict(arrays.to_torch_state_dict())
        testloader, img_key = load_centralized_dataset(dataset, batch_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss, accuracy = test(model, testloader, device, img_key)
        print(f"[Round {round_num}] Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        return {"accuracy": accuracy, "loss": loss}
    
    print("=" * 60)
    print("PASS + GTG-Shapley - Federated Learning (Fixed)")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Learning rate: {lr}")
    print(f"GTG-Shapley Parameters: α={alpha}, β={beta}")
    print("=" * 60)
    
    # 初始评估
    arrays = ArrayRecord(global_params)
    eval_result = global_evaluate(0, arrays, dataset, batch_size)
    print(f"Initial accuracy: {eval_result.get('accuracy', 0):.4f}")
    print(f"Initial model saved as baseline for GTG-Shapley")
    
    # 主训练循环
    for server_round in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"[ROUND {server_round}/{num_rounds}]")
        print(f"{'='*60}")
        
        # 获取所有节点
        node_ids = list(grid.get_node_ids())
        
        # ============================================================
        # 阶段 1: 训练阶段 - 客户端训练并返回参数
        # ============================================================
        train_config = ConfigRecord({
            "lr": lr,
            "phase": "train",
        })
        
        # 发送训练消息
        train_messages = []
        for node_id in node_ids:
            content = RecordDict({
                "arrays": ArrayRecord(global_params),
                "config": train_config
            })
            msg = Message(content=content, dst_node_id=node_id, message_type="train")
            train_messages.append(msg)
        
        # 发送并等待响应
        train_responses = list(grid.send_and_receive(train_messages))
        
        # 收集客户端训练后的模型参数
        client_params: Dict[str, dict] = {}
        client_info: List[dict] = []
        
        for response in train_responses:
            metric_record = response.content["metrics"]
            arrays_record = response.content["arrays"]
            partition_id = str(int(metric_record.get("partition_id", -1)))
            train_loss = metric_record.get("train_loss", 0.0)
            num_examples = metric_record.get("num-examples", 1)
            
            # 获取客户端训练后的模型参数（FedAvg 标准方式）
            params = arrays_record.to_torch_state_dict()
            client_params[partition_id] = params
            
            client_info.append({
                "partition_id": partition_id,
                "train_loss": train_loss,
                "num_examples": num_examples,
                "params": params,
            })
        
        # ============================================================
        # 阶段 2: FedAvg 加权平均聚合参数
        # ============================================================
        # 使用 FedAvg 加权平均聚合客户端参数
        new_global_params = fedavg_aggregate(
            [info["params"] for info in client_info],
            [info["num_examples"] for info in client_info]
        )
        
        # ============================================================
        # 阶段 3: 审计阶段 - 客户端使用 GTG-Shapley 互评
        # ============================================================
        if pass_gtg_state.previous_client_params:
            # 序列化上一轮的客户端参数（用于 GTG-Shapley）
            client_params_encoded = serialize_updates_dict(pass_gtg_state.previous_client_params)
            # 使用初始模型作为基准
            initial_params_encoded = serialize_state_dict(pass_gtg_state.initial_global_params)
            
            audit_config = ConfigRecord({
                "lr": lr,
                "phase": "audit",
                "other_client_params": client_params_encoded,  # 客户端参数
                "initial_global_params": initial_params_encoded,
            })
            
            # 发送审计消息（发送新全局模型）
            audit_messages = []
            for node_id in node_ids:
                content = RecordDict({
                    "arrays": ArrayRecord(new_global_params),
                    "config": audit_config
                })
                msg = Message(content=content, dst_node_id=node_id, message_type="train")
                audit_messages.append(msg)
            
            # 发送并等待响应
            audit_responses = list(grid.send_and_receive(audit_messages))
            
            # 收集 Shapley 值
            shapley_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
            
            for response in audit_responses:
                metric_record = response.content["metrics"]
                auditor_id = str(int(metric_record.get("partition_id", -1)))
                
                # 收集 Shapley 值
                for key in metric_record.keys():
                    if key.startswith("shapley_"):
                        audited_cid = key[len("shapley_"):]
                        shapley_value = float(metric_record.get(key, 0.0))
                        shapley_matrix[audited_cid][auditor_id] = shapley_value
            
            # 打印 Shapley 值统计
            if shapley_matrix:
                print(f"\n[Round {server_round}] GTG-Shapley Values:")
                for cid in sorted(shapley_matrix.keys(), key=lambda x: int(x)):
                    shapley_values = list(shapley_matrix[cid].values())
                    if shapley_values:
                        mean_sv = np.mean(shapley_values)
                        std_sv = np.std(shapley_values)
                        print(f"  Client {cid}: mean = {mean_sv:.6f}, std = {std_sv:.6f}")
            
            # 更新贡献分数
            all_clients = [info["partition_id"] for info in client_info]
            if shapley_matrix:
                update_contribution_scores_from_shapley(shapley_matrix, all_clients)
        
        # 保存本轮的客户端参数，用于下一轮审计
        pass_gtg_state.previous_client_params = client_params
        
        # ============================================================
        # 阶段 4: 打印贡献分数并剔除搭便车者
        # ============================================================
        num_clients = len(client_info)
        threshold = 1.0 / (pass_gtg_state.beta * num_clients) if num_clients > 0 else 0.0
        
        print(f"\n[Round {server_round}] Contribution Scores (threshold={threshold:.4f}):")
        eliminated_clients = []
        for info in client_info:
            partition_id = info["partition_id"]
            score = pass_gtg_state.contribution_scores[partition_id]
            
            status = ""
            if score < threshold:
                eliminated_clients.append(partition_id)
                status = " ⚠️ ELIMINATED"
            
            print(f"  Client {partition_id}: score = {score:.4f}{status}")
        
        if eliminated_clients:
            print(f"\n🚨 FREE-RIDERS DETECTED: {eliminated_clients}")
        
        # 更新全局模型
        global_params = new_global_params
        
        # 评估
        arrays = ArrayRecord(global_params)
        eval_result = global_evaluate(server_round, arrays, dataset, batch_size)
    
    # 保存最终模型
    print("\nSaving final model to disk...")
    torch.save(global_params, "final_model_gtg.pt")
    print("Training complete!")


def fedavg_aggregate(params_list: List[dict], weights: List[int]) -> dict:
    """FedAvg 加权平均聚合参数
    
    按照 GTG-Shapley 源码的方式进行参数聚合：
    aggregated_param = sum(weight_i * param_i) / sum(weight_i)
    
    Args:
        params_list: 客户端模型参数列表
        weights: 聚合权重（样本数量）
        
    Returns:
        聚合后的模型参数
    """
    if not params_list:
        return {}
    
    total_weight = sum(weights)
    aggregated = {}
    
    # 初始化
    for key in params_list[0]:
        aggregated[key] = torch.zeros_like(params_list[0][key], dtype=torch.float64)
    
    # 加权聚合
    for params, weight in zip(params_list, weights):
        for key in aggregated:
            if key in params:
                aggregated[key] += (weight / total_weight) * params[key].to(torch.float64)
    
    # 转换回原始数据类型
    for key in aggregated:
        aggregated[key] = aggregated[key].to(params_list[0][key].dtype)
    
    return aggregated


def update_contribution_scores_from_shapley(
    shapley_matrix: Dict[str, Dict[str, float]],
    all_clients: List[str]
):
    """基于 Shapley 值更新贡献分数
    
    使用累积 Shapley 值和相对排名：
    1. 累积每轮的 Shapley 值
    2. 基于累积值进行排名
    3. 使用百分位数作为贡献分数
    
    Args:
        shapley_matrix: Shapley 值矩阵 {被评估客户端: {评估者: Shapley值}}
        all_clients: 所有客户端ID列表
    """
    # 计算每个客户端的平均 Shapley 值
    current_shapley = {}
    for cid in all_clients:
        if cid in shapley_matrix and shapley_matrix[cid]:
            shapley_values = list(shapley_matrix[cid].values())
            mean_shapley = np.mean(shapley_values)
            current_shapley[cid] = mean_shapley
        else:
            current_shapley[cid] = 0.0
    
    # 累积 Shapley 值
    for cid in all_clients:
        pass_gtg_state.cumulative_shapley[cid] += current_shapley[cid]
    
    # 基于累积 Shapley 值排序
    sorted_clients = sorted(
        all_clients, 
        key=lambda x: pass_gtg_state.cumulative_shapley[x]
    )
    
    # 使用相对排名作为贡献分数
    # 排名最低的客户端分数接近 0，排名最高的接近 1
    n = len(sorted_clients)
    for rank, cid in enumerate(sorted_clients):
        # 百分位数分数：rank / (n - 1)，范围 [0, 1]
        if n > 1:
            rank_score = rank / (n - 1)
        else:
            rank_score = 1.0
        
        # 使用移动平均平滑分数变化
        old_score = pass_gtg_state.contribution_scores[cid]
        new_score = pass_gtg_state.alpha * old_score + \
                   (1 - pass_gtg_state.alpha) * rank_score
        
        pass_gtg_state.contribution_scores[cid] = new_score

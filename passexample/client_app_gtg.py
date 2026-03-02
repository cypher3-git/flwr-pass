"""PASS + GTG-Shapley - Client Application
集成 GTG-Shapley 贡献评估的客户端实现
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from passexample.task import (
    get_model,
    load_data,
    train as train_fn,
    test as test_fn,
    deserialize_updates_dict,
    deserialize_state_dict,
)
from passexample.gtg_shapley import GTGShapley

# Flower ClientApp
app = ClientApp()

# AFR 和 SFR 客户端 ID
AFR_CLIENT_ID = 2
SFR_CLIENT_ID = 7  # 禁用


@app.train()
def train(msg: Message, context: Context):
    """执行本地训练和 GTG-Shapley 审计"""
    
    # 获取配置
    dataset = context.run_config["dataset"]
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    lr = context.run_config["learning-rate"]
    
    # GTG-Shapley 参数
    gtg_sampling_rounds = context.run_config.get("gtg-sampling-rounds", 10)
    gtg_within_threshold = context.run_config.get("gtg-within-threshold", 0.01)
    gtg_between_threshold = context.run_config.get("gtg-between-threshold", 0.005)
    gtg_convergence_threshold = context.run_config.get("gtg-convergence-threshold", 0.05)
    gtg_guided_sampling_m = context.run_config.get("gtg-guided-sampling-m", 1)
    
    # 加载模型
    model = get_model(dataset)
    current_state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(current_state_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 保存训练前的参数
    old_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    
    # 加载数据
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, testloader, img_key = load_data(partition_id, num_partitions, batch_size, dataset)
    
    # 获取阶段和配置
    config_dict = msg.content.get("config", {})
    phase = config_dict.get("phase", "train")
    
    metrics = {}
    metrics["partition_id"] = partition_id
    
    if phase == "audit":
        # ============================================================
        # 阶段 2: 审计阶段 - 使用 GTG-Shapley 评估其他客户端
        # 按照论文设计：使用梯度更新 Δ_i 和基础模型 M^{(t)}
        # ============================================================
        
        client_deltas_encoded = config_dict.get("client_deltas", "")
        base_model_encoded = config_dict.get("base_model", "")
        data_sizes_encoded = config_dict.get("data_sizes", "")
        
        if client_deltas_encoded and base_model_encoded:
            shapley_values = gtg_shapley_audit(
                model=model,
                client_deltas_encoded=client_deltas_encoded,
                base_model_encoded=base_model_encoded,
                data_sizes_encoded=data_sizes_encoded,
                testloader=testloader,
                device=device,
                partition_id=partition_id,
                num_sampling_rounds=gtg_sampling_rounds,
                within_threshold=gtg_within_threshold,
                between_threshold=gtg_between_threshold,
                convergence_threshold=gtg_convergence_threshold,
                guided_sampling_m=gtg_guided_sampling_m,
            )
            
            # 将 Shapley 值添加到 metrics
            for other_cid, sv in shapley_values.items():
                metrics[f"shapley_{other_cid}"] = float(sv)
        
        # 审计阶段不返回模型更新
        model_record = ArrayRecord(current_state_dict)
        
    else:
        # ============================================================
        # 阶段 1: 训练阶段
        # ============================================================
        if partition_id == AFR_CLIENT_ID:
            # AFR: 不训练，返回原始参数加小噪声（模拟搭便车者）
            print(f"[Client {partition_id}] AFR - Returning original params with noise")
            new_state_dict = {}
            for key, param in old_state_dict.items():
                # 添加很小的随机噪声，模拟搭便车者
                new_state_dict[key] = param + torch.randn_like(param) * 0.001
            train_loss = 0.0
            
        elif partition_id == SFR_CLIENT_ID:
            # SFR: 在错误数据集上训练（当前禁用）
            print(f"[Client {partition_id}] SFR - Training on wrong dataset")
            train_loss = 0.0
            new_state_dict = old_state_dict
            
        else:
            # 正常客户端训练，返回训练后的模型参数
            train_loss = train_fn(model, trainloader, local_epochs, lr, device, img_key)
            new_state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        
        metrics["train_loss"] = train_loss
        metrics["num-examples"] = len(trainloader.dataset)
        
        # 返回训练后的模型参数（FedAvg 标准方式）
        model_record = ArrayRecord(new_state_dict)
    
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """评估模型"""
    
    dataset = context.run_config.get("dataset", "cifar10")
    batch_size = context.run_config.get("batch-size", 64)
    
    model = get_model(dataset)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, testloader, img_key = load_data(partition_id, num_partitions, batch_size, dataset)
    
    eval_loss, eval_acc = test_fn(model, testloader, device, img_key)
    
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(testloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    return Message(content=content, reply_to=msg)


def gtg_shapley_audit(
    model,
    client_deltas_encoded: str,
    base_model_encoded: str,
    data_sizes_encoded: str,
    testloader,
    device,
    partition_id: int,
    num_sampling_rounds: int,
    within_threshold: float,
    between_threshold: float,
    convergence_threshold: float = 0.05,
    guided_sampling_m: int = 1,
) -> dict:
    """使用 GTG-Shapley 进行互评
    
    严格按照论文 Algorithm 1 实现：
    - 轮间截断 (Between-round Truncation)
    - 轮内截断 (Within-round Truncation)
    - 引导采样 (Guided Sampling)
    - 基于梯度更新的子模型重构: M̃_S = M^{(t)} + Σ_{i∈S} (|D_i|/|D_S|) * Δ_i
    
    Args:
        model: 神经网络模型
        client_deltas_encoded: 客户端梯度更新 Δ_i（序列化）
        base_model_encoded: 基础模型 M^{(t)}（序列化）
        data_sizes_encoded: 客户端数据集大小 |D_i|（序列化）
        testloader: 测试数据加载器
        device: 计算设备
        partition_id: 当前客户端ID
        num_sampling_rounds: Monte Carlo 最大采样轮数
        within_threshold: ε_i - 轮内截断阈值
        between_threshold: ε_b - 轮间截断阈值
        convergence_threshold: 蒙特卡洛收敛判定阈值
        guided_sampling_m: 引导采样中前 m 个位置的循环占据数
        
    Returns:
        其他客户端的 Shapley 值字典
    """
    shapley_values = {}
    
    try:
        # 反序列化客户端梯度更新 Δ_i
        client_deltas = deserialize_updates_dict(client_deltas_encoded)
    except Exception as e:
        print(f"[GTG-Shapley] Failed to deserialize client_deltas: {e}")
        return shapley_values
    
    if not client_deltas:
        return shapley_values
    
    # 反序列化基础模型 M^{(t)}
    try:
        base_model = deserialize_state_dict(base_model_encoded)
    except Exception as e:
        print(f"[GTG-Shapley] Failed to deserialize base_model: {e}")
        return shapley_values
    
    # 反序列化数据集大小 |D_i|
    data_sizes = {}
    try:
        if data_sizes_encoded:
            data_sizes_raw = deserialize_state_dict(data_sizes_encoded)
            data_sizes = {k: int(v.item()) for k, v in data_sizes_raw.items()}
    except Exception as e:
        print(f"[GTG-Shapley] Failed to deserialize data_sizes: {e}")
        # 默认等权
        data_sizes = {cid: 1 for cid in client_deltas.keys()}
    
    # 初始化 GTG-Shapley 计算器
    gtg = GTGShapley(
        model=model,
        test_loader=testloader,
        device=device,
        learning_rate=0.01,  # 不再需要，但保留接口兼容
        num_sampling_rounds=num_sampling_rounds,
        within_round_threshold=within_threshold,
        between_round_threshold=between_threshold,
        convergence_threshold=convergence_threshold,
        guided_sampling_m=guided_sampling_m,
    )
    
    # 使用论文公式计算 Shapley 值（基于梯度更新重构子模型）
    shapley_values = gtg.evaluate_all_clients_with_deltas(
        base_model=base_model,
        client_deltas=client_deltas,
        data_sizes=data_sizes,
        auditor_id=str(partition_id),
    )
    
    return shapley_values

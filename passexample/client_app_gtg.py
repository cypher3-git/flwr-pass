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
SFR_CLIENT_ID = -1  # 禁用


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
    gtg_between_threshold = context.run_config.get("gtg-between-threshold", 0.05)
    
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
        # ============================================================
        pass  # Audit phase
        
        other_client_params_encoded = config_dict.get("other_client_params", "")
        initial_params_encoded = config_dict.get("initial_global_params", "")
        
        if other_client_params_encoded and initial_params_encoded:
            shapley_values = gtg_shapley_audit(
                model=model,
                current_state_dict=current_state_dict,
                other_client_params_encoded=other_client_params_encoded,
                initial_params_encoded=initial_params_encoded,
                testloader=testloader,
                device=device,
                img_key=img_key,
                partition_id=partition_id,
                lr=lr,
                num_sampling_rounds=gtg_sampling_rounds,
                within_threshold=gtg_within_threshold,
                between_threshold=gtg_between_threshold,
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
    current_state_dict: dict,
    other_client_params_encoded: str,
    initial_params_encoded: str,
    testloader,
    device,
    img_key: str,
    partition_id: int,
    lr: float,
    num_sampling_rounds: int,
    within_threshold: float,
    between_threshold: float,
) -> dict:
    """使用 GTG-Shapley 进行互评
    
    按照 GTG-Shapley 源码的方式：
    - 客户端发送训练后的模型参数
    - Shapley 值基于参数子集聚合后的模型准确率计算
    
    Args:
        model: 神经网络模型
        current_state_dict: 当前全局模型参数
        other_client_params_encoded: 其他客户端的模型参数（序列化）
        initial_params_encoded: 初始全局模型参数（序列化）- 用作基准
        testloader: 测试数据加载器
        device: 计算设备
        img_key: 图像键
        partition_id: 当前客户端ID
        lr: 学习率
        num_sampling_rounds: Monte Carlo 采样轮数
        within_threshold: Within-round 截断阈值
        between_threshold: Between-round 截断阈值
        
    Returns:
        其他客户端的 Shapley 值字典
    """
    shapley_values = {}
    
    try:
        # 反序列化其他客户端的模型参数
        other_client_params = deserialize_updates_dict(other_client_params_encoded)
    except Exception as e:
        return shapley_values
    
    if not other_client_params:
        return shapley_values
    
    # 获取初始全局参数（用作 GTG-Shapley 基准）
    try:
        initial_params = deserialize_state_dict(initial_params_encoded)
    except Exception as e:
        initial_params = current_state_dict
    
    # 初始化 GTG-Shapley 计算器
    gtg = GTGShapley(
        model=model,
        test_loader=testloader,
        device=device,
        learning_rate=lr,
        num_sampling_rounds=num_sampling_rounds,
        within_round_threshold=within_threshold,
        between_round_threshold=between_threshold,
    )
    
    # 计算每个客户端的 Shapley 值（基于参数子集聚合）
    shapley_values = gtg.evaluate_all_clients_fedavg(
        initial_params=initial_params,
        all_client_params=other_client_params,
        auditor_id=str(partition_id),
    )
    
    return shapley_values

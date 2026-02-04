"""PASS Algorithm - Client Application
使用新版 Flower ClientApp API 实现 PASS 客户端
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from passexample.task import (
    get_model,
    load_data,
    train as train_fn,
    test as test_fn,
    add_gaussian_noise,
    apply_pruning,
    compute_update,
    apply_update,
    deserialize_updates_dict,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """执行本地训练和参数审计"""
    
    # 获取配置（从 pyproject.toml 的 [tool.flwr.app.config] 读取）
    dataset = context.run_config["dataset"]
    sigma_squared = context.run_config["sigma-squared"]
    gamma = context.run_config["gamma"]
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    lr = context.run_config["learning-rate"]
    
    # 加载模型
    model = get_model(dataset)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 保存训练前的参数
    old_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
    # 加载数据
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, testloader, img_key = load_data(partition_id, num_partitions, batch_size, dataset)
    
    # 参数审计
    metrics = {}
    config_dict = msg.content.get("config", {})
    other_updates_encoded = config_dict.get("other_updates", "")
    previous_params_encoded = config_dict.get("previous_global_params", "")
    
    if other_updates_encoded:
        acc_div_results = parameter_audit(
            model=model,
            current_state_dict=old_state_dict,
            other_updates_encoded=other_updates_encoded,
            previous_params_encoded=previous_params_encoded,
            testloader=testloader,
            device=device,
            img_key=img_key,
            partition_id=partition_id,
        )
        for other_cid, acc_div in acc_div_results.items():
            metrics[f"acc_div_{other_cid}"] = float(acc_div)
    
    # 模拟搭便车者：partition_id = 0 的客户端不进行训练
    if partition_id == 0:
        print(f"[Client {partition_id}] Acting as FREE-RIDER (no training)")
        # 不训练，直接使用全局模型参数（零更新）
        new_state_dict = old_state_dict
        train_loss = 0.0
    else:
        # 本地训练
        train_loss = train_fn(model, trainloader, local_epochs, lr, device, img_key)
        # 获取训练后的参数
        new_state_dict = model.state_dict()
    
    # 计算更新量
    update = compute_update(old_state_dict, new_state_dict)
    
    # PASS-PPS: 添加高斯噪声
    noisy_update = add_gaussian_noise(update, sigma_squared)
    
    # PASS-PPS: 应用剪枝
    pruned_update = apply_pruning(noisy_update, gamma)
    
    # 应用处理后的更新
    final_state_dict = apply_update(old_state_dict, pruned_update)
    
    # 构造返回消息
    model_record = ArrayRecord(final_state_dict)
    metrics["train_loss"] = train_loss
    metrics["num-examples"] = len(trainloader.dataset)
    metrics["partition_id"] = partition_id
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """评估模型"""
    
    dataset = context.run_config.get("dataset", "mnist")
    batch_size = context.run_config.get("batch-size", 32)
    
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


def parameter_audit(
    model,
    current_state_dict: dict,
    other_updates_encoded: str,
    previous_params_encoded: str,
    testloader,
    device,
    img_key: str,
    partition_id: int,
) -> dict:
    """执行参数审计，计算 AccDiv
    
    AccDiv = Acc(θ^r) - Acc(Δθ_j + θ^(r-1))
    """
    acc_div_results = {}
    
    try:
        other_updates = deserialize_updates_dict(other_updates_encoded)
    except Exception:
        return acc_div_results
    
    if not other_updates:
        return acc_div_results
    
    # 计算当前全局模型的准确率
    model.load_state_dict(current_state_dict)
    model.to(device)
    _, current_acc = test_fn(model, testloader, device, img_key)
    
    # 获取上一轮全局参数
    if previous_params_encoded:
        try:
            from passexample.task import deserialize_state_dict
            previous_params = deserialize_state_dict(previous_params_encoded)
        except Exception:
            previous_params = current_state_dict
    else:
        previous_params = current_state_dict
    
    # 采样审计：每个客户端只随机审计1个其他客户端以减少内存消耗
    import random
    other_cids = [cid for cid in other_updates.keys() if str(cid) != str(partition_id)]
    
    if other_cids:
        # 随机选择1个客户端进行审计
        selected_cid = random.choice(other_cids)
        update = other_updates[selected_cid]
        
        try:
            # 将更新应用到上一轮参数
            applied_params = apply_update(previous_params, update)
            model.load_state_dict(applied_params)
            model.to(device)
            _, other_acc = test_fn(model, testloader, device, img_key)
            
            # 计算准确率差异
            acc_div = current_acc - other_acc
            acc_div_results[str(selected_cid)] = acc_div
            print(f"[Client {partition_id}] Audited Client {selected_cid}: AccDiv = {acc_div:.4f}")
        except Exception as e:
            print(f"[Client {partition_id}] Audit failed: {e}")
    
    # 恢复模型参数
    model.load_state_dict(current_state_dict)
    
    return acc_div_results

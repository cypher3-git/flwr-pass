"""PASS Algorithm - Task Module
模型定义、数据加载和训练/测试函数
"""

import pickle
import base64
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


# ==================== 模型定义 ====================

class MNISTNet(nn.Module):
    """2层 CNN 用于 MNIST 数据集"""
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CIFAR10Net(nn.Module):
    """3层 CNN 用于 CIFAR-10 数据集"""
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(dataset: str) -> nn.Module:
    """根据数据集名称获取对应模型"""
    if dataset.lower() == "mnist":
        return MNISTNet()
    elif dataset.lower() == "cifar10":
        return CIFAR10Net()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ==================== 数据加载 ====================

fds_mnist = None
fds_cifar10 = None

mnist_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
cifar10_transforms = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


def apply_mnist_transforms(batch):
    """Apply transforms to MNIST data."""
    batch["image"] = [mnist_transforms(img.convert("L")) for img in batch["image"]]
    return batch


def apply_cifar10_transforms(batch):
    """Apply transforms to CIFAR-10 data."""
    batch["img"] = [cifar10_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int, dataset: str, test_size: float = 0.2):
    """Load partition data for MNIST or CIFAR-10."""
    global fds_mnist, fds_cifar10
    
    if dataset.lower() == "mnist":
        if fds_mnist is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds_mnist = FederatedDataset(
                dataset="ylecun/mnist",
                partitioners={"train": partitioner},
            )
        partition = fds_mnist.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=test_size, seed=42)
        partition_train_test = partition_train_test.with_transform(apply_mnist_transforms)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        img_key = "image"
    else:
        if fds_cifar10 is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds_cifar10 = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        partition = fds_cifar10.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=test_size, seed=42)
        partition_train_test = partition_train_test.with_transform(apply_cifar10_transforms)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        img_key = "img"
    
    return trainloader, testloader, img_key


def load_centralized_dataset(dataset: str, batch_size: int):
    """Load centralized test set."""
    if dataset.lower() == "mnist":
        test_dataset = load_dataset("ylecun/mnist", split="test")
        dataset_transformed = test_dataset.with_format("torch").with_transform(apply_mnist_transforms)
        img_key = "image"
    else:
        test_dataset = load_dataset("uoft-cs/cifar10", split="test")
        dataset_transformed = test_dataset.with_format("torch").with_transform(apply_cifar10_transforms)
        img_key = "img"
    
    return DataLoader(dataset_transformed, batch_size=batch_size), img_key


# ==================== 训练和测试函数 ====================

def train(model, trainloader, epochs, lr, device, img_key):
    """Train the model on the training set."""
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for batch in trainloader:
            if isinstance(batch, dict):
                images = batch[img_key].to(device)
                labels = batch["label"].to(device)
            else:
                images, labels = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    return loss.item()


def train_and_get_gradients(model, trainloader, epochs, lr, device, img_key):
    """训练模型并返回累积的真实梯度
    
    Returns:
        tuple: (loss, accumulated_gradients)
    """
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    # 初始化累积梯度字典
    accumulated_grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            accumulated_grads[name] = torch.zeros_like(param.data).cpu()
    
    total_batches = 0
    final_loss = 0.0
    
    for epoch in range(epochs):
        for batch in trainloader:
            if isinstance(batch, dict):
                images = batch[img_key].to(device)
                labels = batch["label"].to(device)
            else:
                images, labels = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue
            
            loss.backward()
            
            # 累积梯度（在裁剪之前）
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    accumulated_grads[name] += param.grad.data.cpu()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_batches += 1
            final_loss = loss.item()
    
    # 平均梯度以避免数值爆炸
    # 注意：累积梯度需要除以批次数，否则梯度会过大
    if total_batches > 0:
        for name in accumulated_grads:
            accumulated_grads[name] /= total_batches
    
    # 转换为 state_dict 格式（使用模型参数的键名）
    gradient_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 将 named_parameters 的名称映射到 state_dict 的键
            state_dict_key = name
            if state_dict_key in accumulated_grads:
                gradient_dict[state_dict_key] = accumulated_grads[state_dict_key]
    
    return final_loss, gradient_dict


def test(net, testloader, device, img_key):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch[img_key].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset) if len(testloader.dataset) > 0 else 0.0
    loss = loss / len(testloader) if len(testloader) > 0 else 0.0
    return loss, accuracy


# ==================== PASS-PPS 辅助函数 ====================

def add_gaussian_noise(state_dict: dict, sigma_squared: float) -> dict:
    """向模型参数添加高斯噪声 (弱差分隐私)"""
    sigma = np.sqrt(sigma_squared)
    noisy_state_dict = {}
    for key, param in state_dict.items():
        noise = torch.randn_like(param) * sigma
        noisy_state_dict[key] = param + noise
    return noisy_state_dict


def apply_pruning(state_dict: dict, gamma: float) -> dict:
    """应用参数剪枝"""
    pruned_state_dict = {}
    for key, param in state_dict.items():
        mask = torch.rand_like(param.float()) > gamma
        pruned_state_dict[key] = param * mask.to(param.dtype)
    return pruned_state_dict


def compute_update(old_state_dict: dict, new_state_dict: dict) -> dict:
    """计算参数更新量"""
    update = {}
    for key in old_state_dict.keys():
        update[key] = new_state_dict[key] - old_state_dict[key]
    return update


def apply_update(base_state_dict: dict, update: dict) -> dict:
    """将更新应用到基础参数"""
    result = {}
    for key in base_state_dict.keys():
        result[key] = base_state_dict[key] + update[key]
    return result


# ==================== 序列化辅助函数 ====================

def serialize_state_dict(state_dict: dict) -> str:
    """将 state_dict 序列化为 base64 字符串"""
    cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
    serialized = pickle.dumps(cpu_state_dict)
    encoded = base64.b64encode(serialized).decode('utf-8')
    return encoded


def deserialize_state_dict(encoded: str) -> dict:
    """将 base64 字符串反序列化为 state_dict"""
    decoded = base64.b64decode(encoded.encode('utf-8'))
    state_dict = pickle.loads(decoded)
    return state_dict


def serialize_updates_dict(updates_dict: Dict[str, dict]) -> str:
    """将客户端更新字典序列化"""
    cpu_updates = {}
    for cid, update in updates_dict.items():
        cpu_updates[cid] = {k: v.cpu() for k, v in update.items()}
    serialized = pickle.dumps(cpu_updates)
    encoded = base64.b64encode(serialized).decode('utf-8')
    return encoded


def deserialize_updates_dict(encoded: str) -> Dict[str, dict]:
    """将 base64 字符串反序列化为客户端更新字典"""
    decoded = base64.b64decode(encoded.encode('utf-8'))
    updates_dict = pickle.loads(decoded)
    return updates_dict

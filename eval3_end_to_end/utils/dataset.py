from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import numpy as np

# 全局变量用于存储已加载的数据集和打乱的索引
_loaded_datasets = {
    "MNIST": {"train": None, "test": None, "train_indices": None},
    "CIFAR10": {"train": None, "test": None, "train_indices": None},
}


# 数据集加载器
def get_data_loaders(dataset_name, batch_size, data_dir="./data", weights=None):
    global _loaded_datasets
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if dataset_name == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        if (
            _loaded_datasets["MNIST"]["train"] is None
            or _loaded_datasets["MNIST"]["test"] is None
        ):
            _loaded_datasets["MNIST"]["train"] = datasets.MNIST(
                data_dir, train=True, download=True, transform=transform
            )
            _loaded_datasets["MNIST"]["test"] = datasets.MNIST(
                data_dir, train=False, download=True, transform=transform
            )
            # 打乱索引并存储
            num_samples = len(_loaded_datasets["MNIST"]["train"])
            _loaded_datasets["MNIST"]["train_indices"] = np.random.permutation(
                num_samples
            )

        train_dataset = _loaded_datasets["MNIST"]["train"]
        test_dataset = _loaded_datasets["MNIST"]["test"]
        train_indices = _loaded_datasets["MNIST"]["train_indices"]

    elif dataset_name == "CIFAR10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        if (
            _loaded_datasets["CIFAR10"]["train"] is None
            or _loaded_datasets["CIFAR10"]["test"] is None
        ):
            _loaded_datasets["CIFAR10"]["train"] = datasets.CIFAR10(
                data_dir, train=True, download=True, transform=transform_train
            )
            _loaded_datasets["CIFAR10"]["test"] = datasets.CIFAR10(
                data_dir, train=False, download=True, transform=transform_test
            )
            # 打乱索引并存储
            num_samples = len(_loaded_datasets["CIFAR10"]["train"])
            _loaded_datasets["CIFAR10"]["train_indices"] = np.random.permutation(
                num_samples
            )

        train_dataset = _loaded_datasets["CIFAR10"]["train"]
        test_dataset = _loaded_datasets["CIFAR10"]["test"]
        train_indices = _loaded_datasets["CIFAR10"]["train_indices"]

    else:
        raise ValueError("Unsupported dataset")

    # 按权重划分训练数据集
    train_loaders = []
    start_idx = 0
    num_samples = len(train_dataset)

    # 归一化权重
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    for weight in normalized_weights:
        subset_size = int(num_samples * weight)
        subset_indices = train_indices[start_idx : start_idx + subset_size]
        train_loader = DataLoader(
            Subset(train_dataset, subset_indices),
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        train_loaders.append(train_loader)
        start_idx += subset_size

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loader

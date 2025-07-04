import pandas as pd
import numpy as np
from collections import Counter
from typing import Tuple, Optional, Union


def random_split_dataset(
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: Optional[int] = None,
        stratify: bool = False
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # 验证输入参数的合法性
    def _validate_inputs():
        # 检查数据类型
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"输入数据必须是pandas DataFrame类型，当前类型: {type(data)}")

        # 检查数据是否为空
        if data.empty:
            raise ValueError("输入数据不能为空DataFrame")

        # 检查目标列是否存在
        if target_column not in data.columns:
            raise KeyError(f"目标列 '{target_column}' 不存在于数据中。可用列: {list(data.columns)}")

        # 检查test_size范围
        if not isinstance(test_size, (int, float)):
            raise ValueError(f"test_size必须是数值类型，当前类型: {type(test_size)}")

        if not (0 < test_size < 1):
            raise ValueError(f"test_size必须在(0,1)范围内，当前值: {test_size}")

        # 检查random_state
        if random_state is not None and not isinstance(random_state, int):
            raise ValueError(f"random_state必须是整数或None，当前类型: {type(random_state)}")

        # 检查stratify参数
        if not isinstance(stratify, bool):
            raise ValueError(f"stratify必须是布尔类型，当前类型: {type(stratify)}")

        # 检查数据是否包含缺失值
        if data.isnull().any().any():
            raise ValueError("数据中包含缺失值，请先处理缺失值后再进行划分")
    # 初始化随机数生成器
    def _initialize_random_generator():
        if random_state is not None:
            np.random.seed(random_state)

    # 计算训练集和测试集大小
    def _calculate_split_sizes():
        total_samples = len(data)
        test_samples = int(test_size * total_samples)

        if test_samples == 0:
            raise ValueError(
                f"数据量太小({total_samples}样本)，按{test_size}比例划分测试集为0样本，请增加数据量或调整test_size")

        train_samples = total_samples - test_samples
        if train_samples == 0:
            raise ValueError(f"test_size({test_size})过大，导致训练集为空，请减小test_size值")

        return total_samples, test_samples, train_samples

    # 分层抽样划分
    def _stratified_split(y_values, total_samples, test_samples):
        # 获取类别统计
        class_counts = Counter(y_values)
        unique_classes = list(class_counts.keys())

        # 检查每个类别是否至少有2个样本
        min_class_count = min(class_counts.values())
        if min_class_count < 2:
            raise ValueError(f"分层抽样要求每个类别至少有2个样本，但发现类别样本数最少为: {min_class_count}")

        train_indices = []
        test_indices = []

        for class_label in unique_classes:
            # 获取当前类别的所有索引
            class_indices = [i for i, label in enumerate(y_values) if label == class_label]
            class_size = len(class_indices)

            # 计算当前类别应分配到测试集的样本数
            class_test_size = max(1, int(test_size * class_size))  # 至少1个样本

            # 确保不超过该类别的总样本数-1（训练集至少保留1个）
            if class_test_size >= class_size:
                class_test_size = class_size - 1

            # 随机选择测试集索引
            np.random.shuffle(class_indices)
            test_indices.extend(class_indices[:class_test_size])
            train_indices.extend(class_indices[class_test_size:])

        return train_indices, test_indices

    # 随机划分
    def _random_split(total_samples, test_samples):
        # 创建索引数组并打乱
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        # 划分索引
        test_indices = indices[:test_samples].tolist()
        train_indices = indices[test_samples:].tolist()

        return train_indices, test_indices

    # 根据索引划分数据
    def _split_data_by_indices(train_indices, test_indices):
        # 分离特征和标签
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # 根据索引划分数据，保持DataFrame和Series格式
        X_train = X.iloc[train_indices].reset_index(drop=True)
        X_test = X.iloc[test_indices].reset_index(drop=True)
        y_train = y.iloc[train_indices].reset_index(drop=True)
        y_test = y.iloc[test_indices].reset_index(drop=True)

        return X_train, X_test, y_train, y_test

    # 主函数
    try:
        # 验证输入参数
        _validate_inputs()

        # 初始化随机数生成器
        _initialize_random_generator()

        # 计算划分大小
        total_samples, test_samples, train_samples = _calculate_split_sizes()

        # 生成划分索引
        y_values = data[target_column].tolist()

        if stratify:
            train_indices, test_indices = _stratified_split(y_values, total_samples, test_samples)
        else:
            train_indices, test_indices = _random_split(total_samples, test_samples)

        # 根据索引划分数据
        X_train, X_test, y_train, y_test = _split_data_by_indices(train_indices, test_indices)

        # 验证划分结果
        assert len(X_train) == len(y_train), "训练集特征和标签长度不匹配"
        assert len(X_test) == len(y_test), "测试集特征和标签长度不匹配"
        assert len(X_train) + len(X_test) == len(data), "划分后总样本数不匹配"

        return X_train, X_test, y_train, y_test

    except Exception as e:
        raise e



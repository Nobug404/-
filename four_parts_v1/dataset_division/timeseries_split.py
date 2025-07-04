import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union


def timeseries_split(
        data: pd.DataFrame,
        target_column: str,
        n_splits: int,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None
        ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:

    # 验证输入参数的合法性
    def _validate_inputs(data, target_column, n_splits, test_size, gap, max_train_size):
        # 检查data是否为DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data必须是pandas DataFrame类型，但收到的是 {type(data)}")

        # 检查DataFrame是否为空
        if data.empty:
            raise ValueError("输入的DataFrame不能为空")

        # 检查目标列是否存在
        if target_column not in data.columns:
            raise KeyError(f"目标列 '{target_column}' 不存在于DataFrame中。可用列名: {list(data.columns)}")

        # 检查n_splits
        if not isinstance(n_splits, int) or n_splits <= 0:
            raise ValueError("n_splits必须是正整数")

        # 检查test_size
        if test_size is not None:
            if not isinstance(test_size, int) or test_size <= 0:
                raise ValueError("test_size必须是正整数或None")
            if test_size >= len(data):
                raise ValueError(f"test_size ({test_size}) 不能大于等于数据总长度 ({len(data)})")

        # 检查gap
        if not isinstance(gap, int) or gap < 0:
            raise ValueError("gap必须是非负整数")

        # 检查max_train_size
        if max_train_size is not None:
            if not isinstance(max_train_size, int) or max_train_size <= 0:
                raise ValueError("max_train_size必须是正整数或None")

    # 计算默认的test_size
    def _calculate_test_size(data_length, n_splits, gap):
        # 考虑gap的影响，确保有足够的数据进行划分
        available_length = data_length - (n_splits - 1) * gap
        if available_length <= n_splits:
            raise ValueError(f"数据长度 ({data_length}) 太小，无法进行 {n_splits} 折划分，考虑gap={gap}")

        test_size = available_length // (n_splits + 1)
        if test_size <= 0:
            raise ValueError(f"计算得到的test_size <= 0，请减少n_splits或gap，或增加数据量")

        return test_size

    # 生成所有划分的索引
    def _generate_split_indices(data_length, n_splits, test_size, gap, max_train_size):
        splits = []

        for i in range(n_splits):
            # 计算测试集的起始和结束位置
            test_start = (i + 1) * test_size + i * gap
            test_end = test_start + test_size

            # 检查是否超出数据范围
            if test_end > data_length:
                print(f"警告: 第 {i + 1} 折的测试集超出数据范围，实际只能生成 {i} 折")
                break

            # 计算训练集的起始和结束位置
            train_end = test_start - gap
            train_start = 0

            # 如果指定了max_train_size，应用滚动窗口
            if max_train_size is not None:
                train_start = max(0, train_end - max_train_size)

            # 检查训练集是否有效
            if train_start >= train_end:
                raise ValueError(f"第 {i + 1} 折的训练集无效：train_start={train_start}, train_end={train_end}")

            # 生成索引
            train_indices = list(range(train_start, train_end))
            test_indices = list(range(test_start, test_end))

            splits.append((train_indices, test_indices))

        if len(splits) == 0:
            raise ValueError("无法生成任何有效的划分，请检查参数设置")

        return splits

    # 根据索引创建DataFrame划分
    def _create_dataframe_splits(data, target_column, index_splits):
        result_splits = []

        # 分离特征和目标
        X = data.drop(columns=[target_column])
        y = data[target_column]

        for train_indices, test_indices in index_splits:
            # 创建训练集
            X_train = X.iloc[train_indices].copy()
            y_train = y.iloc[train_indices].copy()

            # 创建测试集
            X_test = X.iloc[test_indices].copy()
            y_test = y.iloc[test_indices].copy()

            # 重置索引以保持连续性，但保留原始索引信息
            X_train = X_train.reset_index(drop=False)
            X_test = X_test.reset_index(drop=False)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            result_splits.append((X_train, X_test, y_train, y_test))

        return result_splits

    # 主函数
    try:
        # 验证输入参数
        _validate_inputs(data, target_column, n_splits, test_size, gap, max_train_size)

        # 获取数据长度
        N = len(data)

        # 计算test_size
        if test_size is None:
            test_size = _calculate_test_size(N, n_splits, gap)

        # 验证计算后的参数
        min_required_length = (n_splits * test_size) + ((n_splits - 1) * gap) + 1
        if N < min_required_length:
            raise ValueError(
                f"数据长度不足: 需要至少 {min_required_length} 行数据，"
                f"但只有 {N} 行。请减少n_splits、test_size或gap"
            )

        # 生成索引划分
        index_splits = _generate_split_indices(N, n_splits, test_size, gap, max_train_size)

        # 创建DataFrame划分
        result_splits = _create_dataframe_splits(data, target_column, index_splits)

        # 打印划分信息
        print(f"成功生成 {len(result_splits)} 折时间序列划分:")
        print(f"- 数据总长度: {N}")
        print(f"- 每折测试集大小: {test_size}")
        print(f"- 间隔gap: {gap}")
        print(f"- 最大训练集大小: {max_train_size if max_train_size else '不限制'}")

        for i, (X_train, X_test, y_train, y_test) in enumerate(result_splits):
            print(f"  第{i + 1}折: 训练集 {len(X_train)} 行, 测试集 {len(X_test)} 行")

        return result_splits

    except Exception as e:
        raise e



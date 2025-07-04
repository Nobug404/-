import pandas as pd
import numpy as np
from collections import Counter
from typing import Union, Any, Dict
import warnings


def statistical_imputation_fill(data: pd.DataFrame,
                                strategy: str = 'mean',
                                fill_value: Any = None,
                                random_state: int = None) -> pd.DataFrame:

    # 验证输入参数的有效性
    def _validate_inputs():
        if not isinstance(data, pd.DataFrame):
            raise TypeError("输入的data必须是pandas DataFrame类型")

        if data.empty:
            raise ValueError("输入的数据表为空，无法进行填充操作")

        valid_strategies = ['mean', 'median', 'most_frequent', 'constant']
        if strategy not in valid_strategies:
            raise ValueError(f"填充策略'{strategy}'无效，支持的策略包括: {valid_strategies}")

        if strategy == 'constant' and fill_value is None:
            raise ValueError("使用'constant'策略时必须提供fill_value参数")

    # 计算均值
    def _calculate_mean(valid_values: list) -> float:
        if not valid_values:
            return np.nan
        return sum(valid_values) / len(valid_values)

    # 计算中位数
    def _calculate_median(valid_values: list) -> float:
        if not valid_values:
            return np.nan
        sorted_values = sorted(valid_values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            return sorted_values[n // 2]

    # 计算众数
    def _calculate_most_frequent(valid_values: list) -> Any:
        if not valid_values:
            return np.nan

        counter = Counter(valid_values)
        max_count = max(counter.values())

        # 找到所有具有最大频率的值
        most_common_values = [value for value, count in counter.items() if count == max_count]

        # 如果只有一个众数，直接返回
        if len(most_common_values) == 1:
            return most_common_values[0]

        # 如果有多个众数，根据数据类型选择策略
        if all(isinstance(v, (int, float, np.number)) for v in most_common_values):
            # 对于数值类型，选择最小值
            return min(most_common_values)
        else:
            # 对于非数值类型，选择字典序最小的
            return min(most_common_values, key=str)

    # 获取列中所有非NaN的有效值
    def _get_valid_values(series: pd.Series) -> list:
        return series.dropna().tolist()

    # 根据策略计算填充值
    def _calculate_fill_value(col_name: str, valid_values: list) -> Any:
        if not valid_values:
            if strategy == 'constant':
                return fill_value
            else:
                raise ValueError(f"列'{col_name}'中全部为缺失值，无法使用'{strategy}'策略进行填充")

        if strategy == 'mean':
            if not all(isinstance(v, (int, float, np.number)) for v in valid_values):
                raise TypeError(f"列'{col_name}'包含非数值型数据，无法使用'mean'策略")
            return _calculate_mean(valid_values)

        elif strategy == 'median':
            if not all(isinstance(v, (int, float, np.number)) for v in valid_values):
                raise TypeError(f"列'{col_name}'包含非数值型数据，无法使用'median'策略")
            return _calculate_median(valid_values)

        elif strategy == 'most_frequent':
            return _calculate_most_frequent(valid_values)

        elif strategy == 'constant':
            return fill_value

    # 填充缺失值
    def _fill_missing_values(data_copy: pd.DataFrame, impute_values: Dict[str, Any]) -> pd.DataFrame:
        for col_name in data_copy.columns:
            if col_name in impute_values:

                mask = data_copy[col_name].isna()
                data_copy.loc[mask, col_name] = impute_values[col_name]

        # 推断对象类型以优化存储
        return data_copy.infer_objects(copy=False)

    # 主函数
    try:
        # 验证输入参数
        _validate_inputs()

        # 设置随机种子（如果提供）
        if random_state is not None:
            np.random.seed(random_state)

        # 创建数据副本，避免修改原始数据
        data_copy = data.copy()

        # 初始化填充值字典
        impute_values = {}

        # 对每一列计算填充值
        for col_name in data_copy.columns:
            try:
                # 获取该列的有效值
                valid_values = _get_valid_values(data_copy[col_name])

                # 计算填充值
                fill_val = _calculate_fill_value(col_name, valid_values)
                impute_values[col_name] = fill_val

            except Exception as e:
                raise ValueError(f"处理列'{col_name}'时发生错误: {str(e)}")

        # 执行填充操作
        filled_data = _fill_missing_values(data_copy, impute_values)

        return filled_data

    except Exception as e:
        raise



import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict, Any


def minmax_scaler(data: pd.DataFrame,
                  columns: Union[str, int, List[Union[str, int]]],
                  feature_range: List[float] = [0, 1]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:

    # 验证输入参数的有效性
    def _validate_inputs(data, columns, feature_range):
        # 验证data是否为DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"输入数据必须是pandas DataFrame，当前类型: {type(data).__name__}")

        # 验证DataFrame是否为空
        if data.empty:
            raise ValueError("输入的DataFrame为空")

        # 验证feature_range
        if not isinstance(feature_range, (list, tuple)) or len(feature_range) != 2:
            raise ValueError("feature_range必须是包含两个元素的列表或元组")

        if not all(isinstance(x, (int, float)) for x in feature_range):
            raise ValueError("feature_range的元素必须是数值类型")

        if feature_range[0] >= feature_range[1]:
            raise ValueError(f"feature_range的最小值({feature_range[0]})必须小于最大值({feature_range[1]})")

        return True

    # 标准化列名参数，统一转换为列名列表
    def _normalize_columns(data, columns):
        if isinstance(columns, (str, int)):
            columns = [columns]
        elif not isinstance(columns, (list, tuple)):
            raise TypeError("columns参数必须是字符串、整数或它们的列表")

        # 验证并转换列索引/列名
        normalized_columns = []
        for col in columns:
            if isinstance(col, str):
                if col not in data.columns:
                    raise KeyError(f"列名 '{col}' 不存在于DataFrame中。可用列名: {list(data.columns)}")
                normalized_columns.append(col)
            elif isinstance(col, int):
                if col < 0 or col >= len(data.columns):
                    raise IndexError(f"列索引 {col} 超出范围。DataFrame共有 {len(data.columns)} 列")
                normalized_columns.append(data.columns[col])
            else:
                raise TypeError(f"列标识符必须是字符串或整数，当前类型: {type(col).__name__}")

        return normalized_columns

    # 验证列数据是否适合进行数值标准化
    def _validate_column_data(data, column):
        col_data = data[column]

        # 检查是否包含数值数据
        numeric_data = pd.to_numeric(col_data, errors='coerce')
        non_numeric_count = numeric_data.isna().sum() - col_data.isna().sum()

        if non_numeric_count > 0:
            raise ValueError(f"列 '{column}' 包含 {non_numeric_count} 个非数值数据，无法进行Min-Max标准化")

        # 检查是否全为缺失值
        valid_data = numeric_data.dropna()
        if len(valid_data) == 0:
            raise ValueError(f"列 '{column}' 的所有数值都是缺失值，无法计算最小值和最大值")

        # 检查数据范围是否为0（最大值等于最小值）
        min_val = valid_data.min()
        max_val = valid_data.max()
        if min_val == max_val:
            raise ValueError(f"列 '{column}' 的所有有效数值都相等({min_val})，数据范围为0，无法进行标准化")

        return numeric_data, min_val, max_val

    # 应用Min-Max标准化算法
    def _apply_minmax_scaling(values, min_val, max_val, feature_range):
        target_min, target_max = feature_range
        scale = max_val - min_val
        target_scale = target_max - target_min

        scaled_values = []
        for v in values:
            if pd.isna(v):  # 处理缺失值
                scaled_values.append(np.nan)
            else:
                # Min-Max归一化公式
                norm_v = (v - min_val) / scale
                # 映射到目标范围
                scaled_v = target_min + norm_v * target_scale
                scaled_values.append(scaled_v)

        return scaled_values

    # 主函数
    try:
        # 验证输入
        _validate_inputs(data, columns, feature_range)

        # 标准化列名参数
        target_columns = _normalize_columns(data, columns)

        # 创建结果DataFrame的副本
        result_df = data.copy()
        scaling_params = {}

        # 对每一列进行Min-Max标准化
        for column in target_columns:
            # 验证列数据
            numeric_data, min_val, max_val = _validate_column_data(result_df, column)

            # 应用Min-Max标准化
            scaled_values = _apply_minmax_scaling(numeric_data, min_val, max_val, feature_range)

            # 更新DataFrame
            result_df[column] = scaled_values

            # 保存标准化参数
            scaling_params[column] = {
                'min_val': min_val,
                'max_val': max_val,
                'scale': max_val - min_val,
                'feature_range': feature_range.copy()
            }

        return result_df, scaling_params

    except Exception as e:
        raise e



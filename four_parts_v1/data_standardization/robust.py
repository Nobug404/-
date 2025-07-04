import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict, Any


def robust_standardization(data: pd.DataFrame,
                           columns: Union[str, List[str], int, List[int]],
                           quantile_range: List[float] = [25, 75],
                           inplace: bool = False) -> Union[
    pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]]:

    # 验证输入参数的合法性
    def _validate_inputs(data, columns, quantile_range):
        # 验证data类型
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"输入data必须是pandas DataFrame类型，当前类型为: {type(data)}")

        if data.empty:
            raise ValueError("输入的DataFrame不能为空")

        # 验证quantile_range
        if not isinstance(quantile_range, (list, tuple)) or len(quantile_range) != 2:
            raise ValueError("quantile_range必须是包含两个元素的列表或元组")

        if not all(isinstance(q, (int, float)) for q in quantile_range):
            raise TypeError("quantile_range中的元素必须是数值类型")

        if not (0 <= quantile_range[0] < quantile_range[1] <= 100):
            raise ValueError("quantile_range必须满足: 0 <= q1 < q2 <= 100")

        # 验证并标准化columns参数
        if isinstance(columns, (str, int)):
            columns = [columns]
        elif not isinstance(columns, (list, tuple)):
            raise TypeError("columns参数必须是字符串、整数、或它们的列表")

        # 转换索引位置为列名
        standardized_columns = []
        for col in columns:
            if isinstance(col, int):
                if col < 0 or col >= len(data.columns):
                    raise ValueError(f"列索引 {col} 超出范围，DataFrame共有 {len(data.columns)} 列")
                standardized_columns.append(data.columns[col])
            elif isinstance(col, str):
                if col not in data.columns:
                    raise KeyError(f"列名 '{col}' 在DataFrame中不存在。可用列名: {list(data.columns)}")
                standardized_columns.append(col)
            else:
                raise TypeError(f"列标识符必须是字符串或整数，当前类型: {type(col)}")

        return standardized_columns

    # 计算robust标准化所需的统计量
    def _compute_robust_stats(values, quantile_range):
        # 过滤非缺失值
        clean_values = values.dropna()

        if len(clean_values) == 0:
            raise ValueError("列中没有有效的数值数据")

        # 检查是否为数值类型
        if not pd.api.types.is_numeric_dtype(clean_values):
            raise TypeError("只能对数值类型的列进行robust标准化")

        # 计算中位数
        median_val = clean_values.median()

        # 计算分位数
        q_low = np.percentile(clean_values, quantile_range[0])
        q_high = np.percentile(clean_values, quantile_range[1])
        iqr_val = q_high - q_low

        # 检查IQR是否为0
        if iqr_val == 0:
            raise ValueError(f"IQR为0，无法进行标准化。可能原因：数据方差过小或存在大量重复值")

        return median_val, iqr_val, q_low, q_high

    # 应用robust标准化变换
    def _apply_robust_scaling(values, median_val, iqr_val):
        # 创建float类型的副本以避免dtype不兼容警告
        scaled_values = pd.Series(values, dtype=float)

        # 使用向量化操作提高效率
        mask = ~values.isna()  # 非缺失值的掩码
        scaled_values[mask] = (values[mask].astype(float) - median_val) / iqr_val

        return scaled_values

    # 主函数
    try:
        # 验证输入参数
        target_columns = _validate_inputs(data, columns, quantile_range)

        # 创建结果DataFrame
        if inplace:
            result_df = data
        else:
            result_df = data.copy()

        # 存储变换参数
        transform_params = {}

        # 对每列进行robust标准化
        for col in target_columns:
            try:
                # 计算统计量
                median_val, iqr_val, q_low, q_high = _compute_robust_stats(
                    data[col], quantile_range
                )

                # 应用标准化
                result_df[col] = _apply_robust_scaling(
                    data[col], median_val, iqr_val
                )

                # 保存变换参数
                transform_params[col] = {
                    'median': median_val,
                    'iqr': iqr_val,
                    'q_low': q_low,
                    'q_high': q_high,
                    'quantile_range': quantile_range.copy()
                }

            except Exception as e:
                raise ValueError(f"处理列 '{col}' 时发生错误: {str(e)}")

        return result_df, transform_params

    except Exception as e:
        raise






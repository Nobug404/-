import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Any


def zscore_standardization(data: pd.DataFrame,
                           columns: Union[str, int, List[Union[str, int]]],
                           with_mean: bool = True,
                           with_std: bool = True) -> Tuple[pd.DataFrame, dict]:

    # 验证输入参数
    def _validate_inputs(data, columns, with_mean, with_std):
        # 验证data是否为DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("输入的data必须是pandas.DataFrame类型")

        # 验证DataFrame不为空
        if data.empty:
            raise ValueError("输入的DataFrame不能为空")

        # 验证with_mean和with_std参数
        if not isinstance(with_mean, bool):
            raise TypeError("with_mean参数必须是布尔类型")
        if not isinstance(with_std, bool):
            raise TypeError("with_std参数必须是布尔类型")

        # 至少有一个参数为True
        if not with_mean and not with_std:
            raise ValueError("with_mean和with_std不能同时为False")

        return True

    # 标准化列名
    def _normalize_column_names(data, columns):
        # 统一转换为列表格式
        if not isinstance(columns, list):
            columns = [columns]

        normalized_columns = []

        for col in columns:
            if isinstance(col, str):
                # 字符串列名
                if col not in data.columns:
                    raise KeyError(f"列名 '{col}' 在DataFrame中不存在")
                normalized_columns.append(col)

            elif isinstance(col, int):
                # 整数索引
                if col < 0 or col >= len(data.columns):
                    raise IndexError(f"列索引 {col} 超出范围，DataFrame共有 {len(data.columns)} 列")
                normalized_columns.append(data.columns[col])

            else:
                raise TypeError(f"列标识符必须是字符串或整数，得到: {type(col)}")

        return normalized_columns

    # 验证指定列是否为数值类型
    def _validate_numeric_columns(data, columns):
        for col in columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"列 '{col}' 不是数值类型，无法进行Z-score标准化")

            # 检查是否全为缺失值
            if data[col].isna().all():
                raise ValueError(f"列 '{col}' 全部为缺失值，无法计算统计量")

        return True

    # 计算均值和标准差
    def _calculate_statistics(values, with_mean, with_std):
        # 过滤缺失值
        valid_values = values.dropna()

        if len(valid_values) == 0:
            raise ValueError("没有有效的数值用于计算统计量")

        mean_val = None
        std_val = None

        if with_mean:
            mean_val = valid_values.mean()

        if with_std:
            std_val = valid_values.std(ddof=0)  # 使用总体标准差
            if std_val == 0:
                raise ValueError("标准差为0，无法进行标准化（所有有效值都相同）")

        return mean_val, std_val

    # 对单个Series进行标准化
    def _standardize_series(series, mean_val, std_val, with_mean, with_std):
        result = series.copy()

        # 对非缺失值进行标准化
        mask = series.notna()

        if with_mean and mean_val is not None:
            result.loc[mask] = result.loc[mask] - mean_val

        if with_std and std_val is not None:
            result.loc[mask] = result.loc[mask] / std_val

        return result

    # 主函数
    try:
        # 验证输入参数
        _validate_inputs(data, columns, with_mean, with_std)

        # 标准化列名
        target_columns = _normalize_column_names(data, columns)

        # 验证数值类型
        _validate_numeric_columns(data, target_columns)

        # 创建结果DataFrame副本
        result_data = data.copy()

        # 存储统计信息
        statistics_info = {}

        # 对每列进行标准化
        for col in target_columns:
            try:
                # 计算统计量
                mean_val, std_val = _calculate_statistics(
                    data[col], with_mean, with_std
                )

                # 执行标准化
                result_data[col] = _standardize_series(
                    data[col], mean_val, std_val, with_mean, with_std
                )

                # 保存统计信息
                statistics_info[col] = {
                    'mean': mean_val,
                    'std': std_val
                }

            except Exception as e:
                raise ValueError(f"处理列 '{col}' 时发生错误: {str(e)}")

        return result_data, statistics_info

    except Exception as e:
        raise e



import pandas as pd
import numpy as np
from typing import Union, Any, Optional
import warnings


def fill_missing_data(data: Union[pd.DataFrame, pd.Series, list, dict],
                      value: Any = None,
                      method: Optional[str] = None) -> pd.DataFrame:


    # 验证输入参数的有效性
    def _validate_inputs():
        # 验证method参数
        allowed_methods = [None, 'ffill', 'bfill']
        if method not in allowed_methods:
            raise ValueError(f"不支持的填充方法: {method}. 支持的方法: {allowed_methods}")

        # 验证固定值填充时value不能为None
        if method is None and value is None:
            raise ValueError("使用固定值填充时，必须提供value参数")

        # 验证数据类型
        if data is None:
            raise ValueError("输入数据不能为None")

        # 检查数据是否为空
        if hasattr(data, '__len__') and len(data) == 0:
            raise ValueError("输入数据不能为空")

    # 将输入数据转换为DataFrame格式
    def _convert_to_dataframe():
        try:
            if isinstance(data, pd.DataFrame):
                return data.copy()
            elif isinstance(data, pd.Series):
                return data.to_frame()
            elif isinstance(data, (list, dict)):
                return pd.DataFrame(data)
            else:
                # 尝试直接转换
                return pd.DataFrame(data)
        except Exception as e:
            raise TypeError(f"无法将输入数据转换为DataFrame: {str(e)}")

    # 固定值填充：用指定值替换所有NaN，处理数据类型兼容性
    def _fixed_value_fill(df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()

        # 遍历每个单元格
        for col in result_df.columns:
            for idx in result_df.index:
                if pd.isna(result_df.loc[idx, col]):
                    # 获取当前列的数据类型
                    col_dtype = result_df[col].dtype

                    # 如果是数值型列且填充值不兼容，尝试类型转换
                    if pd.api.types.is_numeric_dtype(col_dtype):
                        try:
                            # 尝试转换为数值类型
                            if isinstance(value, str):
                                # 如果是字符串，先尝试转换为数值
                                try:
                                    numeric_value = pd.to_numeric(value)
                                    result_df.loc[idx, col] = numeric_value
                                except (ValueError, TypeError):
                                    # 如果无法转换，将整列转换为object类型
                                    result_df[col] = result_df[col].astype('object')
                                    result_df.loc[idx, col] = value
                            else:
                                result_df.loc[idx, col] = value
                        except Exception:
                            # 如果类型转换失败，将整列转换为object类型
                            result_df[col] = result_df[col].astype('object')
                            result_df.loc[idx, col] = value
                    else:
                        # 非数值型列直接赋值
                        result_df.loc[idx, col] = value

        return result_df

    # 前向填充：用前面的有效值填充NaN
    def _forward_fill(df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()

        # 对每列进行遍历
        for col in result_df.columns:
            last_valid = None

            # 逐行遍历该列
            for idx in result_df.index:
                current_value = result_df.loc[idx, col]

                if not pd.isna(current_value):
                    # 当前值不是NaN，更新last_valid
                    last_valid = current_value
                elif pd.isna(current_value) and last_valid is not None:
                    # 当前值是NaN且有有效的前值，进行填充
                    result_df.loc[idx, col] = last_valid

        return result_df

    # 后向填充：用后面的有效值填充NaN
    def _backward_fill(df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()

        # 对每列进行遍历
        for col in result_df.columns:
            # 首先收集该列的所有有效值及其位置
            valid_values = {}
            for idx in result_df.index:
                if not pd.isna(result_df.loc[idx, col]):
                    valid_values[idx] = result_df.loc[idx, col]

            # 自底向上逐行遍历该列
            next_valid = None
            for idx in reversed(result_df.index):
                current_value = result_df.loc[idx, col]

                if not pd.isna(current_value):
                    # 当前值不是NaN，更新next_valid
                    next_valid = current_value
                elif pd.isna(current_value) and next_valid is not None:
                    # 当前值是NaN且有有效的后值，进行填充
                    result_df.loc[idx, col] = next_valid

        return result_df

    # 检查并警告剩余的缺失值
    def _check_remaining_nulls(df: pd.DataFrame) -> None:
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            print(f"警告: 填充后仍有 {null_count} 个缺失值未被填充")

    # 主函数
    try:
        # 验证输入参数
        _validate_inputs()

        # 转换数据格式
        df = _convert_to_dataframe()

        # 检查是否有缺失值需要处理
        if not df.isnull().any().any():
            print("数据中没有缺失值，返回原数据")
            return df

        # 执行相应的填充策略
        if method is None:
            # 固定值填充
            result = _fixed_value_fill(df)
        elif method == 'ffill':
            # 前向填充
            result = _forward_fill(df)
        elif method == 'bfill':
            # 后向填充
            result = _backward_fill(df)

        # 检查填充结果
        _check_remaining_nulls(result)

        return result

    except Exception as e:
        raise



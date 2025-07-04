import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Union, List, Tuple, Dict, Any


def dataframe_one_hot_encode(data: pd.DataFrame,
                             columns: Union[str, int, List[Union[str, int]]],
                             drop_first: bool = False,
                             handle_unknown: str = 'error',
                             sparse: bool = False,
                             return_dataframe: bool = True,
                             prefix: Union[str, Dict[str, str], None] = None,
                             prefix_sep: str = '_') -> Union[Tuple[pd.DataFrame, Dict], Tuple[np.ndarray, Dict]]:
    # 验证DataFrame和列参数的合法性
    def _validate_dataframe_input(df, cols):
        # 验证DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"参数 'data' 必须是 pandas.DataFrame 类型，当前类型: {type(df)}")

        if df.empty:
            raise ValueError("输入的DataFrame不能为空")

        if df.shape[0] == 0:
            raise ValueError("输入的DataFrame必须包含至少一行数据")

        # 验证参数
        if isinstance(cols, (str, int)):
            cols = [cols]
        elif isinstance(cols, (list, tuple)):
            if len(cols) == 0:
                raise ValueError("参数 'columns' 不能为空列表")
        else:
            raise TypeError(f"参数 'columns' 必须是 str, int 或它们的列表，当前类型: {type(cols)}")

        return cols

    # 将列索引转换为列名，并验证列的存在性
    def _resolve_column_names(df, cols):
        resolved_columns = []

        for col in cols:
            # 列名方式索引
            if isinstance(col, str):
                if col not in df.columns:
                    raise KeyError(f"列名 '{col}' 在DataFrame中不存在。可用列名: {list(df.columns)}")
                resolved_columns.append(col)
            # 位置索引方式
            elif isinstance(col, int):
                if col < 0 or col >= len(df.columns):
                    raise IndexError(
                        f"列索引 {col} 超出范围。DataFrame共有 {len(df.columns)} 列，有效索引范围: 0-{len(df.columns) - 1}")
                resolved_columns.append(df.columns[col])
            else:
                raise TypeError(f"列标识符必须是字符串或整数，当前类型: {type(col)}")

        return resolved_columns

    # 验证其他参数的合法性
    def _validate_other_params(drop_f, handle_unk, sp, ret_df, pref, pref_sep):
        if not isinstance(drop_f, bool):
            raise TypeError(f"参数 'drop_first' 必须是 bool 类型，当前类型: {type(drop_f)}")

        valid_handle = ['error', 'ignore', 'zero']
        if handle_unk not in valid_handle:
            raise ValueError(f"参数 'handle_unknown' 必须是 {valid_handle} 中的一个，当前值: '{handle_unk}'")

        if not isinstance(sp, bool):
            raise TypeError(f"参数 'sparse' 必须是 bool 类型，当前类型: {type(sp)}")

        if not isinstance(ret_df, bool):
            raise TypeError(f"参数 'return_dataframe' 必须是 bool 类型，当前类型: {type(ret_df)}")

        if pref is not None and not isinstance(pref, (str, dict)):
            raise TypeError(f"参数 'prefix' 必须是 str, dict 或 None，当前类型: {type(pref)}")

        if not isinstance(pref_sep, str):
            raise TypeError(f"参数 'prefix_sep' 必须是 str 类型，当前类型: {type(pref_sep)}")

    # 获取指定列的前缀
    def _get_column_prefix(col_name, pref, pref_sep):
        if pref is None:
            return col_name + pref_sep
        elif isinstance(pref, str):
            return pref + pref_sep
        elif isinstance(pref, dict):
            return pref.get(col_name, col_name) + pref_sep
        else:
            return col_name + pref_sep

    # 对单个列进行独热编码 - 修复版本
    def _encode_single_column(series, col_name, drop_f, handle_unk):
        # 检查列中是否有 None 或 NaN 值
        if series.isnull().any():
            null_indices = series.isnull()
            raise ValueError(f"列 '{col_name}' 中包含 NaN/None 值，位置索引: {list(null_indices[null_indices].index)}")

        # 获取唯一类别并排序
        try:
            unique_categories = sorted(series.unique())
        except TypeError as e:
            raise ValueError(f"列 '{col_name}' 中包含不可比较的类型，无法排序: {e}")

        original_categories = unique_categories.copy()

        # 确定要编码的类别（用于生成列）
        encoded_categories = unique_categories.copy()
        dropped_category = None

        # 启用drop_first且类别数大于1，则移除第一个类别
        if drop_f and len(unique_categories) > 1:
            # 记录被丢弃的类别
            dropped_category = unique_categories[0]
            # 用于生成编码列的类别
            encoded_categories = unique_categories[1:]
        elif drop_f and len(unique_categories) == 1:
            raise ValueError(
                f"对列 '{col_name}' 启用 'drop_first=True' 时，至少需要2个不同的类别，当前只有1个类别: '{unique_categories[0]}'")

        if len(encoded_categories) == 0:
            raise ValueError(f"列 '{col_name}' 经过处理后没有可用的类别进行编码")

        # 构建编码类别到列索引的映射（不包括dropped_category）
        category_index = {cat: idx for idx, cat in enumerate(encoded_categories)}

        # 构建编码矩阵
        n_samples = len(series)
        n_categories = len(encoded_categories)

        encoding_matrix = np.zeros((n_samples, n_categories), dtype=np.int8)
        unknown_indices = []

        for i, category in enumerate(series):
            # 正常编码
            if category in category_index:
                col_idx = category_index[category]
                encoding_matrix[i, col_idx] = 1
            # 被丢弃的类别编码为全0向量（不做任何操作，保持默认的0值）
            elif drop_f and category == dropped_category:
                pass

            else:
                # 处理真正的未知类别
                if handle_unk == 'error':
                    raise ValueError(f"在列 '{col_name}' 中遇到训练时未见的类别: '{category}' (行索引: {i})")
                elif handle_unk == 'ignore':
                    unknown_indices.append(i)
                elif handle_unk == 'zero':
                    # 该行保持全为0，不做处理
                    pass

        return encoding_matrix, category_index, original_categories, unknown_indices

    # 构建编码后的DataFrame
    def _build_encoded_dataframe(df, col_encodings, encoding_info, sp):
        if sp:
            raise NotImplementedError("稀疏矩阵模式下暂不支持返回DataFrame格式，请设置 return_dataframe=False")

        # 获取非编码列
        encoded_columns = list(encoding_info.keys())
        other_columns = [col for col in df.columns if col not in encoded_columns]

        # 创建新的DataFrame
        result_df = df[other_columns].copy() if other_columns else pd.DataFrame(index=df.index)

        # 添加编码列
        for col_name, (encoding_matrix, category_index, original_categories, unknown_indices) in col_encodings.items():
            encoded_col_names = encoding_info[col_name]['encoded_columns']

            encoded_df = pd.DataFrame(encoding_matrix, columns=encoded_col_names, index=df.index)
            result_df = pd.concat([result_df, encoded_df], axis=1)

        return result_df

    # 构建完整的编码矩阵
    def _build_encoding_matrix(col_encodings, sp):
        all_matrices = [matrix for matrix, _, _, _ in col_encodings.values()]

        if len(all_matrices) == 0:
            raise ValueError("没有可用的编码矩阵")

        combined_matrix = np.hstack(all_matrices)

        if sp:
            combined_matrix = csr_matrix(combined_matrix)

        return combined_matrix

    # 主函数
    try:
        # 验证输入参数
        columns_list = _validate_dataframe_input(data, columns)
        resolved_columns = _resolve_column_names(data, columns_list)
        _validate_other_params(drop_first, handle_unknown, sparse, return_dataframe, prefix, prefix_sep)

        # 对每列进行独热编码
        column_encodings = {}
        encoding_info = {}
        global_col_index = 0

        for col_name in resolved_columns:
            series = data[col_name]

            # 对单列进行编码
            encoding_matrix, category_index, original_categories, unknown_indices = _encode_single_column(
                series, col_name, drop_first, handle_unknown
            )

            # 生成编码后的列名
            col_prefix = _get_column_prefix(col_name, prefix, prefix_sep)
            encoded_col_names = [f"{col_prefix}{cat}" for cat in category_index.keys()]

            # 更新全局列索引映射
            global_category_index = {}
            for local_idx, (cat, _) in enumerate(category_index.items()):
                global_category_index[cat] = global_col_index + local_idx

            # 存储编码结果
            column_encodings[col_name] = (encoding_matrix, category_index, original_categories, unknown_indices)

            # 存储编码信息
            encoding_info[col_name] = {
                'category_index': global_category_index,
                'local_category_index': category_index,
                'original_categories': original_categories,
                'encoded_columns': encoded_col_names,
                'summary': {
                    'n_samples': len(series),
                    'n_original_categories': len(original_categories),
                    'n_encoded_categories': len(category_index),
                    'dropped_first': drop_first,
                    'n_unknown_handled': len(unknown_indices),
                    'unknown_indices': unknown_indices
                }
            }

            global_col_index += len(category_index)

        # 构建返回结果
        if return_dataframe:
            result = _build_encoded_dataframe(data, column_encodings, encoding_info, sparse)
        else:
            result = _build_encoding_matrix(column_encodings, sparse)

        return result, encoding_info

    except Exception as e:
        # 重新抛出异常，保持原始异常类型和信息
        raise e



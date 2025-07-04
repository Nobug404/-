import pandas as pd
import numpy as np
from typing import Union, List, Optional, Tuple, Any


def feature_binning(data: pd.DataFrame,
                    columns: Union[str, List[str], int, List[int]],
                    bins: Union[int, List[float]],
                    method: str = "equal_width",
                    labels: Optional[List[str]] = None,
                    right_closed: bool = True,
                    handle_na: str = "ignore",
                    precision: int = 2) -> pd.DataFrame:
    #输入参数验证
    def _validate_inputs():
        # 验证数据类型
        if not isinstance(data, pd.DataFrame):
            raise TypeError("输入数据必须是pandas DataFrame类型")

        if data.empty:
            raise ValueError("输入数据不能为空")

        # 验证分箱方法
        if method not in ["equal_width", "equal_freq"]:
            raise ValueError(f"分箱方法必须是'equal_width'或'equal_freq'，当前输入: {method}")

        # 验证缺失值处理方式
        if handle_na not in ["ignore", "separate"]:
            raise ValueError(f"缺失值处理方式必须是'ignore'或'separate'，当前输入: {handle_na}")

        # 验证精度
        if not isinstance(precision, int) or precision < 0:
            raise ValueError("精度必须是非负整数")

        # 验证bins参数
        if isinstance(bins, int):
            if bins <= 0:
                raise ValueError("分箱数量必须是正整数")
        elif isinstance(bins, list):
            if len(bins) < 2:
                raise ValueError("分箱边界列表至少需要2个元素")
            if not all(isinstance(x, (int, float)) for x in bins):
                raise ValueError("分箱边界必须是数值类型")
        else:
            raise TypeError("bins参数必须是整数或数值列表")

    # 获取列名列表
    def _get_column_names(columns):
        if isinstance(columns, str):
            if columns not in data.columns:
                raise ValueError(f"列名'{columns}'不存在于数据中")
            return [columns]
        elif isinstance(columns, int):
            if columns < 0 or columns >= len(data.columns):
                raise IndexError(f"列索引{columns}超出范围，数据共有{len(data.columns)}列")
            return [data.columns[columns]]
        elif isinstance(columns, list):
            result_columns = []
            for col in columns:
                if isinstance(col, str):
                    if col not in data.columns:
                        raise ValueError(f"列名'{col}'不存在于数据中")
                    result_columns.append(col)
                elif isinstance(col, int):
                    if col < 0 or col >= len(data.columns):
                        raise IndexError(f"列索引{col}超出范围，数据共有{len(data.columns)}列")
                    result_columns.append(data.columns[col])
                else:
                    raise TypeError(f"列标识符必须是字符串或整数，当前类型: {type(col)}")
            return result_columns
        else:
            raise TypeError("columns参数必须是字符串、整数或它们的列表")

    # 验证列是否为数值型
    def _validate_numeric_column(col_name, values):
        # 检查非缺失值是否都是数值型
        non_na_values = values.dropna()
        if len(non_na_values) == 0:
            raise ValueError(f"列'{col_name}'全部为缺失值，无法进行分箱")

        if not pd.api.types.is_numeric_dtype(non_na_values):
            raise TypeError(f"列'{col_name}'必须是数值型数据，当前类型: {values.dtype}")

    #计算分箱边界
    def _calculate_bin_edges(values, bins, method):
        non_na_values = values.dropna()

        if isinstance(bins, int):
            if method == "equal_width":
                min_val = float(non_na_values.min())
                max_val = float(non_na_values.max())
                if min_val == max_val:
                    raise ValueError("数据中所有非缺失值相同，无法进行等宽分箱")
                step = (max_val - min_val) / bins
                edges = [min_val + i * step for i in range(bins + 1)]
                edges[-1] = max_val

            elif method == "equal_freq":
                if len(non_na_values) < bins:
                    raise ValueError(f"非缺失值数量({len(non_na_values)})少于分箱数量({bins})")
                sorted_values = sorted(non_na_values)
                edges = []
                for i in range(bins + 1):
                    if i == 0:
                        edges.append(sorted_values[0])
                    elif i == bins:
                        edges.append(sorted_values[-1])
                    else:
                        idx = int(i * len(sorted_values) / bins)
                        edges.append(sorted_values[idx])

                # 去重并保持顺序
                unique_edges = []
                for edge in edges:
                    if not unique_edges or edge != unique_edges[-1]:
                        unique_edges.append(edge)

                if len(unique_edges) < 2:
                    raise ValueError("等频分箱后边界数量不足，可能是数据重复值过多")
                edges = unique_edges
        else:
            # 排序并去重，bins是边界列表
            edges = sorted(set(bins))
            if len(edges) != len(bins):
                print(f"警告: 分箱边界存在重复值，已自动去重。原始边界数: {len(bins)}, 去重后: {len(edges)}")

        return edges

    # 生成分箱标签
    def _generate_labels(bin_edges, labels, right_closed, precision):
        if labels is not None:
            if len(labels) != len(bin_edges) - 1:
                raise ValueError(f"标签数量({len(labels)})与分箱数量({len(bin_edges) - 1})不匹配")
            return labels


        generated_labels = []
        for i in range(len(bin_edges) - 1):
            left = round(bin_edges[i], precision)
            right = round(bin_edges[i + 1], precision)
            if right_closed:
                if i == 0:
                    label = f"[{left}, {right}]"
                else:
                    label = f"({left}, {right}]"
            else:
                if i == len(bin_edges) - 2:
                    label = f"[{left}, {right}]"
                else:
                    label = f"[{left}, {right})"
            generated_labels.append(label)

        return generated_labels

    # 分配箱字名称
    def _assign_bins(values, bin_edges, right_closed, handle_na):
        bin_indices = []
        na_bin_index = len(bin_edges) - 1 if handle_na == "separate" else None

        for value in values:
            if pd.isna(value):
                if handle_na == "separate":
                    bin_indices.append(na_bin_index)
                else:
                    bin_indices.append(None)
            else:
                # 查找合适的箱
                assigned = False
                for i in range(len(bin_edges) - 1):
                    if right_closed:
                        if i == 0:
                            # 第一个区间左闭右闭
                            if bin_edges[i] <= value <= bin_edges[i + 1]:
                                bin_indices.append(i)
                                assigned = True
                                break
                        else:
                            # 其他区间左开右闭
                            if bin_edges[i] < value <= bin_edges[i + 1]:
                                bin_indices.append(i)
                                assigned = True
                                break
                    else:
                        if i == len(bin_edges) - 2:
                            # 最后一个区间左闭右闭
                            if bin_edges[i] <= value <= bin_edges[i + 1]:
                                bin_indices.append(i)
                                assigned = True
                                break
                        else:
                            # 其他区间左闭右开
                            if bin_edges[i] <= value < bin_edges[i + 1]:
                                bin_indices.append(i)
                                assigned = True
                                break

                if not assigned:
                    # 超出边界的值
                    if value < bin_edges[0]:
                        # 分配到第一个箱
                        bin_indices.append(0)
                    else:
                        # 分配到最后一个箱
                        bin_indices.append(len(bin_edges) - 2)

        return bin_indices

    # 处理单个列的分箱
    def _process_single_column(col_name, values):
        # 验证数值型
        _validate_numeric_column(col_name, values)

        # 计算分箱边界
        bin_edges = _calculate_bin_edges(values, bins, method)

        # 生成标签
        bin_labels = _generate_labels(bin_edges, labels, right_closed, precision)

        # 分配箱字名称
        bin_indices = _assign_bins(values, bin_edges, right_closed, handle_na)

        # 创建分箱结果
        binned_values = []
        for idx in bin_indices:
            if idx is None:
                binned_values.append(None)
            elif handle_na == "separate" and idx == len(bin_edges) - 1:
                binned_values.append("Missing")
            else:
                binned_values.append(bin_labels[idx])

        return binned_values, bin_edges, bin_labels

    # 主函数
    try:
        # 输入验证
        _validate_inputs()

        # 获取要处理的列名
        target_columns = _get_column_names(columns)

        # 复制数据避免修改原数据
        result_data = data.copy()

        # 处理每一列
        for col_name in target_columns:
            binned_values, bin_edges, bin_labels = _process_single_column(col_name, data[col_name])

            # 添加分箱结果列
            result_data[f"{col_name}_binned"] = binned_values

        return result_data

    except Exception as e:
        raise RuntimeError(f"分箱处理过程中发生错误: {str(e)}")



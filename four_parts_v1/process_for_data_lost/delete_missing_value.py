import math


def delete_missing_values(data_table, axis=0, how='any', subset=None):

    # 验证输入参数的有效性
    def _validate_parameters(data_table, axis, how, subset):
        # 验证数据格式
        if not isinstance(data_table, list):
            raise TypeError("数据必须是列表格式")

        if len(data_table) == 0:
            raise ValueError("输入数据不能为空")

        # 检查数据是否为二维结构
        if not all(isinstance(row, (list, tuple)) for row in data_table):
            raise TypeError("数据必须是二维结构(列表的列表)")

        # 检查所有行长度是否一致
        if len(data_table) > 0:
            row_length = len(data_table[0])
            if not all(len(row) == row_length for row in data_table):
                raise ValueError("所有行的列数必须一致")

        # 验证axis参数
        if axis not in [0, 1]:
            raise ValueError("axis参数必须是0(按行删除)或1(按列删除)")

        # 验证how参数
        if how not in ['any', 'all']:
            raise ValueError("how参数必须是'any'或'all'")

        # 验证subset参数
        if subset is not None:
            if not isinstance(subset, (list, tuple)):
                raise TypeError("subset参数必须是列表或元组")

            if len(data_table) > 0:
                if axis == 0:  # 按行删除时，subset指定要检查的列
                    max_col_index = len(data_table[0]) - 1
                    for col_idx in subset:
                        if not isinstance(col_idx, int):
                            raise TypeError("subset中的列索引必须是整数")
                        if col_idx < 0 or col_idx > max_col_index:
                            raise IndexError(f"列索引 {col_idx} 超出范围 [0, {max_col_index}]")
                else:  # 按列删除时，subset指定要检查的行
                    max_row_index = len(data_table) - 1
                    for row_idx in subset:
                        if not isinstance(row_idx, int):
                            raise TypeError("subset中的行索引必须是整数")
                        if row_idx < 0 or row_idx > max_row_index:
                            raise IndexError(f"行索引 {row_idx} 超出范围 [0, {max_row_index}]")

    # 判断一个值是否为缺失值
    def _is_missing_value(value):
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        if isinstance(value, str) and value.strip() == '':
            return True
        # 可以根据需要添加其他缺失值判断条件
        return False

    # 统计行中缺失值的数量
    def _count_missing_in_row(row, check_columns):
        if check_columns is None:
            # 检查整行
            missing_count = sum(1 for value in row if _is_missing_value(value))
            total_count = len(row)
        else:
            # 只检查指定列
            missing_count = sum(1 for col_idx in check_columns if _is_missing_value(row[col_idx]))
            total_count = len(check_columns)

        return missing_count, total_count

    # 统计列中缺失值的数量
    def _count_missing_in_column(data_table, col_idx, check_rows):
        if check_rows is None:
            # 检查整列
            missing_count = sum(1 for row in data_table if _is_missing_value(row[col_idx]))
            total_count = len(data_table)
        else:
            # 只检查指定行
            missing_count = sum(1 for row_idx in check_rows if _is_missing_value(data_table[row_idx][col_idx]))
            total_count = len(check_rows)

        return missing_count, total_count

    # 判断是否应该删除某一行
    def _should_remove_row(row, check_columns, how):
        missing_count, total_count = _count_missing_in_row(row, check_columns)

        if how == 'any':
            return missing_count > 0
        elif how == 'all':
            return missing_count == total_count

    # 判断是否应该删除某一列
    def _should_remove_column(data_table, col_idx, check_rows, how):
        missing_count, total_count = _count_missing_in_column(data_table, col_idx, check_rows)

        if how == 'any':
            return missing_count > 0
        elif how == 'all':
            return missing_count == total_count

    # 找出需要删除的行索引
    def _find_rows_to_remove(data_table, subset, how):
        rows_to_remove = []

        for row_idx, row in enumerate(data_table):
            if _should_remove_row(row, subset, how):
                rows_to_remove.append(row_idx)

        return rows_to_remove

    # 找出需要删除的列索引
    def _find_columns_to_remove(data_table, subset, how):
        if len(data_table) == 0:
            return []

        columns_to_remove = []
        num_columns = len(data_table[0])

        for col_idx in range(num_columns):
            if _should_remove_column(data_table, col_idx, subset, how):
                columns_to_remove.append(col_idx)

        return columns_to_remove

    # 删除指定的行，返回新的数据表
    def _remove_rows(data_table, rows_to_remove):
        if not rows_to_remove:
            return [row[:] for row in data_table]  # 返回深拷贝

        rows_to_remove_set = set(rows_to_remove)
        return [row[:] for row_idx, row in enumerate(data_table) if row_idx not in rows_to_remove_set]

    # 删除指定的列，返回新的数据表
    def _remove_columns(data_table, columns_to_remove):
        if not columns_to_remove:
            return [row[:] for row in data_table]  # 返回深拷贝

        columns_to_remove_set = set(columns_to_remove)
        return [[value for col_idx, value in enumerate(row) if col_idx not in columns_to_remove_set]
                for row in data_table]

    # 主函数
    try:
        # 验证参数
        _validate_parameters(data_table, axis, how, subset)

        # 处理空数据情况
        if len(data_table) == 0:
            return []

        # 根据axis执行不同的删除逻辑
        if axis == 0:
            # 按行删除
            rows_to_remove = _find_rows_to_remove(data_table, subset, how)
            result_data = _remove_rows(data_table, rows_to_remove)
        else:
            # 按列删除
            columns_to_remove = _find_columns_to_remove(data_table, subset, how)
            result_data = _remove_columns(data_table, columns_to_remove)

        return result_data

    except Exception as e:
        raise e



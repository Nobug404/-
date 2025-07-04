
"""
删除包含缺失值的行或列

参数:
data_table: 二维列表，输入数据
axis: int, 0表示按行删除，1表示按列删除
how: str, 'any'表示有任何缺失值就删除，'all'表示全部为缺失值才删除
subset: list, 指定要检查的列索引(axis=0时)或行索引(axis=1时)

返回:
删除缺失值后的新数据表
"""


def print_table(data, title="数据表"):
    """辅助函数：格式化打印数据表"""
    print(f"\n{title}:")
    if not data:
        print("  (空数据)")
        return

    for i, row in enumerate(data):
        print(f"  {i}: {row}")


def run_comprehensive_tests():
    """运行全面的测试集"""
    print("=" * 60)
    print("删除缺失数据函数 - 全面测试")
    print("=" * 60)

    # 测试数据集1：基本功能测试
    print("\n【测试集1：基本功能测试】")
    test_data1 = [
        ['Alice', 25, 'Engineer', 50000],
        ['Bob', None, 'Designer', 45000],  # None缺失
        [None, 30, 'Manager', None],  # 多个None
        ['Charlie', 35, '', 60000],  # 空字符串
        ['David', float('nan'), 'Developer', 55000],  # NaN缺失
        [None, None, None, None],  # 全部缺失
        ['Eva', 28, 'Analyst', 48000]  # 完整数据
    ]

    print_table(test_data1, "原始数据")

    # 测试1.1: 删除包含任意缺失值的行
    print("\n测试1.1: 删除包含任意缺失值的行 (axis=0, how='any')")
    try:
        result = delete_missing_values(test_data1, axis=0, how='any')
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    # 测试1.2: 删除全部缺失的行
    print("\n测试1.2: 删除全部缺失的行 (axis=0, how='all')")
    try:
        result = delete_missing_values(test_data1, axis=0, how='all')
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    # 测试1.3: 删除包含任意缺失值的列
    print("\n测试1.3: 删除包含任意缺失值的列 (axis=1, how='any')")
    try:
        result = delete_missing_values(test_data1, axis=1, how='any')
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    # 测试1.4: 删除全部缺失的列
    print("\n测试1.4: 删除全部缺失的列 (axis=1, how='all')")
    try:
        result = delete_missing_values(test_data1, axis=1, how='all')
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    # 测试数据集2：subset功能测试
    print("\n\n【测试集2：subset功能测试】")
    test_data2 = [
        ['Name', 'Age', 'Job', 'Salary'],
        ['Alice', 25, None, 50000],
        ['Bob', None, 'Designer', 45000],
        ['Charlie', 35, 'Manager', None],
        ['David', 40, 'Developer', 55000]
    ]

    print_table(test_data2, "原始数据")

    # 测试2.1: 只检查指定列的缺失值
    print("\n测试2.1: 只检查第1列(Age)的缺失值 (subset=[1])")
    try:
        result = delete_missing_values(test_data2, axis=0, how='any', subset=[1])
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    # 测试2.2: 检查多个列的缺失值
    print("\n测试2.2: 检查第1列和第2列的缺失值 (subset=[1, 2])")
    try:
        result = delete_missing_values(test_data2, axis=0, how='any', subset=[1, 2])
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    # 测试2.3: 列删除时的subset测试
    print("\n测试2.3: 列删除时只检查指定行 (axis=1, subset=[1, 2])")
    try:
        result = delete_missing_values(test_data2, axis=1, how='any', subset=[1, 2])
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    # 测试数据集3：边界情况测试
    print("\n\n【测试集3：边界情况测试】")

    # 测试3.1: 单行数据
    print("\n测试3.1: 单行数据")
    single_row = [['Alice', None, 'Engineer']]
    print_table(single_row, "原始数据")
    try:
        result = delete_missing_values(single_row, axis=0, how='any')
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    # 测试3.2: 单列数据
    print("\n测试3.2: 单列数据")
    single_col = [['Alice'], [None], ['Bob']]
    print_table(single_col, "原始数据")
    try:
        result = delete_missing_values(single_col, axis=1, how='any')
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    # 测试3.3: 全部完整数据
    print("\n测试3.3: 全部完整数据")
    complete_data = [
        ['Alice', 25, 'Engineer'],
        ['Bob', 30, 'Designer'],
        ['Charlie', 35, 'Manager']
    ]
    print_table(complete_data, "原始数据")
    try:
        result = delete_missing_values(complete_data, axis=0, how='any')
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    # 测试3.4: 全部缺失数据
    print("\n测试3.4: 全部缺失数据")
    all_missing = [
        [None, '', float('nan')],
        ['', None, float('nan')],
        [float('nan'), None, '']
    ]
    print_table(all_missing, "原始数据")
    try:
        result = delete_missing_values(all_missing, axis=0, how='any')
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    # 测试数据集4：异常情况测试
    print("\n\n【测试集4：异常情况测试】")

    # 测试4.1: 无效的axis参数
    print("\n测试4.1: 无效的axis参数")
    try:
        result = delete_missing_values(test_data1, axis=2)
    except Exception as e:
        print(f"成功捕获异常: {e}")

    # 测试4.2: 无效的how参数
    print("\n测试4.2: 无效的how参数")
    try:
        result = delete_missing_values(test_data1, how='invalid')
    except Exception as e:
        print(f"成功捕获异常: {e}")

    # 测试4.3: subset索引超范围
    print("\n测试4.3: subset索引超范围")
    try:
        result = delete_missing_values(test_data1, subset=[10])
    except Exception as e:
        print(f"成功捕获异常: {e}")

    # 测试4.4: 非列表输入
    print("\n测试4.4: 非列表输入")
    try:
        result = delete_missing_values("not a list")
    except Exception as e:
        print(f"成功捕获异常: {e}")

    # 测试4.5: 不一致的行长度
    print("\n测试4.5: 不一致的行长度")
    inconsistent_data = [
        ['Alice', 25, 'Engineer'],
        ['Bob', 30],  # 缺少一列
        ['Charlie', 35, 'Manager', 60000]  # 多一列
    ]
    try:
        result = delete_missing_values(inconsistent_data)
    except Exception as e:
        print(f"成功捕获异常: {e}")

    # 测试数据集5：特殊缺失值类型测试
    print("\n\n【测试集5：特殊缺失值类型测试】")
    special_missing = [
        ['Alice', 25, 'Engineer', 50000],
        ['Bob', None, 'Designer', 45000],  # None
        ['Charlie', float('nan'), 'Manager', 60000],  # NaN
        ['David', 40, '', 55000],  # 空字符串
        ['Eva', 35, '   ', 48000],  # 空白字符串
        ['Frank', 30, 'Developer', 52000]  # 完整数据
    ]

    print_table(special_missing, "原始数据（包含各种缺失值类型）")

    print("\n测试5.1: 删除包含各种缺失值类型的行")
    try:
        result = delete_missing_values(special_missing, axis=0, how='any')
        print_table(result, "结果")
    except Exception as e:
        print(f"错误: {e}")

    print("\n测试完成！")
    print("=" * 60)


# 运行测试
if __name__ == "__main__":
    run_comprehensive_tests()
#------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
"""
填充缺失数据的函数

参数:
data: 输入数据，支持 DataFrame, Series, list, dict
value: 固定值填充时使用的值
method: 填充方法，支持 None(固定值), 'ffill'(前向填充), 'bfill'(后向填充)

返回:
pd.DataFrame: 填充后的数据
"""

# 使用示例和测试函数
def comprehensive_test_suite():
    """全面测试填充缺失值函数的各种情况"""

    def print_test_header(test_name):
        print(f"\n{'=' * 60}")
        print(f"测试: {test_name}")
        print(f"{'=' * 60}")

    def print_test_result(test_name, data, result, expected_behavior=""):
        print(f"\n{test_name}:")
        print(f"原始数据:")
        print(data)
        print(f"填充结果:")
        print(result)
        if expected_behavior:
            print(f"预期行为: {expected_behavior}")
        print(f"{'-' * 40}")

    # 测试1: 数据类型兼容性问题修复测试
    print_test_header("1. 数据类型兼容性测试")

    # 测试数值列用字符串填充
    numeric_data = pd.DataFrame({
        'integers': [1, np.nan, 3, np.nan, 5],
        'floats': [1.1, np.nan, 3.3, np.nan, 5.5]
    })

    result1 = fill_missing_data(numeric_data, value="MISSING")
    print_test_result("数值列用字符串填充", numeric_data, result1, "应该将列转换为object类型")

    # 测试数值列用数值填充
    result2 = fill_missing_data(numeric_data, value=0)
    print_test_result("数值列用数值填充", numeric_data, result2, "应该保持原数据类型")

    # 测试数值列用可转换的字符串填充
    result3 = fill_missing_data(numeric_data, value="999")
    print_test_result("数值列用可转换字符串填充", numeric_data, result3, "应该转换为数值并填充")

    # 测试2: 边界情况修复测试
    print_test_header("2. 边界情况修复测试")

    # 测试单行前向填充
    single_row = pd.DataFrame([[1, np.nan, 3, np.nan, 5]])
    result4 = fill_missing_data(single_row, method='ffill')
    print_test_result("单行前向填充", single_row, result4, "应该正确前向填充")

    # 测试后向填充的改进
    end_nan_data = pd.DataFrame({
        'A': [1, 2, 3, np.nan, np.nan],
        'B': [1, 2, np.nan, np.nan, np.nan]
    })

    result5 = fill_missing_data(end_nan_data, method='bfill')
    print_test_result("结尾NaN后向填充", end_nan_data, result5, "结尾NaN应该保留")

    # 测试3: 性能对比测试
    print_test_header("3. 性能对比测试")

    import time

    # 创建较大的测试数据
    large_data = pd.DataFrame({
        'A': np.random.choice([1, 2, 3, np.nan], size=1000),
        'B': np.random.choice([10.5, 20.5, 30.5, np.nan], size=1000),
        'C': np.random.choice(['x', 'y', 'z', np.nan], size=1000)
    })

    print(f"测试数据形状: {large_data.shape}")
    print(f"原始缺失值数量: {large_data.isnull().sum().sum()}")

    # 性能测试
    methods = [
        ("固定值填充", lambda: fill_missing_data(large_data, value="MISSING")),
        ("前向填充", lambda: fill_missing_data(large_data, method='ffill')),
        ("后向填充", lambda: fill_missing_data(large_data, method='bfill'))
    ]

    for method_name, method_func in methods:
        start_time = time.time()
        result = method_func()
        end_time = time.time()
        remaining_nulls = result.isnull().sum().sum()
        print(f"{method_name}: {end_time - start_time:.4f} 秒, 剩余缺失值: {remaining_nulls}")

    # 测试4: 异常处理完善测试
    print_test_header("4. 异常处理测试")

    exception_tests = [
        ("错误的method参数", lambda: fill_missing_data(numeric_data, method='invalid')),
        ("固定值填充但未提供value", lambda: fill_missing_data(numeric_data, method=None)),
        ("输入数据为None", lambda: fill_missing_data(None)),
        ("输入数据为空列表", lambda: fill_missing_data([])),
        ("输入数据为空字典", lambda: fill_missing_data({}))
    ]

    for test_name, test_func in exception_tests:
        try:
            test_func()
            print(f"❌ {test_name}: 应该抛出异常但没有")
        except Exception as e:
            print(f"✅ {test_name}: 正确捕获异常 - {type(e).__name__}: {str(e)}")

    # 测试5: 复杂数据类型测试
    print_test_header("5. 复杂数据类型处理测试")

    # 混合数据类型测试
    mixed_data = pd.DataFrame({
        'integers': [1, np.nan, 3, np.nan, 5],
        'floats': [1.1, np.nan, 3.3, np.nan, 5.5],
        'strings': ['a', np.nan, 'c', np.nan, 'e'],
        'booleans': [True, np.nan, False, np.nan, True],
        'dates': pd.to_datetime(['2023-01-01', np.nan, '2023-01-03', np.nan, '2023-01-05'])
    })

    result6 = fill_missing_data(mixed_data, value="MISSING")
    print_test_result("复杂混合数据类型", mixed_data, result6, "应该正确处理各种数据类型")

    print(f"\n{'=' * 60}")
    print("改进版本测试完成！")
    print(f"{'=' * 60}")


# 如果直接运行此模块，执行测试
if __name__ == "__main__":
    comprehensive_test_suite()
#-----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
"""
使用统计方法填充DataFrame中的缺失值

参数:
data: 输入的DataFrame
strategy: 填充策略，可选项：
    'mean': 使用均值填充（仅适用于数值列）
    'median': 使用中位数填充（仅适用于数值列）
    'most_frequent': 使用众数填充（适用于所有数据类型）
    'constant': 使用常量填充（适用于所有数据类型）
fill_value: 当strategy='constant'时使用的填充值
random_state: 随机种子，用于保证结果的可重现性

返回:
填充后的DataFrame副本

异常:
TypeError: 当输入不是DataFrame时
ValueError: 当输入为空或策略无效时
"""

def batch_statistical_imputation(data: pd.DataFrame,
                                 numeric_strategy: str = 'mean',
                                 categorical_strategy: str = 'most_frequent',
                                 fill_value: Any = None) -> pd.DataFrame:

    # 验证输入
    if not isinstance(data, pd.DataFrame):
        raise TypeError("输入的data必须是pandas DataFrame类型")

    if data.empty:
        raise ValueError("输入的数据表为空，无法进行填充操作")

    data_copy = data.copy()

    # 更精确的数值类型判断（排除时间相关类型和复数类型）
    numeric_columns = data_copy.select_dtypes(include=[np.number]).columns

    # 如果使用均值或中位数策略，排除复数类型
    if numeric_strategy in ['mean', 'median']:
        complex_columns = data_copy.select_dtypes(include=[np.complexfloating]).columns
        numeric_columns = numeric_columns.difference(complex_columns)

    # 所有非数值列都当作分类列处理
    categorical_columns = data_copy.columns.difference(numeric_columns)

    # 处理数值列
    if len(numeric_columns) > 0:
        numeric_data = data_copy[numeric_columns]
        filled_numeric = statistical_imputation_fill(
            numeric_data,
            strategy=numeric_strategy,
            fill_value=fill_value
        )
        data_copy[numeric_columns] = filled_numeric

    # 处理分类列
    if len(categorical_columns) > 0:
        categorical_data = data_copy[categorical_columns]
        filled_categorical = statistical_imputation_fill(
            categorical_data,
            strategy=categorical_strategy,
            fill_value=fill_value
        )
        data_copy[categorical_columns] = filled_categorical

    return data_copy


def imputation_report(data: pd.DataFrame,
                      strategy: str = 'mean',
                      fill_value: Any = None) -> Dict[str, Any]:

    # 验证输入
    if not isinstance(data, pd.DataFrame):
        raise TypeError("输入的data必须是pandas DataFrame类型")

    if data.empty:
        raise ValueError("输入的数据表为空，无法进行填充操作")

    report = {
        'original_shape': data.shape,
        'missing_values_before': data.isnull().sum().to_dict(),
        'missing_percentage_before': (data.isnull().sum() / len(data) * 100).to_dict(),
        'fill_values_used': {},
        'columns_processed': [],
        'strategy_used': strategy
    }

    # 计算每列的填充值
    for col_name in data.columns:
        if data[col_name].isnull().any():
            valid_values = data[col_name].dropna().tolist()

            if valid_values:
                try:
                    if strategy == 'mean' and all(isinstance(v, (int, float, np.number)) for v in valid_values):
                        fill_val = sum(valid_values) / len(valid_values)
                    elif strategy == 'median' and all(isinstance(v, (int, float, np.number)) for v in valid_values):
                        sorted_values = sorted(valid_values)
                        n = len(sorted_values)
                        fill_val = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2 if n % 2 == 0 else \
                        sorted_values[n // 2]
                    elif strategy == 'most_frequent':
                        counter = Counter(valid_values)
                        fill_val = counter.most_common(1)[0][0]
                    elif strategy == 'constant':
                        fill_val = fill_value
                    else:
                        continue

                    report['fill_values_used'][col_name] = fill_val
                    report['columns_processed'].append(col_name)

                except Exception:
                    # 如果某列无法用指定策略处理，跳过
                    continue

    return report


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建测试数据
    test_data = pd.DataFrame({
        'numeric_int': [1, 2, np.nan, 4, 5, np.nan, 7],
        'numeric_float': [1.1, 2.2, np.nan, 4.4, 5.5, np.nan, 7.7],
        'string_col': ['apple', 'banana', np.nan, 'apple', 'cherry', np.nan, 'banana'],
        'boolean_col': [True, False, np.nan, True, False, np.nan, True],
        'datetime_col': pd.to_datetime(
            ['2023-01-01', '2023-01-02', np.nan, '2023-01-01', '2023-01-04', np.nan, '2023-01-01'])
    })

    print("=== 统计填充函数使用示例 ===")
    print("\n原始数据:")
    print(test_data)
    print(f"\n数据形状: {test_data.shape}")
    print(f"缺失值统计:\n{test_data.isnull().sum()}")

    print("\n" + "=" * 50)

    # 示例1: 使用均值填充数值列
    print("\n1. 使用均值策略填充数值列:")
    try:
        numeric_data = test_data.select_dtypes(include=[np.number])
        result_mean = statistical_imputation_fill(numeric_data, strategy='mean')
        print(result_mean)
        print(f"剩余缺失值: {result_mean.isnull().sum().sum()}")
    except Exception as e:
        print(f"错误: {e}")

    # 示例2: 使用众数填充所有列
    print("\n2. 使用众数策略填充所有列:")
    try:
        result_mode = statistical_imputation_fill(test_data, strategy='most_frequent')
        print(result_mode)
        print(f"剩余缺失值: {result_mode.isnull().sum().sum()}")
    except Exception as e:
        print(f"错误: {e}")

    # 示例3: 使用批量处理
    print("\n3. 使用批量处理（自动选择策略）:")
    try:
        result_batch = batch_statistical_imputation(
            test_data,
            numeric_strategy='median',
            categorical_strategy='most_frequent'
        )
        print(result_batch)
        print(f"剩余缺失值: {result_batch.isnull().sum().sum()}")
    except Exception as e:
        print(f"错误: {e}")

    # 示例4: 生成填充报告
    print("\n4. 生成填充报告:")
    try:
        report = imputation_report(test_data, strategy='most_frequent')
        print("填充报告:")
        for key, value in report.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"错误: {e}")

    # 示例5: 常量填充
    print("\n5. 使用常量填充:")
    try:
        result_constant = statistical_imputation_fill(
            test_data,
            strategy='constant',
            fill_value='MISSING'
        )
        print(result_constant)
        print(f"剩余缺失值: {result_constant.isnull().sum().sum()}")
    except Exception as e:
        print(f"错误: {e}")

    print("\n" + "=" * 50)
    print("所有示例执行完成！")
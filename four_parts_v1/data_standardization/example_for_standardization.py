"""
Z-score标准化模块

将指定列的数据缩放为均值为0，标准差为1的标准正态分布

参数:
data : pd.DataFrame，输入的DataFrame数据
columns : str, int, or List[Union[str, int]]，需要标准化的列名或列索引，可以是单个值或列表
with_mean : bool, default=True，是否减去均值
with_std : bool, default=True，是否除以标准差

返回:
Tuple[pd.DataFrame, dict]
标准化后的DataFrame
包含各列统计信息的字典 {'列名': {'mean': 均值, 'std': 标准差}}

异常:
ValueError: 输入参数错误
KeyError: 列名不存在
TypeError: 数据类型错误
"""
# 使用示例和测试代码
if __name__ == "__main__":
    # 创建测试数据（修正版本）
    np.random.seed(42)
    test_data = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'score': np.random.uniform(0, 100, 100)
    })

    # 添加一些缺失值
    test_data.loc[5:10, 'age'] = np.nan
    test_data.loc[15:20, 'income'] = np.nan

    print("原始数据:")
    print(test_data.head(10))
    print("\n数据信息:")
    print(test_data.info())

    try:
        # 测试1: 单列标准化（age列）
        print("\n=== 测试单列标准化 ===")
        result1, stats1 = zscore_standardization(test_data, 'age')
        print("标准化后的age列前15行:")
        print(result1[['age']].head(15))
        print("统计信息:", stats1)

        # 测试2: 多列标准化
        print("\n=== 测试多列标准化 ===")
        result2, stats2 = zscore_standardization(test_data, ['income', 'score'])
        print("标准化结果:")
        print(result2[['income', 'score']].head(10))
        print("统计信息:", stats2)

        # 测试3: 使用列索引
        print("\n=== 测试使用列索引 ===")
        result3, stats3 = zscore_standardization(test_data, [0, 2])  # age和score列
        print("使用索引标准化结果:")
        print(result3[['age', 'score']].head(10))
        print("统计信息:", stats3)

        # 测试4: 验证标准化结果
        print("\n=== 验证标准化结果 ===")
        for col in ['age', 'income', 'score']:
            if col in result2.columns:
                valid_data = result2[col].dropna()
                print(f"{col}列 - 均值: {valid_data.mean():.6f}, 标准差: {valid_data.std():.6f}")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")

    # 测试异常情况
    print("\n=== 异常测试 ===")

    try:
        # 创建一个包含非数值列的测试数据
        test_data_with_string = test_data.copy()
        test_data_with_string['category'] = ['A', 'B', 'C'] * 33 + ['A']

        # 测试非数值列
        zscore_standardization(test_data_with_string, 'category')
    except Exception as e:
        print(f"非数值列错误: {e}")

    try:
        # 测试不存在的列
        zscore_standardization(test_data, 'nonexistent_column')
    except Exception as e:
        print(f"不存在列错误: {e}")

    try:
        # 测试索引超出范围
        zscore_standardization(test_data, 10)
    except Exception as e:
        print(f"索引超出范围错误: {e}")

#-----------------------------------------------------------------------------------------------------------------------

"""
Min-Max标准化模块

功能：将指定列的特征缩放到给定范围（通常是 [0, 1]）之间

参数:
data: pd.DataFrame，输入的DataFrame数据
columns: str/int/List[str/int]，需要标准化的列名或列索引
feature_range: List[float]，归一化范围，默认 [0, 1]

返回:
tuple: (标准化后的DataFrame, 标准化参数字典)
其中，标准化参数字典格式: {列名: {'min_val': 最小值, 'max_val': 最大值, 'scale': 原始范围}}

异常:
TypeError: 输入类型错误
ValueError: 输入值错误
KeyError: 列名不存在
IndexError: 列索引超出范围
"""
# 使用示例和测试函数
def test_minmax_scaler():
    """测试函数，展示模块的使用方法"""
    print("=== Min-Max标准化模块全面测试 ===\n")

    # 创建基础测试数据
    test_data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500],
        'D': ['a', 'b', 'c', 'd', 'e']  # 非数值列
    })

    print("基础测试数据:")
    print(test_data)
    print()

    # 基础功能测试
    print("=== 基础功能测试 ===")
    try:
        # 测试1：标准化单列（使用列名）
        print("测试1：标准化列 'A' (默认范围 [0,1])")
        result1, params1 = minmax_scaler(test_data, 'A')
        print("结果:")
        print(result1[['A']])
        print("标准化参数:", params1)
        print()

        # 测试2：标准化多列（使用列索引）
        print("测试2：标准化列索引 [0, 1] (范围 [-1,1])")
        result2, params2 = minmax_scaler(test_data, [0, 1], feature_range=[-1, 1])
        print("结果:")
        print(result2[['A', 'B']])
        print("说明：出现负数是正常的，因为指定了范围[-1,1]")
        print("标准化参数:", params2)
        print()

        # 测试3：混合使用列名和索引
        print("测试3：混合使用列名和索引")
        result3, params3 = minmax_scaler(test_data, ['A', 2])
        print("结果:")
        print(result3[['A', 'C']])
        print("标准化参数:", params3)
        print()

    except Exception as e:
        print(f"错误: {e}")

    # 高级功能测试
    print("=== 高级功能测试 ===")

    # 测试4：包含缺失值的数据
    print("测试4：包含缺失值的数据")
    data_with_nan = pd.DataFrame({
        'score': [85, 90, np.nan, 95, 80, np.nan, 88],
        'age': [25, np.nan, 30, 35, 28, 32, np.nan]
    })
    print("原始数据:")
    print(data_with_nan)
    try:
        result4, params4 = minmax_scaler(data_with_nan, ['score', 'age'])
        print("标准化后:")
        print(result4)
        print("说明：缺失值保持为NaN")
        print()
    except Exception as e:
        print(f"错误: {e}")

    # 测试5：浮点数数据
    print("测试5：浮点数数据")
    float_data = pd.DataFrame({
        'price': [19.99, 29.95, 15.50, 45.00, 12.99],
        'rating': [4.2, 3.8, 4.5, 3.9, 4.1]
    })
    print("原始数据:")
    print(float_data)
    try:
        result5, params5 = minmax_scaler(float_data, ['price', 'rating'], feature_range=[0, 10])
        print("标准化后 (范围 [0,10]):")
        print(result5)
        print()
    except Exception as e:
        print(f"错误: {e}")

    # 测试6：不同范围的标准化
    print("测试6：不同范围的标准化")
    try:
        # 测试不同的feature_range
        ranges_to_test = [[0, 1], [-1, 1], [0, 100], [-5, 5]]
        for range_val in ranges_to_test:
            result, params = minmax_scaler(test_data, 'A', feature_range=range_val)
            print(f"范围 {range_val}: {result['A'].tolist()}")
        print()
    except Exception as e:
        print(f"错误: {e}")

    # 测试7：大数据集性能测试
    print("测试7：大数据集性能测试")
    large_data = pd.DataFrame({
        'values': np.random.randn(10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000)
    })
    print(f"大数据集: {large_data.shape}")
    try:
        import time
        start_time = time.time()
        result7, params7 = minmax_scaler(large_data, 'values')
        end_time = time.time()
        print(f"处理时间: {end_time - start_time:.4f}秒")
        print(f"结果统计: min={result7['values'].min():.4f}, max={result7['values'].max():.4f}")
        print()
    except Exception as e:
        print(f"错误: {e}")

    # 异常情况测试
    print("=== 异常情况测试 ===")

    # 测试8：非数值列
    try:
        minmax_scaler(test_data, 'D')
    except ValueError as e:
        print(f"✓ 非数值列异常捕获: {e}")

    # 测试9：不存在的列
    try:
        minmax_scaler(test_data, 'X')
    except KeyError as e:
        print(f"✓ 不存在列异常捕获: {e}")

    # 测试10：错误的feature_range
    try:
        minmax_scaler(test_data, 'A', feature_range=[1, 0])
    except ValueError as e:
        print(f"✓ 错误范围异常捕获: {e}")

    # 测试11：全缺失值数据
    try:
        all_nan_data = pd.DataFrame({'col': [np.nan, np.nan, np.nan]})
        minmax_scaler(all_nan_data, 'col')
    except ValueError as e:
        print(f"✓ 全缺失值异常捕获: {e}")

    # 测试12：所有值相同的数据
    try:
        same_value_data = pd.DataFrame({'col': [5, 5, 5, 5]})
        minmax_scaler(same_value_data, 'col')
    except ValueError as e:
        print(f"✓ 相同值异常捕获: {e}")

    # 测试13：空DataFrame
    try:
        empty_data = pd.DataFrame()
        minmax_scaler(empty_data, 'A')
    except ValueError as e:
        print(f"✓ 空DataFrame异常捕获: {e}")

    # 测试14：错误的输入类型
    try:
        minmax_scaler([1, 2, 3], 'A')
    except TypeError as e:
        print(f"✓ 错误输入类型异常捕获: {e}")

    # 测试15：列索引超出范围
    try:
        minmax_scaler(test_data, 10)
    except IndexError as e:
        print(f"✓ 列索引超范围异常捕获: {e}")

    # 测试16：混合数据类型
    print("\n测试16：混合数据类型处理")
    mixed_data = pd.DataFrame({
        'numbers': [1, 2, 3, 4, 5],
        'mixed': [1, 'text', 3.5, 4, None]
    })
    print("混合数据:")
    print(mixed_data)
    try:
        result16, params16 = minmax_scaler(mixed_data, 'mixed')
        print("处理结果:")
        print(result16[['mixed']])
    except ValueError as e:
        print(f"✓ 混合数据类型异常捕获: {e}")

    print("\n=== 测试完成 ===")
    print("说明：")
    print("1. 测试2出现负数是正常的，因为指定了范围[-1,1]")
    print("2. Min-Max标准化可以将数据缩放到任意指定范围")
    print("3. 缺失值会被保持为NaN")
    print("4. 所有异常情况都得到了正确处理")


if __name__ == "__main__":
    test_minmax_scaler()
#-----------------------------------------------------------------------------------------------------------------------
"""
对DataFrame进行Robust标准化处理（基于中位数和IQR）

参数:
data : pd.DataFrame，输入的DataFrame数据
columns : str, List[str], int, List[int]，需要标准化的列名或列索引位置
quantile_range : List[float], default=[25, 75]，IQR范围的百分位数，默认使用25%-75%分位数
inplace : bool, default=False，是否在原DataFrame上直接修改

返回:
如果inplace=True: 
    返回修改后的DataFrame和变换参数字典
如果inplace=False: 
    返回新的DataFrame和变换参数字典

异常:
ValueError: 输入参数不合法时抛出
TypeError: 数据类型不匹配时抛出
KeyError: 列名不存在时抛出
"""

# 测试集
def run_robust_standardization_tests():
    """
    运行robust标准化函数的全面测试集
    """
    print("=" * 70)
    print("ROBUST标准化函数测试集")
    print("=" * 70)

    # 测试计数器
    test_count = 0
    passed_count = 0

    def test_case(test_name, test_func):
        nonlocal test_count, passed_count
        test_count += 1
        print(f"\n测试 {test_count}: {test_name}")
        print("-" * 50)
        try:
            test_func()
            print("✅ 测试通过")
            passed_count += 1
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

    # 测试1: 基本功能测试
    def test_basic_functionality():
        np.random.seed(42)
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })

        scaled_df, params = robust_standardization(df, columns=['A', 'B'])

        # 验证返回类型
        assert isinstance(scaled_df, pd.DataFrame)
        assert isinstance(params, dict)

        # 验证列数不变
        assert len(scaled_df.columns) == len(df.columns)

        # 验证参数结构
        for col in ['A', 'B']:
            assert col in params
            assert 'median' in params[col]
            assert 'iqr' in params[col]
            assert 'q_low' in params[col]
            assert 'q_high' in params[col]

        print("基本功能验证通过")

    # 测试2: 列名和列索引混合使用
    def test_column_specification():
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'col3': [100, 200, 300, 400, 500]
        })

        # 使用列名
        scaled_df1, _ = robust_standardization(df, columns=['col1', 'col2'])

        # 使用列索引
        scaled_df2, _ = robust_standardization(df, columns=[0, 1])

        # 混合使用
        scaled_df3, _ = robust_standardization(df, columns=['col1', 1])

        # 验证结果一致
        pd.testing.assert_frame_equal(scaled_df1, scaled_df2)
        pd.testing.assert_frame_equal(scaled_df1, scaled_df3)

        print("列名和索引混合使用验证通过")

    # 测试3: 缺失值处理
    def test_missing_values():
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
            'B': [10, np.nan, 30, 40, 50, 60, np.nan, 80, 90, 100]
        })

        scaled_df, params = robust_standardization(df, columns=['A', 'B'])

        # 验证缺失值位置保持不变
        assert scaled_df['A'].isna().sum() == df['A'].isna().sum()
        assert scaled_df['B'].isna().sum() == df['B'].isna().sum()

        # 验证缺失值位置相同
        pd.testing.assert_series_equal(scaled_df['A'].isna(), df['A'].isna())
        pd.testing.assert_series_equal(scaled_df['B'].isna(), df['B'].isna())

        print("缺失值处理验证通过")

    # 测试4: 离群值鲁棒性
    def test_outlier_robustness():
        # 创建包含离群值的数据
        normal_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        outlier_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]  # 1000是离群值

        df_normal = pd.DataFrame({'A': normal_data})
        df_outlier = pd.DataFrame({'A': outlier_data})

        # 标准化
        scaled_normal, params_normal = robust_standardization(df_normal, columns=['A'])
        scaled_outlier, params_outlier = robust_standardization(df_outlier, columns=['A'])

        # Robust标准化对离群值应该不敏感
        # 中位数应该相同或接近
        assert abs(params_normal['A']['median'] - params_outlier['A']['median']) < 1

        print("离群值鲁棒性验证通过")

    # 测试5: 不同分位数范围
    def test_quantile_ranges():
        df = pd.DataFrame({'A': np.random.normal(0, 1, 100)})

        # 测试不同的分位数范围
        ranges = [[10, 90], [25, 75], [5, 95]]

        for q_range in ranges:
            scaled_df, params = robust_standardization(df, columns=['A'], quantile_range=q_range)

            # 验证分位数范围被正确记录
            assert params['A']['quantile_range'] == q_range

            # 验证IQR > 0
            assert params['A']['iqr'] > 0

        print("不同分位数范围验证通过")

    # 测试6: inplace参数
    def test_inplace_parameter():
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
        df_original = df.copy()

        # 测试inplace=False（默认）
        scaled_df, _ = robust_standardization(df, columns=['A'])
        pd.testing.assert_frame_equal(df, df_original)  # 原数据不变

        # 测试inplace=True
        scaled_df_inplace, _ = robust_standardization(df, columns=['A'], inplace=True)
        assert scaled_df_inplace is df  # 返回同一对象
        assert not df.equals(df_original)  # 原数据已修改

        print("inplace参数验证通过")

    # 测试7: 异常情况处理
    def test_exception_handling():
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],  # 非数值列
            'C': [1, 1, 1, 1, 1]  # 所有值相同（IQR=0）
        })

        exception_tests = [
            # 测试用例: (测试函数, 期望异常类型, 描述)
            (lambda: robust_standardization([1, 2, 3], columns=['A']), TypeError, "非DataFrame输入"),
            (lambda: robust_standardization(pd.DataFrame(), columns=['A']), ValueError, "空DataFrame"),
            (lambda: robust_standardization(df, columns=['nonexistent']), KeyError, "不存在的列名"),
            (lambda: robust_standardization(df, columns=[10]), ValueError, "列索引超出范围"),
            (lambda: robust_standardization(df, columns=['B']), ValueError, "非数值列"),
            (lambda: robust_standardization(df, columns=['C']), ValueError, "IQR为0"),
            (lambda: robust_standardization(df, columns=['A'], quantile_range=[75, 25]), ValueError, "无效分位数范围"),
        ]

        for test_func, expected_exception, description in exception_tests:
            try:
                test_func()
                assert False, f"{description} 应该抛出 {expected_exception.__name__}"
            except expected_exception:
                pass  # 正确抛出了期望的异常
            except Exception as e:
                # 如果抛出了其他异常，检查是否是包装后的ValueError
                if expected_exception != ValueError or "ValueError" not in str(type(e)):
                    raise AssertionError(
                        f"{description} 期望抛出 {expected_exception.__name__}，但抛出了 {type(e).__name__}: {e}")

        print("异常处理验证通过")

    # 测试8: 边界情况
    def test_edge_cases():
        # 只有两行数据
        df_two = pd.DataFrame({'A': [1, 2]})
        scaled_df, params = robust_standardization(df_two, columns=['A'])
        assert params['A']['median'] == 1.5
        # 两行数据的IQR应该不为0
        assert params['A']['iqr'] > 0

        # 只有三行数据（确保IQR不为0）
        df_three = pd.DataFrame({'A': [1, 2, 3]})
        scaled_df, params = robust_standardization(df_three, columns=['A'])
        assert params['A']['median'] == 2.0
        assert params['A']['iqr'] > 0

        # 包含极大值
        df_extreme = pd.DataFrame({'A': [1, 2, 3, 1e10]})
        scaled_df, params = robust_standardization(df_extreme, columns=['A'])
        # 中位数应该不受极值影响
        assert params['A']['median'] == 2.5

        # 测试单行数据会抛出异常（因为无法计算有效的IQR）
        df_single = pd.DataFrame({'A': [1]})
        try:
            robust_standardization(df_single, columns=['A'])
            assert False, "单行数据应该抛出异常"
        except ValueError:
            pass  # 正确抛出异常

        print("边界情况验证通过")

    # 测试9: 数据类型保持
    def test_data_type_preservation():
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'mixed_col': [1, 2.5, 3, 4.7, 5]
        })

        scaled_df, _ = robust_standardization(df, columns=['int_col', 'float_col', 'mixed_col'])

        # 标准化后应该都是float类型
        for col in ['int_col', 'float_col', 'mixed_col']:
            assert pd.api.types.is_float_dtype(scaled_df[col])

        print("数据类型处理验证通过")

    # 运行所有测试
    test_case("基本功能测试", test_basic_functionality)
    test_case("列名和列索引混合使用", test_column_specification)
    test_case("缺失值处理", test_missing_values)
    test_case("离群值鲁棒性", test_outlier_robustness)
    test_case("不同分位数范围", test_quantile_ranges)
    test_case("inplace参数", test_inplace_parameter)
    test_case("异常情况处理", test_exception_handling)
    test_case("边界情况", test_edge_cases)
    test_case("数据类型保持", test_data_type_preservation)

    # 测试总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"总测试数: {test_count}")
    print(f"通过测试数: {passed_count}")
    print(f"失败测试数: {test_count - passed_count}")
    print(f"通过率: {passed_count / test_count * 100:.1f}%")

    if passed_count == test_count:
        print("🎉 所有测试通过！代码质量良好。")
    else:
        print("⚠️  部分测试失败，需要检查代码。")


# 性能测试
def run_performance_test():
    """
    运行性能测试
    """
    print("\n" + "=" * 70)
    print("性能测试")
    print("=" * 70)

    import time

    # 创建大数据集
    np.random.seed(42)
    sizes = [1000, 10000, 100000]

    for size in sizes:
        df = pd.DataFrame({
            'A': np.random.normal(0, 1, size),
            'B': np.random.exponential(2, size),
            'C': np.random.uniform(0, 100, size)
        })

        # 添加一些缺失值
        nan_indices = np.random.choice(size, size // 20, replace=False)
        df.loc[nan_indices, 'A'] = np.nan

        start_time = time.time()
        scaled_df, params = robust_standardization(df, columns=['A', 'B', 'C'])
        end_time = time.time()

        print(f"数据量: {size:,} 行 × 3 列")
        print(f"处理时间: {end_time - start_time:.4f} 秒")
        print(f"处理速度: {size / (end_time - start_time):,.0f} 行/秒")
        print()


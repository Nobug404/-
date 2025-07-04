"""
独热编码转换器

对DataFrame中指定列进行独热编码转换，支持列名或位置索引指定。

参数:
    data: pandas DataFrame，输入数据
    columns: 需要编码的列名(str)、位置索引(int)或它们的列表
    drop_first: 是否丢弃第一类防止共线性，默认 False
    handle_unknown: 处理未见类别的方式 ('error', 'ignore', 'zero')，默认 'error'
    sparse: 是否使用稀疏矩阵存储，默认 False
    return_dataframe: 是否返回DataFrame格式，默认 True
    prefix: 编码列的前缀，可以是字符串或字典 {原列名: 前缀}
    prefix_sep: 前缀分隔符，默认 '_'

返回:
    如果 return_dataframe=True:
        返回: (encoded_dataframe, encoding_info)
    如果 return_dataframe=False:
        返回: (encoding_matrix, encoding_info)

    其中encoding_info为包含编码信息的字典
    {
        'column_name': {
            'category_index': {类别: 全局列索引},
            'original_categories': [原始类别列表],
            'encoded_columns': [编码后的列名],
            'summary': 处理摘要信息
        }
    }

异常:
    ValueError: 输入参数不合法时抛出
    TypeError: 输入类型不正确时抛出
    KeyError: 指定的列名不存在时抛出
    IndexError: 指定的列索引超出范围时抛出
"""
# 使用示例和测试函数
def demo_dataframe_one_hot_encoder():
    """演示DataFrame独热编码器的使用方法"""
    print("=== DataFrame独热编码器演示 ===\n")

    # 创建示例数据
    df = pd.DataFrame({
        'color': ['red', 'blue', 'red', 'green', 'blue'],
        'size': ['S', 'M', 'L', 'S', 'M'],
        'price': [10, 20, 15, 25, 18],
        'category': ['A', 'B', 'A', 'C', 'B']
    })

    print("原始数据:")
    print(df)
    print()

    # 示例1：使用列名进行单列编码
    print("示例1：使用列名进行单列编码")
    result1, info1 = dataframe_one_hot_encode(df, 'color')
    print("编码后的DataFrame:")
    print(result1)
    print(f"编码信息: {info1['color']['summary']}\n")

    # 示例2：使用位置索引进行多列编码
    print("示例2：使用位置索引进行多列编码")
    result2, info2 = dataframe_one_hot_encode(df, [0, 1], prefix={'color': 'col', 'size': 'sz'})
    print("编码后的DataFrame:")
    print(result2)
    print()

    # 示例3：丢弃第一类并返回矩阵
    print("示例3：丢弃第一类并返回编码矩阵")
    result3, info3 = dataframe_one_hot_encode(df, ['color', 'category'],
                                              drop_first=True,
                                              return_dataframe=False)
    print("编码矩阵:")
    print(result3)
    print(f"矩阵形状: {result3.shape}")
    print(f"列信息: {list(info3.keys())}")

    # 显示详细的编码信息
    for col_name, info in info3.items():
        print(f"\n{col_name} 列编码信息:")
        print(f"  原始类别: {info['original_categories']}")
        print(f"  编码列名: {info['encoded_columns']}")
        print(f"  是否丢弃第一类: {info['summary']['dropped_first']}")
    print()

    # 示例4：错误处理演示
    print("示例4：错误处理演示")
    try:
        # 尝试编码不存在的列
        dataframe_one_hot_encode(df, 'nonexistent_column')
    except KeyError as e:
        print(f"捕获到KeyError: {e}")

    try:
        # 尝试使用超出范围的索引
        dataframe_one_hot_encode(df, 10)
    except IndexError as e:
        print(f"捕获到IndexError: {e}")


if __name__ == "__main__":
    demo_dataframe_one_hot_encoder()
#-----------------------------------------------------------------------------------------------------------------------

"""
特征工程分箱模块

参数:
data: DataFrame形式的输入数据
columns: 需要分箱的列名(str/List[str])或列索引(int/List[int])
bins: 分箱数量(int)或分箱边界(List[float])
method: 分箱方法，"equal_width"(等宽)或"equal_freq"(等频)
labels: 自定义箱子标签
right_closed: 是否右闭区间
handle_na: 缺失值处理方式，"ignore"或"separate"
precision: 标签显示精度

返回:
包含原数据和分箱结果的DataFrame
"""

# 使用示例和测试代码
if __name__ == "__main__":
    # 创建测试数据
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
        # 测试等宽分箱
        print("\n=== 测试等宽分箱 ===")
        result1 = feature_binning(test_data, 'age', 5, method='equal_width')
        print("等宽分箱结果:")
        print(result1[['age', 'age_binned']].head(15))

        # 测试等频分箱
        print("\n=== 测试等频分箱 ===")
        result2 = feature_binning(test_data, ['income'], 4, method='equal_freq', handle_na='separate')
        print("等频分箱结果:")
        print(result2[['income', 'income_binned']].head(25))

        # 测试自定义边界
        print("\n=== 测试自定义边界 ===")
        result3 = feature_binning(test_data, 2, [0, 25, 50, 75, 100], method='equal_width')
        print("自定义边界分箱结果:")
        print(result3[['score', 'score_binned']].head(10))

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
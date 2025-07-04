"""
将DataFrame数据随机划分为训练集和测试集，保证行数据的对应性

参数:
data: pd.DataFrame, 输入的数据集
target_column: str, 目标列名（标签列）
test_size: float, 测试集比例，默认0.2（20%）
random_state: int, 随机种子，确保可复现性，默认None
stratify: bool, 是否进行分层抽样，默认False

返回:
X_train: pd.DataFrame, 训练集特征
X_test: pd.DataFrame, 测试集特征
y_train: pd.Series, 训练集标签
y_test: pd.Series, 测试集标签

异常:
ValueError: 输入参数不合法时抛出
KeyError: 目标列不存在时抛出
"""

# 使用示例和测试函数
def test_random_split_dataset():
    """测试函数"""
    # 创建示例数据
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randint(0, 10, 100),
        'target': np.random.choice(['A', 'B', 'C'], 100)
    })

    print("原始数据形状:", sample_data.shape)
    print("标签分布:", sample_data['target'].value_counts().to_dict())

    # 测试基本划分
    X_train, X_test, y_train, y_test = random_split_dataset(
        data=sample_data,
        target_column='target',
        test_size=0.2,
        random_state=42
    )

    print(f"\n基本划分结果:")
    print(f"训练集: X_train{X_train.shape}, y_train{y_train.shape}")
    print(f"测试集: X_test{X_test.shape}, y_test{y_test.shape}")
    print(f"训练集标签分布: {y_train.value_counts().to_dict()}")
    print(f"测试集标签分布: {y_test.value_counts().to_dict()}")

    # 测试分层抽样
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = random_split_dataset(
        data=sample_data,
        target_column='target',
        test_size=0.2,
        random_state=42,
        stratify=True
    )

    print(f"\n分层抽样结果:")
    print(f"训练集: X_train{X_train_strat.shape}, y_train{y_train_strat.shape}")
    print(f"测试集: X_test{X_test_strat.shape}, y_test{y_test_strat.shape}")
    print(f"训练集标签分布: {y_train_strat.value_counts().to_dict()}")
    print(f"测试集标签分布: {y_test_strat.value_counts().to_dict()}")


if __name__ == "__main__":
    test_random_split_dataset()

#-----------------------------------------------------------------------------------------------------------------------

"""
时间序列数据集划分模块

将按时间排序的DataFrame划分为多组训练集和测试集，保证：
    1. 不打乱样本顺序
    2. 每一折的训练集为过去数据，测试集为未来数据
    3. 可选间隔gap避免信息泄露
    4. 保持行数据的对应性

参数:
data : pd.DataFrame，输入的DataFrame数据，必须按时间顺序排列
target_column : str，目标列名称
n_splits : int，折数，表示要生成几组训练/测试对
test_size : int, optional，每组测试集的样本数量，若未指定将自动推算
gap : int, optional，训练集与测试集之间的间隔长度，防止信息泄露，默认为0
max_train_size : int, optional，限制训练集最大长度（模拟滚动窗口），默认不限制

返回值:
List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]，每个元素包含 (X_train, X_test, y_train, y_test)

异常:
ValueError: 当输入参数不合法时
KeyError: 当目标列不存在时
"""
# 使用示例和测试函数
def demo_timeseries_split():
    """演示函数的使用方法"""
    print("=== 时间序列划分模块演示 ===\n")

    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100)
    })

    print("原始数据形状:", data.shape)
    print("原始数据前5行:")
    print(data.head())
    print()

    try:
        # 进行时间序列划分
        splits = timeseries_split(
            data=data,
            target_column='target',
            n_splits=3,
            test_size=10,
            gap=5,
            max_train_size=30
        )

        print("\n=== 划分结果详情 ===")
        for i, (X_train, X_test, y_train, y_test) in enumerate(splits):
            print(f"\n第{i + 1}折:")
            print(f"  训练集特征形状: {X_train.shape}")
            print(f"  训练集目标形状: {y_train.shape}")
            print(f"  测试集特征形状: {X_test.shape}")
            print(f"  测试集目标形状: {y_test.shape}")

            # 显示原始索引范围
            print(f"  训练集原始索引范围: {X_train['index'].min()} ~ {X_train['index'].max()}")
            print(f"  测试集原始索引范围: {X_test['index'].min()} ~ {X_test['index'].max()}")

    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    demo_timeseries_split()
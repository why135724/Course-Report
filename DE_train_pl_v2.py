import os
import numpy as np
from typing import Tuple
import time
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import fractional_matrix_power
import random

# SVM相关库
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import joblib

# 设置随机种子
seed = 1
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
print(f'Seed:{seed}')


# 差分熵计算
def compute_differential_entropy(data, fs=300, band_freqs=None):
    """
    计算差分熵(Differential Entropy)

    参数:
        data: EEG数据, shape (n_samples, n_channels, n_times) 或 (n_channels, n_times)
        fs: 采样频率
        band_freqs: 频带划分, 默认使用DEAP数据集的5个频带

    返回:
        de_features: 差分熵特征
    """
    if band_freqs is None:
        # 默认使用DEAP数据集的5个频带
        band_freqs = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

    if len(data.shape) == 2:
        data = data[np.newaxis, ...]

    n_samples, n_channels, n_times = data.shape
    n_bands = len(band_freqs)

    de_features = np.zeros((n_samples, n_channels, n_bands))

    for i in range(n_samples):
        for ch in range(n_channels):
            # 计算PSD
            freqs, psd = signal.welch(data[i, ch], fs=fs, nperseg=min(256, n_times))

            for band_idx, (band_name, (low, high)) in enumerate(band_freqs.items()):
                # 找到频带对应的频率索引
                idx = np.where((freqs >= low) & (freqs <= high))[0]

                if len(idx) > 0:
                    # 计算频带内的功率
                    band_power = np.trapz(psd[idx], freqs[idx])

                    if band_power > 0:
                        # 计算差分熵: DE = 0.5 * log(2πeσ²) ≈ 0.5 * log(功率)
                        de = 0.5 * np.log(2 * np.pi * np.e * band_power)
                        de_features[i, ch, band_idx] = de

    return de_features


def extract_de_features(X_data, fs=200):
    """
    提取差分熵特征

    参数:
        X_data: EEG数据, shape (n_samples, n_channels, n_times)
        fs: 采样频率

    返回:
        X_features: 特征矩阵, shape (n_samples, n_features)
    """
    n_samples = X_data.shape[0]

    # 计算差分熵特征
    de_features = compute_differential_entropy(X_data, fs=fs)

    # 展平特征: (n_samples, n_channels, n_bands) -> (n_samples, n_channels * n_bands)
    n_samples, n_channels, n_bands = de_features.shape
    X_features = de_features.reshape(n_samples, -1)

    return X_features


# EA (保持与原始代码相同)
def EA(x):
    """欧几里得对齐"""
    cov = np.zeros((x.shape[0], 24, 24))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (0.00000001) * np.eye(24)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def load_car_2d_data_split() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 CAR 2D 窗口，并按"每类前半段 -> train/val，后半段 -> test"划分。
    sad_car_2D.npy: [N_sad, C, T]
    happy_car_2D.npy: [N_happy, C, T]

    返回:
        X_trainval, y_trainval, X_test, y_test
    """
    sad = np.load("./processed_data/sad_car_2D.npy")  # [N_sad, C, T]
    happy = np.load("./processed_data/happy_car_2D.npy")  # [N_happy, C, T]

    n_sad = sad.shape[0]
    n_happy = happy.shape[0]
    mid_sad = n_sad // 2
    mid_happy = n_happy // 2

    # 前半段 → train/val
    sad_trainval = sad[:mid_sad]
    happy_trainval = happy[:mid_happy]
    X_trainval = np.concatenate([sad_trainval, happy_trainval], axis=0)
    y_trainval = np.concatenate([
        np.zeros(sad_trainval.shape[0], dtype=np.int64),
        np.ones(happy_trainval.shape[0], dtype=np.int64)
    ], axis=0)

    # 后半段 → test
    sad_test = sad[mid_sad:]
    happy_test = happy[mid_happy:]
    X_test = np.concatenate([sad_test, happy_test], axis=0)
    y_test = np.concatenate([
        np.zeros(sad_test.shape[0], dtype=np.int64),
        np.ones(happy_test.shape[0], dtype=np.int64)
    ], axis=0)

    # 应用EA预处理（与原始代码保持一致）
    # X_trainval = EA(X_trainval)
    # X_test = EA(X_test)

    print(f'X_trainval shape: {X_trainval.shape}')
    print(f'X_test shape: {X_test.shape}')

    return X_trainval, y_trainval, X_test, y_test


def split_train_val_data(X_trainval, y_trainval, val_ratio=0.125):
    """
    将trainval数据划分为训练集和验证集
    按照原始代码的7:1:2划分，这里使用7:1（丢弃2部分）
    """
    n_total = len(X_trainval)

    # 生成随机索引
    indices = np.arange(n_total)
    np.random.seed(seed)
    np.random.shuffle(indices)

    # 计算划分点
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)

    # 划分训练集和验证集
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]

    X_train = X_trainval[train_indices]
    y_train = y_trainval[train_indices]
    X_val = X_trainval[val_indices]
    y_val = y_trainval[val_indices]

    print(f"Train set: {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")

    return X_train, y_train, X_val, y_val


def train_svm_model(X_train, y_train, X_val, y_val, param_grid=None):
    """
    训练SVM模型

    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征
        y_val: 验证标签
        param_grid: 网格搜索参数

    返回:
        best_model: 最佳SVM模型
        scaler: 标准化器
        val_acc: 验证集准确率
    """
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if param_grid is None:
        # 默认参数网格
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }

    # 使用网格搜索寻找最佳参数
    print("开始网格搜索...")
    start_time = time.time()

    # 使用分层交叉验证
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=seed)

    svm = SVC(random_state=seed, probability=True)
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    # 获取最佳模型
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # 在验证集上评估
    y_val_pred = best_model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, y_val_pred)

    end_time = time.time()

    print(f"网格搜索完成，耗时: {end_time - start_time:.2f}秒")
    print(f"最佳参数: {best_params}")
    print(f"最佳交叉验证准确率: {best_score:.4f}")
    print(f"验证集准确率: {val_acc:.4f}")
    print("验证集分类报告:")
    print(classification_report(y_val, y_val_pred))
    print("验证集混淆矩阵:")
    print(confusion_matrix(y_val, y_val_pred))

    return best_model, scaler, val_acc


def DE_SVM_train_test():
    """
    基于差分熵(DE)特征和SVM的训练和测试
    """
    print("=" * 60)
    print("开始DE+SVM训练与测试")
    print("=" * 60)

    # 1. 加载数据（与原始代码完全相同）
    X_trainval, y_trainval, X_test, y_test = load_car_2d_data_split()

    # 2. 划分训练集和验证集（与原始代码相同的划分策略）
    X_train, y_train, X_val, y_val = split_train_val_data(X_trainval, y_trainval)

    # 3. 提取差分熵特征
    print("\n提取训练集DE特征...")
    X_train_de = extract_de_features(X_train)
    print(f"训练集DE特征形状: {X_train_de.shape}")

    print("\n提取验证集DE特征...")
    X_val_de = extract_de_features(X_val)
    print(f"验证集DE特征形状: {X_val_de.shape}")

    print("\n提取测试集DE特征...")
    X_test_de = extract_de_features(X_test)
    print(f"测试集DE特征形状: {X_test_de.shape}")

    # 4. 训练SVM模型
    print("\n" + "=" * 40)
    print("训练SVM模型")
    print("=" * 40)

    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }

    svm_model, scaler, val_acc = train_svm_model(
        X_train_de, y_train,
        X_val_de, y_val,
        param_grid=param_grid
    )

    # 5. 在测试集上评估
    print("\n" + "=" * 40)
    print("测试集评估")
    print("=" * 40)

    # 标准化测试集特征
    X_test_scaled = scaler.transform(X_test_de)

    # 预测
    y_test_pred = svm_model.predict(X_test_scaled)
    y_test_prob = svm_model.predict_proba(X_test_scaled)[:, 1]

    # 计算评估指标
    test_acc = accuracy_score(y_test, y_test_pred)
    test_report = classification_report(y_test, y_test_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    print(f"测试集准确率: {test_acc:.4f}")
    print("测试集分类报告:")
    print(test_report)
    print("测试集混淆矩阵:")
    print(test_cm)

    # 6. 保存模型
    print("\n" + "=" * 40)
    print("保存模型")
    print("=" * 40)

    # 保存SVM模型和标准化器
    model_save_path = 'best_de_svm_model.pkl'
    scaler_save_path = 'de_svm_scaler.pkl'

    joblib.dump(svm_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    print(f"SVM模型已保存到: {model_save_path}")
    print(f"标准化器已保存到: {scaler_save_path}")

    # 7. 可视化特征
    plot_de_features(X_train_de, y_train, X_test_de, y_test)

    return {
        'model': svm_model,
        'scaler': scaler,
        'train_acc': svm_model.score(scaler.transform(X_train_de), y_train),
        'val_acc': val_acc,
        'test_acc': test_acc,
        'X_train_de': X_train_de,
        'X_val_de': X_val_de,
        'X_test_de': X_test_de
    }


def plot_de_features(X_train_de, y_train, X_test_de, y_test):
    """
    可视化DE特征
    """
    # 使用PCA降维可视化
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # 合并训练和测试集进行PCA拟合
    X_all = np.vstack([X_train_de, X_test_de])

    # PCA降维到2D
    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_all)

    # 分离训练和测试集的PCA结果
    n_train = len(X_train_de)
    X_train_pca = X_pca[:n_train]
    X_test_pca = X_pca[n_train:]

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 训练集PCA可视化
    scatter1 = axes[0].scatter(
        X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1],
        c='blue', alpha=0.6, label='Sad (Train)', edgecolors='w', linewidths=0.5
    )
    scatter2 = axes[0].scatter(
        X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1],
        c='red', alpha=0.6, label='Happy (Train)', edgecolors='w', linewidths=0.5
    )
    axes[0].set_xlabel('PCA Component 1')
    axes[0].set_ylabel('PCA Component 2')
    axes[0].set_title('DE Features - Training Set (PCA)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 测试集PCA可视化
    scatter3 = axes[1].scatter(
        X_test_pca[y_test == 0, 0], X_test_pca[y_test == 0, 1],
        c='blue', alpha=0.6, label='Sad (Test)', edgecolors='w', linewidths=0.5
    )
    scatter4 = axes[1].scatter(
        X_test_pca[y_test == 1, 0], X_test_pca[y_test == 1, 1],
        c='red', alpha=0.6, label='Happy (Test)', edgecolors='w', linewidths=0.5
    )
    axes[1].set_xlabel('PCA Component 1')
    axes[1].set_ylabel('PCA Component 2')
    axes[1].set_title('DE Features - Test Set (PCA)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Differential Entropy Features Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('de_features_visualization.png', dpi=100, bbox_inches='tight')
    plt.show()

    # 解释方差比
    print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
    print(f"累计解释方差: {sum(pca.explained_variance_ratio_):.4f}")


def load_and_predict(model_path, scaler_path, new_data):
    """
    加载训练好的模型进行预测

    参数:
        model_path: 模型路径
        scaler_path: 标准化器路径
        new_data: 新数据, shape (n_samples, n_channels, n_times)

    返回:
        predictions: 预测结果
        probabilities: 预测概率
    """
    # 加载模型和标准化器
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 提取DE特征
    de_features = extract_de_features(new_data)

    # 标准化
    features_scaled = scaler.transform(de_features)

    # 预测
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)

    return predictions, probabilities


if __name__ == "__main__":
    # 运行DE+SVM训练测试
    results = DE_SVM_train_test()

    print("\n" + "=" * 60)
    print("训练结果总结")
    print("=" * 60)
    print(f"训练集准确率: {results['train_acc']:.4f}")
    print(f"验证集准确率: {results['val_acc']:.4f}")
    print(f"测试集准确率: {results['test_acc']:.4f}")

    # 保存特征用于后续分析
    np.save('X_train_de.npy', results['X_train_de'])
    np.save('X_test_de.npy', results['X_test_de'])
    print("\nDE特征已保存为 X_train_de.npy 和 X_test_de.npy")
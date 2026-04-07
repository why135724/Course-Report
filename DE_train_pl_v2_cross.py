import os
import numpy as np
from typing import Tuple, Dict, List
import time
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import fractional_matrix_power
import random
import glob

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


def load_all_subjects_data(data_dir="./processed_data"):
    """
    加载所有被试的数据

    假设文件名格式为: subject_XX_happy_car_2D.npy 和 subject_XX_sad_car_2D.npy
    XX: 两位数被试编号

    返回:
        data_dict: 字典，键为被试ID，值为元组(happy_data, sad_data)
    """
    data_dict = {}

    # 获取所有npy文件
    happy_files = sorted(glob.glob(os.path.join(data_dir, "subject_*_happy_car_2D.npy")))
    sad_files = sorted(glob.glob(os.path.join(data_dir, "subject_*_sad_car_2D.npy")))

    print(f"找到 {len(happy_files)} 个高兴数据文件")
    print(f"找到 {len(sad_files)} 个悲伤数据文件")

    # 提取所有被试ID
    subject_ids = []
    for f in happy_files:
        filename = os.path.basename(f)
        # 提取subject_XX中的XX
        subject_id = filename.split('_')[1]
        subject_ids.append(subject_id)

    subject_ids = sorted(list(set(subject_ids)))
    print(f"找到 {len(subject_ids)} 个被试: {subject_ids}")

    # 加载每个被试的数据
    for sub_id in subject_ids:
        # 构建文件名
        happy_file = os.path.join(data_dir, f"subject_{sub_id}_happy_car_2D.npy")
        sad_file = os.path.join(data_dir, f"subject_{sub_id}_sad_car_2D.npy")

        if os.path.exists(happy_file) and os.path.exists(sad_file):
            # 加载数据
            happy_data = np.load(happy_file)  # [N_happy, C, T]
            sad_data = np.load(sad_file)  # [N_sad, C, T]

            data_dict[sub_id] = (happy_data, sad_data)
            print(f"被试 {sub_id}: happy={happy_data.shape}, sad={sad_data.shape}")
        else:
            print(f"警告: 被试 {sub_id} 的数据文件不完整")

    return data_dict


def prepare_cross_subject_data(data_dict, test_subject_id):
    """
    准备跨被试训练测试数据

    参数:
        data_dict: 所有被试的数据字典
        test_subject_id: 测试被试ID

    返回:
        X_train, y_train, X_test, y_test
    """
    train_happy_list = []
    train_sad_list = []
    test_happy = None
    test_sad = None

    for sub_id, (happy_data, sad_data) in data_dict.items():
        if sub_id == test_subject_id:
            # 测试被试
            test_happy = happy_data
            test_sad = sad_data
        else:
            # 训练被试
            train_happy_list.append(happy_data)
            train_sad_list.append(sad_data)

    # 检查测试被试是否存在
    if test_happy is None or test_sad is None:
        raise ValueError(f"测试被试 {test_subject_id} 不存在于数据中")

    # 合并训练数据
    if train_happy_list and train_sad_list:
        X_train_happy = np.concatenate(train_happy_list, axis=0)
        X_train_sad = np.concatenate(train_sad_list, axis=0)
    else:
        # 如果训练数据为空（只有一个被试的情况）
        X_train_happy = np.array([]).reshape(0, *test_happy.shape[1:])
        X_train_sad = np.array([]).reshape(0, *test_sad.shape[1:])

    # 构建训练集
    X_train = np.concatenate([X_train_sad, X_train_happy], axis=0)
    y_train = np.concatenate([
        np.zeros(len(X_train_sad), dtype=np.int64),  # 0: sad
        np.ones(len(X_train_happy), dtype=np.int64)  # 1: happy
    ], axis=0)

    # 构建测试集
    X_test = np.concatenate([test_sad, test_happy], axis=0)
    y_test = np.concatenate([
        np.zeros(len(test_sad), dtype=np.int64),  # 0: sad
        np.ones(len(test_happy), dtype=np.int64)  # 1: happy
    ], axis=0)

    # 应用EA预处理（可选）
    # if len(X_train) > 0:
    #     X_train = EA(X_train)
    # X_test = EA(X_test)

    print(f"训练集: {X_train.shape}, 标签: {y_train.shape}, Sad: {np.sum(y_train == 0)}, Happy: {np.sum(y_train == 1)}")
    print(f"测试集: {X_test.shape}, 标签: {y_test.shape}, Sad: {np.sum(y_test == 0)}, Happy: {np.sum(y_test == 1)}")

    return X_train, y_train, X_test, y_test


def split_train_val_data(X_train, y_train, val_ratio=0.1):
    """
    将训练数据划分为训练集和验证集
    """
    n_total = len(X_train)

    if n_total == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 生成随机索引
    indices = np.arange(n_total)
    np.random.seed(seed)
    np.random.shuffle(indices)

    # 计算验证集大小
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    # 划分训练集和验证集
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    X_train_split = X_train[train_indices]
    y_train_split = y_train[train_indices]
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    print(f"训练子集: {len(X_train_split)} 样本")
    print(f"验证集: {len(X_val)} 样本")

    return X_train_split, y_train_split, X_val, y_val


def train_svm_model(X_train, y_train, X_val, y_val, param_grid=None):
    """
    训练SVM模型
    """
    if len(X_train) == 0 or len(X_val) == 0:
        print("训练或验证数据为空，跳过模型训练")
        return None, None, 0.0

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1],
            'kernel': ['rbf']
        }

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
        verbose=0
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

    return best_model, scaler, val_acc


def cross_subject_de_svm():
    """
    跨被试的DE+SVM训练与测试
    使用留一被试交叉验证
    """
    print("=" * 60)
    print("开始跨被试DE+SVM训练与测试")
    print("=" * 60)

    # 1. 加载所有被试的数据
    print("\n加载所有被试数据...")
    data_dict = load_all_subjects_data()

    if not data_dict:
        print("错误: 没有找到任何被试数据")
        return None

    subject_ids = list(data_dict.keys())
    print(f"总共 {len(subject_ids)} 个被试: {subject_ids}")

    # 存储每个被试的结果
    all_results = {}

    # 2. 对每个被试进行留一法测试
    for i, test_subject in enumerate(subject_ids):
        print(f"\n{'=' * 60}")
        print(f"测试被试 {test_subject} ({i + 1}/{len(subject_ids)})")
        print(f"{'=' * 60}")

        # 准备训练和测试数据
        X_train, y_train, X_test, y_test = prepare_cross_subject_data(
            data_dict, test_subject
        )

        if len(X_train) == 0:
            print(f"警告: 被试 {test_subject} 没有训练数据，跳过")
            continue

        # 3. 提取差分熵特征
        print("\n提取训练集DE特征...")
        X_train_de = extract_de_features(X_train)
        print(f"训练集DE特征形状: {X_train_de.shape}")

        print("\n提取测试集DE特征...")
        X_test_de = extract_de_features(X_test)
        print(f"测试集DE特征形状: {X_test_de.shape}")

        # 4. 划分训练集和验证集
        X_train_split, y_train_split, X_val, y_val = split_train_val_data(
            X_train_de, y_train
        )

        if len(X_train_split) == 0 or len(X_val) == 0:
            print("训练或验证数据为空，跳过当前被试")
            continue

        # 5. 训练SVM模型
        print("\n训练SVM模型...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
            'kernel': ['rbf']
        }

        svm_model, scaler, val_acc = train_svm_model(
            X_train_split, y_train_split,
            X_val, y_val,
            param_grid=param_grid
        )

        if svm_model is None:
            print("模型训练失败，跳过当前被试")
            continue

        # 6. 在测试集上评估
        print("\n在测试集上评估...")
        X_test_scaled = scaler.transform(X_test_de)
        y_test_pred = svm_model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_test_pred)

        print(f"\n测试被试 {test_subject} 结果:")
        print(f"训练集大小: {len(X_train)} 样本")
        print(f"测试集大小: {len(X_test)} 样本")
        print(f"验证集准确率: {val_acc:.4f}")
        print(f"测试集准确率: {test_acc:.4f}")
        print("\n测试集分类报告:")
        print(classification_report(y_test, y_test_pred))
        print("测试集混淆矩阵:")
        print(confusion_matrix(y_test, y_test_pred))

        # 存储结果
        all_results[test_subject] = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'val_acc': val_acc,
            'test_acc': test_acc,
            'model': svm_model,
            'scaler': scaler,
            'X_test': X_test,
            'y_test': y_test,
            'X_test_de': X_test_de,
            'y_pred': y_test_pred
        }

        # 保存每个被试的模型
        model_dir = "cross_subject_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"svm_model_subject_{test_subject}.pkl")
        scaler_path = os.path.join(model_dir, f"scaler_subject_{test_subject}.pkl")

        joblib.dump(svm_model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"模型已保存到: {model_path}")

    # 7. 汇总所有被试的结果
    if all_results:
        print_summary(all_results)

    return all_results


def print_summary(all_results):
    """
    打印跨被试测试结果汇总
    """
    print("\n" + "=" * 60)
    print("跨被试测试结果汇总")
    print("=" * 60)

    test_accs = []
    val_accs = []
    train_sizes = []
    test_sizes = []

    for sub_id, results in all_results.items():
        test_acc = results['test_acc']
        val_acc = results['val_acc']
        train_size = results['train_size']
        test_size = results['test_size']

        test_accs.append(test_acc)
        val_accs.append(val_acc)
        train_sizes.append(train_size)
        test_sizes.append(test_size)

        print(f"被试 {sub_id}: 训练样本={train_size}, 测试样本={test_size}, "
              f"验证准确率={val_acc:.4f}, 测试准确率={test_acc:.4f}")

    if test_accs:
        print(f"\n平均测试准确率: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        print(f"平均验证准确率: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
        print(f"总训练样本数: {sum(train_sizes)}")
        print(f"总测试样本数: {sum(test_sizes)}")
        print(f"被试数量: {len(all_results)}")

        # 可视化准确率分布
        plot_cross_subject_results(all_results)


def plot_cross_subject_results(all_results):
    """
    可视化跨被试测试结果
    """
    subject_ids = list(all_results.keys())
    test_accs = [all_results[sub_id]['test_acc'] for sub_id in subject_ids]
    val_accs = [all_results[sub_id]['val_acc'] for sub_id in subject_ids]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 各被试测试准确率
    x_pos = np.arange(len(subject_ids))
    axes[0, 0].bar(x_pos, test_accs, alpha=0.7)
    axes[0, 0].axhline(y=np.mean(test_accs), color='r', linestyle='--',
                       label=f'平均准确率: {np.mean(test_accs):.4f}')
    axes[0, 0].set_xlabel('被试ID')
    axes[0, 0].set_ylabel('测试准确率')
    axes[0, 0].set_title('各被试测试准确率')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(subject_ids, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 验证集 vs 测试集准确率
    axes[0, 1].scatter(val_accs, test_accs, alpha=0.6, s=100)
    for i, sub_id in enumerate(subject_ids):
        axes[0, 1].annotate(sub_id, (val_accs[i], test_accs[i]), fontsize=9)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[0, 1].set_xlabel('验证集准确率')
    axes[0, 1].set_ylabel('测试集准确率')
    axes[0, 1].set_title('验证集 vs 测试集准确率')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 训练样本数 vs 测试准确率
    train_sizes = [all_results[sub_id]['train_size'] for sub_id in subject_ids]
    axes[1, 0].scatter(train_sizes, test_accs, alpha=0.6, s=100)
    for i, sub_id in enumerate(subject_ids):
        axes[1, 0].annotate(sub_id, (train_sizes[i], test_accs[i]), fontsize=9)
    axes[1, 0].set_xlabel('训练样本数')
    axes[1, 0].set_ylabel('测试准确率')
    axes[1, 0].set_title('训练样本数 vs 测试准确率')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 准确率分布直方图
    axes[1, 1].hist(test_accs, bins=10, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=np.mean(test_accs), color='r', linestyle='--',
                       label=f'均值: {np.mean(test_accs):.4f}')
    axes[1, 1].set_xlabel('测试准确率')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('测试准确率分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('跨被试测试结果分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cross_subject_results.png', dpi=100, bbox_inches='tight')
    plt.show()

    # 保存结果到文件
    save_results_to_file(all_results)


def save_results_to_file(all_results):
    """保存结果到文件"""
    with open('cross_subject_results.txt', 'w') as f:
        f.write("跨被试测试结果汇总\n")
        f.write("=" * 50 + "\n")

        for sub_id, results in all_results.items():
            f.write(f"\n被试 {sub_id}:\n")
            f.write(f"  训练样本数: {results['train_size']}\n")
            f.write(f"  测试样本数: {results['test_size']}\n")
            f.write(f"  验证准确率: {results['val_acc']:.4f}\n")
            f.write(f"  测试准确率: {results['test_acc']:.4f}\n")

        test_accs = [results['test_acc'] for results in all_results.values()]
        val_accs = [results['val_acc'] for results in all_results.values()]

        f.write(f"\n{'=' * 50}\n")
        f.write(f"平均测试准确率: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}\n")
        f.write(f"平均验证准确率: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}\n")
        f.write(f"总被试数: {len(all_results)}\n")

    print("结果已保存到 cross_subject_results.txt")


def load_and_predict_cross_subject(model_dir, test_subject_id, new_data):
    """
    加载训练好的跨被试模型进行预测

    参数:
        model_dir: 模型保存目录
        test_subject_id: 测试被试ID
        new_data: 新数据, shape (n_samples, n_channels, n_times)
    """
    model_path = os.path.join(model_dir, f"svm_model_subject_{test_subject_id}.pkl")
    scaler_path = os.path.join(model_dir, f"scaler_subject_{test_subject_id}.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"错误: 找不到被试 {test_subject_id} 的模型文件")
        return None, None

    # 加载模型
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
    # 运行跨被试DE+SVM
    results = cross_subject_de_svm()

    if results:
        print("\n跨被试测试完成!")
    else:
        print("\n跨被试测试失败!")
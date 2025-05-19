import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import seaborn as sns
import pandas as pd
import argparse
import logging
import time
from tqdm import tqdm
import itertools
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib为非交互式后端
import matplotlib

matplotlib.use('Agg')

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eeg_classification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EEG_Classification")

# 情绪标签映射
EMOTION_LABELS = {
    0: "Sad",
    1: "Disgust",
    2: "Fear",
    3: "Anger",
    4: "Surprise",
    5: "Happy",
    6: "Neutral"
}

# 每种情绪的VideoID映射
EMOTION_VIDEO_IDS = {
    "Sad": [4, 7, 14, 17, 22, 29, 32, 39, 62, 69, 72, 79],
    "Disgust": [3, 8, 13, 18, 43, 48, 53, 58, 61, 70, 71, 80],
    "Fear": [23, 28, 33, 38, 44, 47, 54, 57, 63, 68, 73, 78],
    "Anger": [5, 6, 15, 16, 21, 30, 31, 40, 45, 46, 55, 56],
    "Surprise": [25, 26, 35, 36, 42, 49, 52, 59, 64, 67, 74, 77],
    "Happy": [1, 10, 11, 20, 41, 50, 51, 60, 65, 66, 75, 76],
    "Neutral": [2, 9, 12, 19, 24, 27, 34, 37]
}

# 将情绪名称转为数字标签
EMOTION_NAME_TO_LABEL = {
    "Sad": 0,
    "Disgust": 1,
    "Fear": 2,
    "Anger": 3,
    "Surprise": 4,
    "Happy": 5,
    "Neutral": 6
}


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='脑电情绪识别模型训练与评估')
    parser.add_argument('--feature_dir', type=str, default='/data/coding/EEG_features',
                        help='特征数据目录')
    parser.add_argument('--output_dir', type=str, default='/data/coding/EEG_results',
                        help='结果保存目录')
    parser.add_argument('--feature_type', type=str, default='de',
                        choices=['de', 'de_lds', 'psd', 'psd_lds', 'faa', 'faa_lds'],
                        help='使用的特征类型')
    parser.add_argument('--classifier', type=str, default='both',
                        choices=['svm', 'rf', 'both'],
                        help='分类器类型: svm (支持向量机), rf (随机森林), both (两者都用)')
    parser.add_argument('--cv_mode', type=str, default='kfold',
                        choices=['loso', 'kfold'],
                        help='交叉验证模式: loso (留一交叉验证), kfold (K折交叉验证)')
    parser.add_argument('--k_folds', type=int, default=3,
                        help='K折交叉验证的折数 (仅在cv_mode=kfold时使用)')
    parser.add_argument('--scenario', type=str, default='both',
                        choices=['intra', 'cross', 'both'],
                        help='实验场景: intra (被试内), cross (跨被试), both (两者都进行)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()


def load_subject_data(feature_dir, subject_id, feature_type):
    """
    加载单个被试的特征数据

    参数:
        feature_dir: 特征目录
        subject_id: 被试ID
        feature_type: 特征类型 ('de', 'de_lds', 'psd', 'psd_lds', 'faa', 'faa_lds')

    返回:
        features_dict: 包含特征和标签的字典
    """
    file_path = os.path.join(feature_dir, f"{subject_id}.mat")

    if not os.path.exists(file_path):
        logger.error(f"被试 {subject_id} 的特征文件不存在: {file_path}")
        return None

    try:
        # 加载MAT文件
        mat_data = sio.loadmat(file_path)

        # 提取所有视频的特征和标签
        features_dict = {}

        # 查找所有对应特征类型的键
        feature_keys = [k for k in mat_data.keys() if k.startswith(f"{feature_type}_video_")]
        label_keys = [k for k in mat_data.keys() if k.startswith("label_video_")]

        # 确保键是按数字顺序排序的
        feature_keys.sort(key=lambda x: int(x.split('_')[-1]))
        label_keys.sort(key=lambda x: int(x.split('_')[-1]))

        # 加载特征和标签
        for feature_key, label_key in zip(feature_keys, label_keys):
            video_id = int(feature_key.split('_')[-1])

            # 获取特征和标签
            feature = mat_data[feature_key]
            label = mat_data[label_key].item()

            # 如果是Neutral情绪，并且我们要进行模型训练，就跳过（因为Neutral不参与划分）
            # 但我们仍然加载它以便于全面分析
            features_dict[video_id] = {
                'feature': feature,
                'label': label,
                'emotion': EMOTION_LABELS[label]
            }

        logger.info(f"成功加载被试 {subject_id} 的 {len(features_dict)} 个视频特征")
        return features_dict

    except Exception as e:
        logger.error(f"加载被试 {subject_id} 的特征时出错: {e}")
        return None


def prepare_intra_subject_data(feature_dir, feature_type, exclude_neutral=True):
    """
    准备被试内实验的数据

    参数:
        feature_dir: 特征目录
        feature_type: 特征类型
        exclude_neutral: 是否排除Neutral情绪

    返回:
        subjects_data: 包含所有被试数据的字典
    """
    # 找出特征目录中的所有被试
    subject_files = [f for f in os.listdir(feature_dir) if f.endswith('.mat') and f[0].isdigit()]
    subject_ids = sorted([int(f.split('.')[0]) for f in subject_files])

    # 加载所有被试数据
    subjects_data = {}
    for subject_id in subject_ids:
        subject_data = load_subject_data(feature_dir, subject_id, feature_type)
        if subject_data:
            subjects_data[subject_id] = subject_data

    logger.info(f"共加载 {len(subjects_data)} 个被试的数据")

    # 统计各情绪类别的样本数
    emotion_counts = {emotion: 0 for emotion in EMOTION_LABELS.values()}
    for subject_id, data in subjects_data.items():
        for video_id, video_data in data.items():
            emotion = video_data['emotion']
            emotion_counts[emotion] += 1

    logger.info("各情绪类别样本数:")
    for emotion, count in emotion_counts.items():
        logger.info(f"  {emotion}: {count}")

    # 确定训练和测试样本
    # 根据要求，每种情绪选择3/4作为训练集，1/4作为测试集
    # Neutral不参与划分
    train_video_ids = []
    test_video_ids = []

    for emotion, video_ids in EMOTION_VIDEO_IDS.items():
        if emotion == "Neutral" and exclude_neutral:
            continue

        # 确保视频ID是排序的，以便分割的一致性
        sorted_video_ids = sorted(video_ids)

        # 划分训练集和测试集 (3:1)
        n_train = int(len(sorted_video_ids) * 0.75)
        train_video_ids.extend(sorted_video_ids[:n_train])
        test_video_ids.extend(sorted_video_ids[n_train:])

    logger.info(f"训练集视频ID: {sorted(train_video_ids)}")
    logger.info(f"测试集视频ID: {sorted(test_video_ids)}")

    return subjects_data, train_video_ids, test_video_ids


def prepare_cross_subject_data(feature_dir, feature_type, random_seed=42):
    """
    准备跨被试实验的数据

    参数:
        feature_dir: 特征目录
        feature_type: 特征类型
        random_seed: 随机种子

    返回:
        combined_data: 所有被试合并后的数据
        train_subject_ids: 训练集被试ID
        test_subject_ids: 测试集被试ID
    """
    # 找出特征目录中的所有被试
    subject_files = [f for f in os.listdir(feature_dir) if f.endswith('.mat') and f[0].isdigit()]
    subject_ids = sorted([int(f.split('.')[0]) for f in subject_files])

    # 加载所有被试数据
    all_subject_data = {}
    for subject_id in subject_ids:
        subject_data = load_subject_data(feature_dir, subject_id, feature_type)
        if subject_data:
            all_subject_data[subject_id] = subject_data

    logger.info(f"共加载 {len(all_subject_data)} 个被试的数据")

    # 按被试划分训练集和测试集 (3:1)
    np.random.seed(random_seed)
    train_subject_ids, test_subject_ids = train_test_split(
        subject_ids,
        test_size=0.25,
        random_state=random_seed
    )

    logger.info(f"训练集被试ID: {sorted(train_subject_ids)}")
    logger.info(f"测试集被试ID: {sorted(test_subject_ids)}")

    return all_subject_data, train_subject_ids, test_subject_ids


def combine_features_for_subject(subject_data, exclude_neutral=True):
    """
    合并单个被试的所有视频特征

    参数:
        subject_data: 被试数据
        exclude_neutral: 是否排除Neutral情绪

    返回:
        X: 特征矩阵
        y: 标签数组
        video_ids: 视频ID数组
    """
    X_list = []
    y_list = []
    video_ids_list = []

    for video_id, video_data in subject_data.items():
        # 如果是Neutral情绪且需要排除，则跳过
        if exclude_neutral and video_data['emotion'] == "Neutral":
            continue

        feature = video_data['feature']
        label = video_data['label']

        # 如果特征是多维的，将其展平为特征向量
        if len(feature.shape) > 2:
            # 计算窗口的平均特征
            feature_mean = np.mean(feature, axis=0)
            X_list.append(feature_mean.flatten())
        else:
            X_list.append(feature.flatten())

        y_list.append(label)
        video_ids_list.append(video_id)

    if X_list:
        X = np.vstack(X_list)
        y = np.array(y_list)
        video_ids = np.array(video_ids_list)
        return X, y, video_ids
    else:
        return None, None, None


def intra_subject_classification(subjects_data, train_video_ids, test_video_ids,
                                 classifier_type='svm', cv_mode='loso', k_folds=3, random_seed=42):
    """
    进行被试内分类

    参数:
        subjects_data: 所有被试的特征数据
        train_video_ids: 训练集视频ID
        test_video_ids: 测试集视频ID
        classifier_type: 分类器类型 ('svm' 或 'rf')
        cv_mode: 交叉验证模式 ('loso' 或 'kfold')
        k_folds: K折交叉验证的折数
        random_seed: 随机种子

    返回:
        results: 分类结果字典
    """
    results = {
        'subject_accuracies': {},
        'average_accuracy': 0,
        'std_accuracy': 0,
        'confusion_matrices': {},
        'classifier_type': classifier_type,
        'cv_mode': cv_mode,
        'k_folds': k_folds
    }

    # 设置分类器
    if classifier_type == 'svm':
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_seed))
        ])
    elif classifier_type == 'rf':
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_seed))
        ])
    else:
        logger.error(f"不支持的分类器类型: {classifier_type}")
        return None

    # 对每个被试进行分类
    all_accuracies = []
    all_y_true = []
    all_y_pred = []

    for subject_id, subject_data in tqdm(subjects_data.items(), desc="Processing subjects"):
        # 准备训练和测试数据
        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []

        for video_id, video_data in subject_data.items():
            feature = video_data['feature']
            label = video_data['label']

            # 确定该视频是训练集还是测试集
            if video_id in train_video_ids:
                # 如果特征是多维的，取平均
                if len(feature.shape) > 2:
                    feature_mean = np.mean(feature, axis=0)
                    X_train_list.append(feature_mean.flatten())
                else:
                    X_train_list.append(feature.flatten())
                y_train_list.append(label)
            elif video_id in test_video_ids:
                # 如果特征是多维的，取平均
                if len(feature.shape) > 2:
                    feature_mean = np.mean(feature, axis=0)
                    X_test_list.append(feature_mean.flatten())
                else:
                    X_test_list.append(feature.flatten())
                y_test_list.append(label)

        # 检查是否有足够的数据
        if len(X_train_list) == 0 or len(X_test_list) == 0:
            logger.warning(f"被试 {subject_id} 的训练集或测试集为空")
            continue

        X_train = np.vstack(X_train_list)
        y_train = np.array(y_train_list)
        X_test = np.vstack(X_test_list)
        y_test = np.array(y_test_list)

        # 进行交叉验证
        cv_accuracies = []

        if cv_mode == 'loso':
            # 留一交叉验证
            loo = LeaveOneOut()
            X_cv = X_train.copy()
            y_cv = y_train.copy()

            for train_idx, val_idx in loo.split(X_cv):
                X_cv_train, X_cv_val = X_cv[train_idx], X_cv[val_idx]
                y_cv_train, y_cv_val = y_cv[train_idx], y_cv[val_idx]

                # 训练模型
                clf.fit(X_cv_train, y_cv_train)

                # 预测并计算准确率
                y_cv_pred = clf.predict(X_cv_val)
                accuracy = accuracy_score(y_cv_val, y_cv_pred)
                cv_accuracies.append(accuracy)

        elif cv_mode == 'kfold':
            # K折交叉验证
            # 确保每一折中的样本都包含所有类别
            unique_labels = np.unique(y_train)
            folds = min(k_folds, len(X_train) // len(unique_labels))
            if folds < 1:
                folds = 1

            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

                # 训练模型
                clf.fit(X_cv_train, y_cv_train)

                # 预测并计算准确率
                y_cv_pred = clf.predict(X_cv_val)
                accuracy = accuracy_score(y_cv_val, y_cv_pred)
                cv_accuracies.append(accuracy)

        # 计算交叉验证的平均准确率
        cv_accuracy = np.mean(cv_accuracies) if cv_accuracies else 0
        logger.info(f"被试 {subject_id} 的交叉验证准确率: {cv_accuracy:.4f}")

        # 在完整训练集上训练模型
        clf.fit(X_train, y_train)

        # 在测试集上评估
        y_pred = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        # 保存结果
        results['subject_accuracies'][subject_id] = {
            'cv_accuracy': cv_accuracy,
            'test_accuracy': test_accuracy
        }

        all_accuracies.append(test_accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        logger.info(f"被试 {subject_id} 的测试准确率: {test_accuracy:.4f}")

    # 计算总体准确率
    results['average_accuracy'] = np.mean(all_accuracies)
    results['std_accuracy'] = np.std(all_accuracies)

    # 计算总体混淆矩阵
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    confusion_mat = confusion_matrix(all_y_true, all_y_pred)
    results['confusion_matrices']['overall'] = confusion_mat

    logger.info(f"被试内分类平均准确率: {results['average_accuracy']:.4f} ± {results['std_accuracy']:.4f}")

    return results


def cross_subject_classification(all_subject_data, train_subject_ids, test_subject_ids,
                                 classifier_type='svm', cv_mode='loso', k_folds=3, random_seed=42):
    """
    进行跨被试分类

    参数:
        all_subject_data: 所有被试的特征数据
        train_subject_ids: 训练集被试ID
        test_subject_ids: 测试集被试ID
        classifier_type: 分类器类型 ('svm' 或 'rf')
        cv_mode: 交叉验证模式 ('loso' 或 'kfold')
        k_folds: K折交叉验证的折数
        random_seed: 随机种子

    返回:
        results: 分类结果字典
    """
    results = {
        'fold_accuracies': {},
        'average_accuracy': 0,
        'std_accuracy': 0,
        'confusion_matrices': {},
        'classifier_type': classifier_type,
        'cv_mode': cv_mode,
        'k_folds': k_folds
    }

    # 设置分类器
    if classifier_type == 'svm':
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_seed))
        ])
    elif classifier_type == 'rf':
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_seed))
        ])
    else:
        logger.error(f"不支持的分类器类型: {classifier_type}")
        return None

    # 准备训练和测试数据
    X_train_list = []
    y_train_list = []
    subject_train_list = []

    X_test_list = []
    y_test_list = []
    subject_test_list = []

    # 收集训练集数据
    for subject_id in train_subject_ids:
        if subject_id not in all_subject_data:
            continue

        subject_data = all_subject_data[subject_id]
        X, y, _ = combine_features_for_subject(subject_data)

        if X is not None and y is not None:
            X_train_list.append(X)
            y_train_list.append(y)
            subject_train_list.extend([subject_id] * len(y))

    # 收集测试集数据
    for subject_id in test_subject_ids:
        if subject_id not in all_subject_data:
            continue

        subject_data = all_subject_data[subject_id]
        X, y, _ = combine_features_for_subject(subject_data)

        if X is not None and y is not None:
            X_test_list.append(X)
            y_test_list.append(y)
            subject_test_list.extend([subject_id] * len(y))

    # 合并数据
    X_train = np.vstack(X_train_list) if X_train_list else None
    y_train = np.concatenate(y_train_list) if y_train_list else None
    subject_train = np.array(subject_train_list)

    X_test = np.vstack(X_test_list) if X_test_list else None
    y_test = np.concatenate(y_test_list) if y_test_list else None

    if X_train is None or y_train is None or X_test is None or y_test is None:
        logger.error("训练集或测试集为空")
        return None

    logger.info(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    # 进行交叉验证
    cv_accuracies = []
    fold_predictions = {}

    if cv_mode == 'loso':
        # 留一交叉验证 (留一被试)
        unique_subjects = np.unique(subject_train)

        for i, left_out_subject in enumerate(unique_subjects):
            # 找出不是留出被试的索引
            train_idx = np.where(subject_train != left_out_subject)[0]
            val_idx = np.where(subject_train == left_out_subject)[0]

            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

            # 训练模型
            clf.fit(X_cv_train, y_cv_train)

            # 预测并计算准确率
            y_cv_pred = clf.predict(X_cv_val)
            accuracy = accuracy_score(y_cv_val, y_cv_pred)
            cv_accuracies.append(accuracy)

            fold_predictions[i] = {
                'subject_id': left_out_subject,
                'true_labels': y_cv_val,
                'predicted_labels': y_cv_pred,
                'accuracy': accuracy
            }

            logger.info(f"留一被试 {left_out_subject} 的交叉验证准确率: {accuracy:.4f}")

    elif cv_mode == 'kfold':
        # K折交叉验证
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

        for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

            # 训练模型
            clf.fit(X_cv_train, y_cv_train)

            # 预测并计算准确率
            y_cv_pred = clf.predict(X_cv_val)
            accuracy = accuracy_score(y_cv_val, y_cv_pred)
            cv_accuracies.append(accuracy)

            fold_predictions[i] = {
                'true_labels': y_cv_val,
                'predicted_labels': y_cv_pred,
                'accuracy': accuracy
            }

            logger.info(f"第 {i + 1} 折交叉验证准确率: {accuracy:.4f}")

    # 计算交叉验证的平均准确率
    cv_accuracy = np.mean(cv_accuracies)
    cv_std = np.std(cv_accuracies)

    # 在完整训练集上训练模型
    clf.fit(X_train, y_train)

    # 在测试集上评估
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # 保存结果
    results['fold_accuracies'] = {i: fold_predictions[i]['accuracy'] for i in fold_predictions}
    results['cv_accuracy'] = cv_accuracy
    results['cv_std'] = cv_std
    results['test_accuracy'] = test_accuracy

    # 计算总体混淆矩阵
    confusion_mat = confusion_matrix(y_test, y_pred)
    results['confusion_matrices']['overall'] = confusion_mat

    logger.info(f"交叉验证平均准确率: {cv_accuracy:.4f} ± {cv_std:.4f}")
    logger.info(f"测试集准确率: {test_accuracy:.4f}")

    return results


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', normalize=True, save_path=None):
    """
    绘制混淆矩阵

    参数:
        cm: 混淆矩阵
        class_names: 类别名称
        title: 图表标题
        normalize: 是否归一化
        save_path: 保存路径
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_subject_accuracies(accuracies, title='Subject Accuracies', save_path=None):
    """
    绘制被试准确率柱状图

    参数:
        accuracies: 准确率字典
        title: 图表标题
        save_path: 保存路径
    """
    subjects = list(accuracies.keys())
    values = [accuracies[s]['test_accuracy'] for s in subjects]

    plt.figure(figsize=(12, 6))
    plt.bar(subjects, values, color='skyblue')
    plt.axhline(y=np.mean(values), color='r', linestyle='-', label=f'Mean: {np.mean(values):.2f}')
    plt.title(title, fontsize=16)
    plt.xlabel('Subject ID', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim([0, 1.0])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_results_to_csv(results, output_path):
    """
    将结果保存为CSV文件

    参数:
        results: 结果字典
        output_path: 保存路径
    """
    # 如果是被试内结果
    if 'subject_accuracies' in results:
        df = pd.DataFrame({
            'subject_id': list(results['subject_accuracies'].keys()),
            'cv_accuracy': [results['subject_accuracies'][s]['cv_accuracy'] for s in results['subject_accuracies']],
            'test_accuracy': [results['subject_accuracies'][s]['test_accuracy'] for s in results['subject_accuracies']]
        })
        df.to_csv(output_path, index=False)

    # 如果是跨被试结果
    elif 'fold_accuracies' in results:
        df = pd.DataFrame({
            'fold': list(results['fold_accuracies'].keys()),
            'accuracy': list(results['fold_accuracies'].values())
        })
        df.loc[len(df)] = ['cv_average', results['cv_accuracy']]
        df.loc[len(df)] = ['cv_std', results['cv_std']]
        df.loc[len(df)] = ['test_accuracy', results['test_accuracy']]
        df.to_csv(output_path, index=False)


def main():
    """主函数"""
    start_time = time.time()

    # 解析命令行参数
    args = parse_arguments()

    # 设置随机种子
    np.random.seed(args.random_seed)

    logger.info("=== SEED VII 脑电情绪识别分类 ===")
    logger.info(f"特征目录: {args.feature_dir}")
    logger.info(f"结果目录: {args.output_dir}")
    logger.info(f"特征类型: {args.feature_type}")
    logger.info(f"分类器: {args.classifier}")
    logger.info(f"交叉验证: {args.cv_mode}")
    logger.info(f"K折数: {args.k_folds}")
    logger.info(f"实验场景: {args.scenario}")

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"创建输出目录: {args.output_dir}")

    # 根据指定的分类器类型和场景进行实验
    classifiers = []
    if args.classifier == 'both':
        classifiers = ['svm', 'rf']
    else:
        classifiers = [args.classifier]

    scenarios = []
    if args.scenario == 'both':
        scenarios = ['intra', 'cross']
    else:
        scenarios = [args.scenario]

    # 进行实验
    for classifier_type in classifiers:
        for scenario in scenarios:
            scenario_dir = os.path.join(args.output_dir, scenario)
            if not os.path.exists(scenario_dir):
                os.makedirs(scenario_dir)

            # 被试内实验
            if scenario == 'intra':
                logger.info(f"\n=== 被试内实验 ({classifier_type}) ===")

                # 准备数据
                subjects_data, train_video_ids, test_video_ids = prepare_intra_subject_data(
                    args.feature_dir, args.feature_type)

                # 进行分类
                results = intra_subject_classification(
                    subjects_data, train_video_ids, test_video_ids,
                    classifier_type=classifier_type,
                    cv_mode=args.cv_mode,
                    k_folds=args.k_folds,
                    random_seed=args.random_seed
                )

                # 保存结果
                results_dir = os.path.join(scenario_dir, classifier_type)
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                # 保存主要结果
                save_results_to_csv(
                    results,
                    os.path.join(results_dir, f"{args.feature_type}_{args.cv_mode}_results.csv")
                )

                # 绘制和保存混淆矩阵
                plot_confusion_matrix(
                    results['confusion_matrices']['overall'],
                    [EMOTION_LABELS[i] for i in range(6)],  # 排除Neutral
                    title=f'Intra-Subject Confusion Matrix ({classifier_type.upper()}, {args.feature_type})',
                    save_path=os.path.join(results_dir, f"{args.feature_type}_{args.cv_mode}_confusion_matrix.png")
                )

                # 绘制和保存被试准确率
                plot_subject_accuracies(
                    results['subject_accuracies'],
                    title=f'Intra-Subject Accuracies ({classifier_type.upper()}, {args.feature_type})',
                    save_path=os.path.join(results_dir, f"{args.feature_type}_{args.cv_mode}_subject_accuracies.png")
                )

                logger.info(f"被试内分类结果已保存到: {results_dir}")

            # 跨被试实验
            elif scenario == 'cross':
                logger.info(f"\n=== 跨被试实验 ({classifier_type}) ===")

                # 准备数据
                all_subject_data, train_subject_ids, test_subject_ids = prepare_cross_subject_data(
                    args.feature_dir, args.feature_type, args.random_seed)

                # 进行分类
                results = cross_subject_classification(
                    all_subject_data, train_subject_ids, test_subject_ids,
                    classifier_type=classifier_type,
                    cv_mode=args.cv_mode,
                    k_folds=args.k_folds,
                    random_seed=args.random_seed
                )

                # 保存结果
                results_dir = os.path.join(scenario_dir, classifier_type)
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                # 保存主要结果
                save_results_to_csv(
                    results,
                    os.path.join(results_dir, f"{args.feature_type}_{args.cv_mode}_results.csv")
                )

                # 绘制和保存混淆矩阵
                plot_confusion_matrix(
                    results['confusion_matrices']['overall'],
                    [EMOTION_LABELS[i] for i in range(6)],  # 排除Neutral
                    title=f'Cross-Subject Confusion Matrix ({classifier_type.upper()}, {args.feature_type})',
                    save_path=os.path.join(results_dir, f"{args.feature_type}_{args.cv_mode}_confusion_matrix.png")
                )

                logger.info(f"跨被试分类结果已保存到: {results_dir}")

    # 计算总处理时间
    elapsed_time = time.time() - start_time
    logger.info(f"处理完成！总用时: {elapsed_time:.2f} 秒")
    logger.info(f"结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
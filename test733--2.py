import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold, LeaveOneOut
import argparse
import logging
from tqdm import tqdm
import random
import concurrent.futures
from collections import defaultdict

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_split.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Dataset_Split")



# 情绪标签字典,由于不连续的数字标签会导致pytorch会报错，这里将标签按效价重新排序.
# 原始标签是0:Disgust, 1:Fear, 2:Sad, 3:Neutral, 4:Happy, 5:Anger, 6:Surprise;
# 修改后标签是0:Sad, 1:Disgust, 2:Fear, 3:Anger, 4:Surprise, 5:Happy, 6:Neutral.
EMOTION_LABELS = {
    0: "Sad",
    1: "Disgust",
    2: "Fear",
    3: "Anger",
    4: "Surprise",
    5: "Happy",
    6: "Neutral"
}

# 每种情绪的视频ID
EMOTION_VIDEO_IDS = {
    "Sad": [4, 7, 14, 17, 22, 29, 32, 39, 62, 69, 72, 79],  # 标签0
    "Disgust": [3, 8, 13, 18, 43, 48, 53, 58, 61, 70, 71, 80],  # 标签1
    "Fear": [23, 28, 33, 38, 44, 47, 54, 57, 63, 68, 73, 78],  # 标签2
    "Anger": [5, 6, 15, 16, 21, 30, 31, 40, 45, 46, 55, 56],  # 标签3
    "Surprise": [25, 26, 35, 36, 42, 49, 52, 59, 64, 67, 74, 77],  # 标签4
    "Happy": [1, 10, 11, 20, 41, 50, 51, 60, 65, 66, 75, 76],  # 标签5
    "Neutral": [2, 9, 12, 19, 24, 27, 34, 37]  # 标签6
}

# 情绪名称到标签的映射
EMOTION_TO_LABEL = {
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
    parser = argparse.ArgumentParser(description='SEED VII数据集划分')
    parser.add_argument('--input_dir', type=str, default='/data/coding/EEG_features',
                        help='特征文件目录')
    parser.add_argument('--output_dir', type=str, default='/data/coding/dataset_split',
                        help='划分后数据集保存目录')
    parser.add_argument('--split_type', type=str, choices=['intra', 'cross'], default='cross',
                        help='划分类型: intra (被试内) 或 cross (跨被试)')
    parser.add_argument('--cv_type', type=str, choices=['loo', 'kfold'], default='loo',
                        help='交叉验证类型: loo (留一法) 或 kfold (k折)')
    parser.add_argument('--k_folds', type=int, default=3,
                        help='使用k折交叉验证时的折数')
    parser.add_argument('--feature_types', type=str, default='de,psd,faa,de_lds,psd_lds,faa_lds',
                        help='要处理的特征类型，逗号分隔')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--threads', type=int, default=1,
                        help='用于并行处理的线程数量')
    return parser.parse_args()


def load_subject_data(subject_id, input_dir, feature_types):
    """
    加载单个被试的特征数据

    参数:
        subject_id: 被试ID
        input_dir: 特征文件目录
        feature_types: 要加载的特征类型列表

    返回:
        包含特征数据的字典
    """
    try:
        file_path = os.path.join(input_dir, f"{subject_id}.mat")
        if not os.path.exists(file_path):
            logger.error(f"被试 {subject_id} 的特征文件不存在: {file_path}")
            return None

        # 加载.mat文件
        mat_data = sio.loadmat(file_path)

        # 提取视频ID和对应的标签
        video_ids = []
        labels = {}

        for key in mat_data.keys():
            if key.startswith('label_video_'):
                video_id = int(key.split('_')[-1])
                label = int(mat_data[key][0][0])
                video_ids.append(video_id)
                labels[video_id] = label

        # 按情绪组织数据
        emotion_data = defaultdict(list)

        for video_id in video_ids:
            label = labels[video_id]

            # 跳过中性情绪 (Neutral)
            if label == 6:  # Neutral
                continue

            # 收集该视频的所有特征
            video_features = {}
            for feature_type in feature_types:
                feature_key = f"{feature_type}_video_{video_id}"
                if feature_key in mat_data:
                    video_features[feature_type] = mat_data[feature_key]
                else:
                    logger.warning(f"被试 {subject_id} 的特征 {feature_key} 不存在")

            # 将视频特征添加到对应情绪的列表中
            emotion_name = EMOTION_LABELS[label]
            emotion_data[emotion_name].append({
                'video_id': video_id,
                'features': video_features,
                'label': label
            })

        # 检查每种情绪的样本数量是否符合预期
        for emotion, samples in emotion_data.items():
            if emotion != "Neutral" and len(samples) != 12:
                logger.warning(f"被试 {subject_id} 的 {emotion} 情绪样本数量为 {len(samples)}，预期为12")

        return {
            'subject_id': subject_id,
            'emotion_data': emotion_data
        }

    except Exception as e:
        logger.error(f"加载被试 {subject_id} 数据时出错: {e}", exc_info=True)
        return None


def intra_subject_split(subject_data, feature_types, cv_type='loo', k_folds=3, random_seed=42):
    """
    对单个被试进行被试内划分

    参数:
        subject_data: 被试数据字典
        feature_types: 要处理的特征类型列表
        cv_type: 交叉验证类型 ('loo'或'kfold')
        k_folds: 使用k折交叉验证时的折数
        random_seed: 随机种子

    返回:
        划分后的数据集字典
    """
    subject_id = subject_data['subject_id']
    emotion_data = subject_data['emotion_data']

    # 设置随机种子
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 情绪列表（不包括Neutral）
    emotions = [e for e in emotion_data.keys() if e != "Neutral"]

    # 为每种情绪分配训练和测试样本
    train_videos = {}
    test_videos = {}

    for emotion in emotions:
        # 获取该情绪的所有样本
        samples = emotion_data[emotion]

        # 随机打乱样本顺序
        shuffled_samples = random.sample(samples, len(samples))

        # 按3:1比例分配训练和测试样本
        n_train = int(len(shuffled_samples) * 0.75)
        train_videos[emotion] = shuffled_samples[:n_train]
        test_videos[emotion] = shuffled_samples[n_train:]

        logger.info(
            f"被试 {subject_id} 的 {emotion} 情绪: {n_train} 个训练样本, {len(shuffled_samples) - n_train} 个测试样本")

    # 创建训练集交叉验证折
    cv_folds = []

    if cv_type == 'loo':
        # 留一交叉验证（对于被试内，每个情绪类别留一样本）
        # 计算每种情绪的训练样本数量
        n_train_per_emotion = min(len(train_videos[emotion]) for emotion in emotions)

        # 创建n_train_per_emotion个折，每折包含6种情绪各一个样本
        for fold_idx in range(n_train_per_emotion):
            fold_train = []
            fold_val = []

            for emotion in emotions:
                # 获取该情绪的训练样本
                emotion_train = train_videos[emotion]

                # 当前折的验证样本
                val_sample = emotion_train[fold_idx]
                fold_val.append(val_sample)

                # 当前折的训练样本（排除验证样本）
                train_samples = [s for s in emotion_train if s['video_id'] != val_sample['video_id']]
                fold_train.extend(train_samples)

            cv_folds.append({
                'train': fold_train,
                'val': fold_val
            })

    else:  # k折交叉验证
        # 为每种情绪创建k折
        emotion_folds = {}
        for emotion in emotions:
            emotion_train = train_videos[emotion]

            # 创建k折分割器
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

            # 为当前情绪生成k折索引
            emotion_folds[emotion] = []
            for train_idx, val_idx in kf.split(emotion_train):
                emotion_folds[emotion].append({
                    'train': [emotion_train[i] for i in train_idx],
                    'val': [emotion_train[i] for i in val_idx]
                })

        # 合并各情绪的折，创建k个总的交叉验证折
        for fold_idx in range(k_folds):
            fold_train = []
            fold_val = []

            for emotion in emotions:
                fold_train.extend(emotion_folds[emotion][fold_idx]['train'])
                fold_val.extend(emotion_folds[emotion][fold_idx]['val'])

            cv_folds.append({
                'train': fold_train,
                'val': fold_val
            })

    # 转换测试集格式
    test_samples = []
    for emotion in emotions:
        test_samples.extend(test_videos[emotion])

    return {
        'subject_id': subject_id,
        'cv_folds': cv_folds,
        'test': test_samples,
        'feature_types': feature_types
    }


def cross_subject_split(all_subjects_data, feature_types, cv_type='loo', k_folds=3, random_seed=42):
    """
    进行跨被试划分

    参数:
        all_subjects_data: 所有被试数据的列表
        feature_types: 要处理的特征类型列表
        cv_type: 交叉验证类型 ('loo'或'kfold')
        k_folds: 使用k折交叉验证时的折数
        random_seed: 随机种子

    返回:
        划分后的数据集字典
    """
    # 设置随机种子
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 过滤掉None值
    valid_subjects = [s for s in all_subjects_data if s is not None]

    if len(valid_subjects) == 0:
        logger.error("没有有效的被试数据可用于跨被试划分")
        return None

    # 随机打乱被试顺序
    shuffled_subjects = random.sample(valid_subjects, len(valid_subjects))

    # 按3:1比例分配训练和测试被试
    n_train = int(len(shuffled_subjects) * 0.75)
    train_subjects = shuffled_subjects[:n_train]
    test_subjects = shuffled_subjects[n_train:]

    logger.info(f"跨被试划分: {len(train_subjects)} 个训练被试, {len(test_subjects)} 个测试被试")

    # 创建交叉验证折
    cv_folds = []

    if cv_type == 'loo':
        # 留一交叉验证（每次留出一个被试）
        loo = LeaveOneOut()

        for train_idx, val_idx in loo.split(train_subjects):
            fold_train_subjects = [train_subjects[i] for i in train_idx]
            fold_val_subjects = [train_subjects[i] for i in val_idx]

            cv_folds.append({
                'train': fold_train_subjects,
                'val': fold_val_subjects
            })

    else:  # k折交叉验证
        # 创建k折分割器
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

        for train_idx, val_idx in kf.split(train_subjects):
            fold_train_subjects = [train_subjects[i] for i in train_idx]
            fold_val_subjects = [train_subjects[i] for i in val_idx]

            cv_folds.append({
                'train': fold_train_subjects,
                'val': fold_val_subjects
            })

    return {
        'cv_folds': cv_folds,
        'test': test_subjects,
        'feature_types': feature_types
    }


def process_intra_subject_data(split_data, output_dir, cv_type):
    """
    处理被试内划分的数据，解决不同长度样本的问题，并保存为.mat文件

    参数:
        split_data: 划分后的数据集字典
        output_dir: 输出目录
    """
    subject_id = split_data['subject_id']
    cv_folds = split_data['cv_folds']
    test_samples = split_data['test']
    feature_types = split_data['feature_types']

    # 创建被试输出目录
    subject_dir = os.path.join(output_dir, f"intra_{cv_type}", f"subject_{subject_id}")
    os.makedirs(subject_dir, exist_ok=True)

    # 处理每个交叉验证折
    for fold_idx, fold in enumerate(cv_folds):
        train_samples = fold['train']
        val_samples = fold['val']

        # 处理训练集
        process_and_save_samples(train_samples, subject_dir, f"train_fold_{fold_idx + 1}", feature_types)

        # 处理验证集
        process_and_save_samples(val_samples, subject_dir, f"val_fold_{fold_idx + 1}", feature_types)

    # 处理测试集
    process_and_save_samples(test_samples, subject_dir, "test", feature_types)

    logger.info(f"被试 {subject_id} 的被试内划分数据已保存到 {subject_dir}")


def process_cross_subject_data(split_data, output_dir, cv_type):
    """
    处理跨被试划分的数据，并保存为.mat文件

    参数:
        split_data: 划分后的数据集字典
        output_dir: 输出目录
    """
    cv_folds = split_data['cv_folds']
    test_subjects = split_data['test']
    feature_types = split_data['feature_types']

    # 创建跨被试输出目录
    cross_dir = os.path.join(output_dir, f"cross_{cv_type}")
    os.makedirs(cross_dir, exist_ok=True)

    # 处理每个交叉验证折
    for fold_idx, fold in enumerate(cv_folds):
        train_subjects = fold['train']
        val_subjects = fold['val']

        # 提取所有训练被试的样本
        all_train_samples = []
        for subject in train_subjects:
            for emotion, samples in subject['emotion_data'].items():
                if emotion != "Neutral":  # 排除Neutral情绪
                    all_train_samples.extend(samples)

        # 提取所有验证被试的样本
        all_val_samples = []
        for subject in val_subjects:
            for emotion, samples in subject['emotion_data'].items():
                if emotion != "Neutral":  # 排除Neutral情绪
                    all_val_samples.extend(samples)

        # 处理训练集
        process_and_save_samples(all_train_samples, cross_dir, f"train_fold_{fold_idx + 1}", feature_types)

        # 处理验证集
        process_and_save_samples(all_val_samples, cross_dir, f"val_fold_{fold_idx + 1}", feature_types)

    # 提取所有测试被试的样本
    all_test_samples = []
    for subject in test_subjects:
        for emotion, samples in subject['emotion_data'].items():
            if emotion != "Neutral":  # 排除Neutral情绪
                all_test_samples.extend(samples)

    # 处理测试集
    process_and_save_samples(all_test_samples, cross_dir, "test", feature_types)

    logger.info(f"跨被试划分数据已保存到 {cross_dir}")


def process_and_save_samples(samples, output_dir, prefix, feature_types):
    """
    处理样本并保存为.mat文件，解决不同长度样本的问题

    参数:
        samples: 样本列表
        output_dir: 输出目录
        prefix: 文件名前缀
        feature_types: 要处理的特征类型列表
    """
    if not samples:
        logger.warning(f"没有样本可处理: {output_dir}/{prefix}")
        return

    # 按情绪分组样本
    emotion_samples = defaultdict(list)
    for sample in samples:
        label = sample['label']
        emotion = EMOTION_LABELS[label]
        emotion_samples[emotion].append(sample)

    # 为每种特征类型创建字典
    feature_matrices = {}
    feature_labels = {}

    # 处理每种情绪
    for emotion, emotion_sample_list in emotion_samples.items():
        # 处理每种特征类型
        for feature_type in feature_types:
            # 为当前情绪和特征类型创建样本点列表
            emotion_sample_points = []
            emotion_sample_labels = []

            # 处理每个样本
            for sample in emotion_sample_list:
                # 获取特征数据
                if feature_type not in sample['features']:
                    continue

                feature_data = sample['features'][feature_type]

                # 对于每个采样点，添加特征和标签
                n_time_points = feature_data.shape[0]  # 时间维度

                for t in range(n_time_points):
                    # 添加单个时间点的特征
                    emotion_sample_points.append(feature_data[t])

                    # 添加对应的标签
                    emotion_sample_labels.append(sample['label'])

            # 将当前情绪的样本点添加到总特征矩阵
            if emotion_sample_points:
                # 转换为numpy数组
                emotion_points_array = np.array(emotion_sample_points)
                emotion_labels_array = np.array(emotion_sample_labels)

                # 添加到特征字典
                if feature_type not in feature_matrices:
                    feature_matrices[feature_type] = emotion_points_array
                    feature_labels[feature_type] = emotion_labels_array
                else:
                    feature_matrices[feature_type] = np.concatenate([
                        feature_matrices[feature_type], emotion_points_array
                    ], axis=0)
                    feature_labels[feature_type] = np.concatenate([
                        feature_labels[feature_type], emotion_labels_array
                    ], axis=0)

    # 创建最终数据字典
    output_dict = {}

    # 添加特征和标签
    for feature_type in feature_types:
        if feature_type in feature_matrices:
            output_dict[f"{feature_type}_features"] = feature_matrices[feature_type]
            output_dict[f"{feature_type}_labels"] = feature_labels[feature_type]

    # 保存为.mat文件
    output_file = os.path.join(output_dir, f"{prefix}.mat")
    sio.savemat(output_file, output_dict)

    # 记录特征矩阵大小
    for feature_type in feature_types:
        if feature_type in feature_matrices:
            logger.info(f"  {prefix}_{feature_type} 特征矩阵: {feature_matrices[feature_type].shape}")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 设置随机种子
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # 解析特征类型
    feature_types = args.feature_types.split(',')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有被试ID
    subject_files = [f for f in os.listdir(args.input_dir) if f.endswith('.mat') and f[0].isdigit()]
    subject_ids = [f.split('.')[0] for f in subject_files]

    logger.info("=== SEED VII 数据集划分 ===")
    logger.info(f"特征目录: {args.input_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"划分类型: {args.split_type}")
    logger.info(f"交叉验证类型: {args.cv_type}")
    logger.info(f"K折数: {args.k_folds}")
    logger.info(f"特征类型: {feature_types}")
    logger.info(f"随机种子: {args.random_seed}")
    logger.info(f"找到 {len(subject_ids)} 个被试")

    # 并行加载所有被试数据
    logger.info("正在加载被试数据...")
    all_subjects_data = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_subject = {
            executor.submit(load_subject_data, subject_id, args.input_dir, feature_types): subject_id
            for subject_id in subject_ids
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_subject), total=len(subject_ids),
                           desc="加载被试数据"):
            subject_id = future_to_subject[future]
            try:
                subject_data = future.result()
                if subject_data:
                    all_subjects_data.append(subject_data)
            except Exception as e:
                logger.error(f"处理被试 {subject_id} 数据时出错: {e}")

    logger.info(f"成功加载 {len(all_subjects_data)} 个被试的数据")

    # 根据划分类型执行不同的划分
    if args.split_type == 'intra':
        # 被试内划分
        logger.info("执行被试内划分...")

        # 为每个被试执行划分
        for subject_data in tqdm(all_subjects_data, desc="被试内划分"):
            split_data = intra_subject_split(
                subject_data,
                feature_types,
                cv_type=args.cv_type,
                k_folds=args.k_folds,
                random_seed=args.random_seed
            )

            # 处理并保存数据
            process_intra_subject_data(split_data, args.output_dir, args.cv_type)

        logger.info("被试内划分完成")

    else:
        # 跨被试划分
        logger.info("执行跨被试划分...")

        split_data = cross_subject_split(
            all_subjects_data,
            feature_types,
            cv_type=args.cv_type,
            k_folds=args.k_folds,
            random_seed=args.random_seed
        )

        if split_data:
            # 处理并保存数据
            process_cross_subject_data(split_data, args.output_dir, args.cv_type)
            logger.info("跨被试划分完成")
        else:
            logger.error("跨被试划分失败")

    logger.info(f"所有数据集划分已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()

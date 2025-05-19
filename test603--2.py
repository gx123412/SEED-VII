import mne
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import os
import math
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
import time
import concurrent.futures
import argparse
import logging
from tqdm import tqdm

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eeg_feature_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EEG_Feature_Extraction")

# 定义频段
FREQ_BANDS = {
    'delta': (1, 4),  # delta: 1-4Hz
    'theta': (4, 8),  # theta: 4-8Hz
    'alpha': (8, 13),  # alpha: 8-13Hz
    'beta': (13, 30),  # beta: 13-30Hz
    'gamma': (30, 50)  # gamma: 30-50Hz
}

# 前额叶电极对（用于计算Alpha不对称性）
FRONTAL_PAIRS = [
    ('F3', 'F4'),  # 左右前额叶中央位置
    ('F7', 'F8'),  # 左右前额叶外侧位置
    ('FP1', 'FP2'),  # 左右额极位置
    ('AF3', 'AF4')  # 左右前额区位置
]

# SEED VII 电极位置
ELECTRODES = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
              'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
              'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
              'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
              'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1',
              'O1', 'OZ', 'O2', 'CB2']


# SEED VII 情绪标签,由于不连续的数字标签会导致pytorch会报错，这里将标签按效价重新排序.
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

# 每个被试的80个视频样本标签
TRIAL_LABELS = [5, 6, 1, 0, 3, 3, 0, 1, 6, 5, 5, 6, 1, 0, 3, 3, 0, 1, 6, 5,
                3, 0, 2, 6, 4, 4, 6, 2, 0, 3, 3, 0, 2, 6, 4, 4, 6, 2, 0, 3,
                5, 4, 1, 2, 3, 3, 2, 1, 4, 5, 5, 4, 1, 2, 3, 3, 2, 1, 4, 5,
                1, 0, 2, 4, 5, 5, 4, 2, 0, 1, 1, 0, 2, 4, 5, 5, 4, 2, 0, 1]



def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='脑电情绪识别特征提取')
    parser.add_argument('--threads', type=int, default=1,
                        help='用于并行处理的线程数量')
    parser.add_argument('--input_dir', type=str, default='/data/coding/EEG_preprocessed',
                        help='预处理EEG数据目录')
    parser.add_argument('--label_file', type=str, default='/data/coding/label/labels.csv',
                        help='标签文件路径')
    parser.add_argument('--output_dir', type=str, default='/data/coding/EEG_features',
                        help='特征保存目录')
    parser.add_argument('--window_size', type=float, default=1.0,
                        help='滑动窗口大小(秒)')
    parser.add_argument('--window_step', type=float, default=0.5,
                        help='滑动窗口步长(秒)')
    parser.add_argument('--smooth_window', type=int, default=5,
                        help='LDS平滑窗口大小')
    parser.add_argument('--subjects', type=str, default='all',
                        help='要处理的被试ID，使用逗号分隔，如"1,2,3"或"all"表示所有被试')
    return parser.parse_args()


def calculate_psd(data, sf, window_size, window_step):
    """
    计算功率谱密度 (PSD)

    参数:
        data: 输入EEG数据，shape为 [channels, samples]
        sf: 采样率
        window_size: 窗口大小(秒)
        window_step: 窗口步长(秒)

    返回:
        各频段的PSD特征，shape为 [windows, channels, bands]
    """
    n_channels = data.shape[0]
    window_samples = int(window_size * sf)
    step_samples = int(window_step * sf)

    # 计算窗口数量
    n_windows = max(1, int(np.floor((data.shape[1] - window_samples) / step_samples + 1)))

    # 创建结果数组
    n_bands = len(FREQ_BANDS)
    psd_features = np.zeros((n_windows, n_channels, n_bands))

    # 对每个窗口计算PSD
    for win_idx in range(n_windows):
        start_sample = win_idx * step_samples
        end_sample = start_sample + window_samples

        if end_sample <= data.shape[1]:
            # 对每个通道计算PSD
            for ch_idx in range(n_channels):
                win_data = data[ch_idx, start_sample:end_sample]

                # 使用Welch方法计算PSD
                freq, psd = signal.welch(win_data, sf, nperseg=min(256, window_samples),
                                         noverlap=min(128, window_samples // 2))

                # 提取各频段的平均PSD
                for band_idx, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
                    # 找到频带范围内的索引
                    idx_band = np.logical_and(freq >= fmin, freq <= fmax)
                    # 计算该频带的平均PSD
                    psd_features[win_idx, ch_idx, band_idx] = np.mean(psd[idx_band])

    return psd_features


def calculate_de(data, sf, window_size, window_step):
    """
    计算差分熵 (DE)

    参数:
        data: 输入EEG数据，shape为 [channels, samples]
        sf: 采样率
        window_size: 窗口大小(秒)
        window_step: 窗口步长(秒)

    返回:
        各频段的DE特征，shape为 [windows, channels, bands]
    """
    n_channels = data.shape[0]
    window_samples = int(window_size * sf)
    step_samples = int(window_step * sf)

    # 计算窗口数量
    n_windows = max(1, int(np.floor((data.shape[1] - window_samples) / step_samples + 1)))

    # 创建结果数组
    n_bands = len(FREQ_BANDS)
    de_features = np.zeros((n_windows, n_channels, n_bands))

    # 定义汉宁窗
    hanning_window = np.hanning(window_samples)

    # 对每个窗口计算DE
    for win_idx in range(n_windows):
        start_sample = win_idx * step_samples
        end_sample = start_sample + window_samples

        if end_sample <= data.shape[1]:
            # 对每个通道计算DE
            for ch_idx in range(n_channels):
                win_data = data[ch_idx, start_sample:end_sample]
                win_data = win_data * hanning_window  # 应用窗函数

                # 计算频谱
                fft_data = np.fft.rfft(win_data)

                # 计算功率谱
                Pxx = np.abs(fft_data) ** 2

                # 计算频率分辨率
                freq_res = sf / window_samples
                freq = np.fft.rfftfreq(window_samples, d=1. / sf)

                # 计算各频段的DE
                for band_idx, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
                    # 获取频带对应的索引
                    idx_band = np.logical_and(freq >= fmin, freq <= fmax)

                    # 提取该频带的功率谱
                    band_power = Pxx[idx_band]

                    # 确保有足够的数据点
                    if len(band_power) > 0:
                        # 计算该频带的差分熵
                        # DE = 0.5 * log(2 * pi * e * variance)
                        # 对于功率谱，方差等于平均功率
                        variance = np.mean(band_power)
                        if variance <= 0:
                            # 处理零或负方差
                            de_features[win_idx, ch_idx, band_idx] = 0
                        else:
                            de_features[win_idx, ch_idx, band_idx] = 0.5 * np.log(2 * np.pi * np.e * variance)
                    else:
                        de_features[win_idx, ch_idx, band_idx] = 0

    return de_features


def calculate_frontal_alpha_asymmetry(data, sf, window_size, window_step, electrode_names):
    """
    计算前额叶Alpha不对称性特征

    参数:
        data: 输入EEG数据，shape为 [channels, samples]
        sf: 采样率
        window_size: 窗口大小(秒)
        window_step: 窗口步长(秒)
        electrode_names: 电极名称列表

    返回:
        前额叶Alpha不对称性特征，shape为 [windows, n_pairs]
    """
    window_samples = int(window_size * sf)
    step_samples = int(window_step * sf)

    # 计算窗口数量
    n_windows = max(1, int(np.floor((data.shape[1] - window_samples) / step_samples + 1)))

    # 获取Alpha频段
    alpha_band = FREQ_BANDS['alpha']

    # 找出所有有效的前额叶电极对索引
    valid_pairs = []
    for left_el, right_el in FRONTAL_PAIRS:
        if left_el in electrode_names and right_el in electrode_names:
            left_idx = electrode_names.index(left_el)
            right_idx = electrode_names.index(right_el)
            valid_pairs.append((left_idx, right_idx))

    # 如果没有有效的电极对，返回空数组
    if not valid_pairs:
        return np.zeros((n_windows, 0))

    # 创建结果数组
    faa_features = np.zeros((n_windows, len(valid_pairs)))

    # 对每个窗口计算不对称性
    for win_idx in range(n_windows):
        start_sample = win_idx * step_samples
        end_sample = start_sample + window_samples

        if end_sample <= data.shape[1]:
            # 对每对电极计算Alpha不对称性
            for pair_idx, (left_idx, right_idx) in enumerate(valid_pairs):
                # 提取左右电极的数据
                left_data = data[left_idx, start_sample:end_sample]
                right_data = data[right_idx, start_sample:end_sample]

                # 计算左右电极的Alpha功率
                freq_left, psd_left = signal.welch(left_data, sf, nperseg=min(256, window_samples))
                freq_right, psd_right = signal.welch(right_data, sf, nperseg=min(256, window_samples))

                # 找到Alpha频段的索引
                idx_alpha = np.logical_and(freq_left >= alpha_band[0], freq_left <= alpha_band[1])

                # 计算Alpha频段的平均功率
                alpha_power_left = np.mean(psd_left[idx_alpha])
                alpha_power_right = np.mean(psd_right[idx_alpha])

                # 计算Alpha不对称性（右 - 左）或 ln(右/左)
                # 防止对数运算中的零或负值
                if alpha_power_left > 0 and alpha_power_right > 0:
                    # 计算ln(右/左)形式的不对称性
                    faa_features[win_idx, pair_idx] = np.log(alpha_power_right / alpha_power_left)
                else:
                    faa_features[win_idx, pair_idx] = 0

    return faa_features


def local_linear_dynamic_system_smoothing(features, window_size=5):
    """
    使用局部线性动态系统算法对特征进行平滑处理

    参数:
        features: 输入特征，shape为 [windows, ...]
        window_size: 平滑窗口大小

    返回:
        平滑后的特征
    """
    if window_size <= 1 or features.shape[0] <= 1:
        return features

    n_windows = features.shape[0]
    feature_shape = features.shape[1:]
    features_flat = features.reshape(n_windows, -1)
    smoothed_features = np.zeros_like(features_flat)

    # 对每个窗口应用LDS平滑
    for i in range(n_windows):
        # 确定当前窗口的平滑范围
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n_windows, i + window_size // 2 + 1)

        # 提取局部窗口特征
        local_features = features_flat[start_idx:end_idx]

        # 如果只有一个样本，直接赋值
        if local_features.shape[0] == 1:
            smoothed_features[i] = local_features[0]
            continue

        # 构建时间向量（归一化到[0,1]区间）
        time_vec = np.linspace(0, 1, end_idx - start_idx)

        # 对每个特征维度拟合线性模型
        for j in range(local_features.shape[1]):
            # 提取特征值
            feature_vals = local_features[:, j]

            # 线性拟合
            if np.std(feature_vals) > 1e-10:  # 避免常数特征
                coeffs = np.polyfit(time_vec, feature_vals, 1)
                # 在当前时间点的预测值
                current_time_idx = i - start_idx
                current_time = time_vec[current_time_idx] if current_time_idx < len(time_vec) else time_vec[-1]
                smoothed_features[i, j] = np.polyval(coeffs, current_time)
            else:
                smoothed_features[i, j] = feature_vals[0]

    # 恢复原始形状
    smoothed_features = smoothed_features.reshape(n_windows, *feature_shape)

    return smoothed_features


def extract_features_for_trial(trial_data, sf, window_size, window_step, electrode_names, smooth_window):
    """
    为单个试验提取特征

    参数:
        trial_data: 试验数据，shape为 [channels, samples]
        sf: 采样率
        window_size: 窗口大小(秒)
        window_step: 窗口步长(秒)
        electrode_names: 电极名称列表
        smooth_window: LDS平滑窗口大小

    返回:
        试验特征字典
    """
    # 计算DE特征
    de_features = calculate_de(trial_data, sf, window_size, window_step)

    # 计算PSD特征
    psd_features = calculate_psd(trial_data, sf, window_size, window_step)

    # 计算额叶Alpha不对称性特征
    faa_features = calculate_frontal_alpha_asymmetry(trial_data, sf, window_size, window_step, electrode_names)

    # 应用LDS平滑
    de_smoothed = local_linear_dynamic_system_smoothing(de_features, smooth_window)
    psd_smoothed = local_linear_dynamic_system_smoothing(psd_features, smooth_window)
    faa_smoothed = local_linear_dynamic_system_smoothing(faa_features, smooth_window)

    # 返回特征字典
    return {
        'de': de_features,
        'de_lds': de_smoothed,
        'psd': psd_features,
        'psd_lds': psd_smoothed,
        'faa': faa_features,
        'faa_lds': faa_smoothed
    }


def process_subject(subject_id, args):
    """
    处理单个被试的数据

    参数:
        subject_id: 被试ID
        args: 命令行参数
    """
    try:
        start_time = time.time()
        logger.info(f"开始处理被试 {subject_id}")

        # 构建输入文件路径
        input_file = os.path.join(args.input_dir, f"{subject_id}.mat")

        # 检查文件是否存在
        if not os.path.exists(input_file):
            logger.error(f"被试 {subject_id} 的数据文件不存在: {input_file}")
            return False

        # 加载被试数据
        mat_data = sio.loadmat(input_file)
        logger.info(f"成功加载被试 {subject_id} 的数据")

        # 创建结果字典
        features_dict = {}

        # 对每个视频试验进行处理
        n_trials = len([key for key in mat_data.keys() if key.startswith('video_')])
        logger.info(f"被试 {subject_id} 共有 {n_trials} 个视频试验")

        for trial_idx in range(1, n_trials + 1):
            trial_key = f"video_{trial_idx}"

            if trial_key not in mat_data:
                logger.warning(f"被试 {subject_id} 的试验 {trial_key} 不存在")
                continue

            # 获取试验数据
            trial_data = mat_data[trial_key]

            # 提取特征
            logger.info(f"正在为被试 {subject_id} 的试验 {trial_key} 提取特征")
            features = extract_features_for_trial(
                trial_data,
                sf=200,  # SEED VII数据集的采样率为200Hz
                window_size=args.window_size,
                window_step=args.window_step,
                electrode_names=ELECTRODES,
                smooth_window=args.smooth_window
            )

            # 保存特征
            label = TRIAL_LABELS[trial_idx - 1]

            # 将特征添加到字典中
            for feature_type in ['de', 'de_lds', 'psd', 'psd_lds', 'faa', 'faa_lds']:
                feature_key = f"{feature_type}_{trial_key}"
                features_dict[feature_key] = features[feature_type]

            # 保存标签
            features_dict[f"label_{trial_key}"] = label

            logger.info(f"完成被试 {subject_id} 的试验 {trial_key} 特征提取，情绪标签: {EMOTION_LABELS[label]}")

        # 创建输出目录
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存特征到.mat文件
        output_file = os.path.join(output_dir, f"{subject_id}.mat")
        sio.savemat(output_file, features_dict)

        elapsed_time = time.time() - start_time
        logger.info(f"成功处理被试 {subject_id}，用时 {elapsed_time:.2f} 秒")

        return True

    except Exception as e:
        logger.error(f"处理被试 {subject_id} 时出错: {e}", exc_info=True)
        return False


def main():
    """主函数"""
    start_time = time.time()

    # 解析命令行参数
    args = parse_arguments()

    logger.info("=== SEED VII 脑电情绪识别特征提取 ===")
    logger.info(f"输入数据目录: {args.input_dir}")
    logger.info(f"标签文件: {args.label_file}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"窗口大小: {args.window_size} 秒")
    logger.info(f"窗口步长: {args.window_step} 秒")
    logger.info(f"LDS平滑窗口: {args.smooth_window}")
    logger.info(f"使用 {args.threads} 个线程")

    # 确定要处理的被试
    if args.subjects.lower() == 'all':
        # 获取所有.mat文件
        subject_files = [f for f in os.listdir(args.input_dir) if f.endswith('.mat') and f[0].isdigit()]
        subject_ids = [f.split('.')[0] for f in subject_files]
    else:
        subject_ids = args.subjects.split(',')

    logger.info(f"将处理 {len(subject_ids)} 个被试: {', '.join(subject_ids)}")

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info(f"已创建输出目录: {args.output_dir}")

    # 并行处理被试
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        # 创建任务列表
        future_to_subject = {
            executor.submit(process_subject, subject_id, args): subject_id
            for subject_id in subject_ids
        }

        # 处理完成的任务，带进度条
        successful = 0
        for future in tqdm(concurrent.futures.as_completed(future_to_subject),
                           total=len(subject_ids),
                           desc="处理被试"):
            subject_id = future_to_subject[future]
            try:
                if future.result():
                    successful += 1
            except Exception as e:
                logger.error(f"处理被试 {subject_id} 的线程出错: {e}")

    # 计算总处理时间
    elapsed_time = time.time() - start_time
    logger.info(f"特征提取完成！成功处理 {successful}/{len(subject_ids)} 个被试")
    logger.info(f"总处理时间: {elapsed_time:.2f} 秒")
    logger.info(f"特征已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
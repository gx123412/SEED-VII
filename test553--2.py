import mne
import pandas as pd
import numpy as np
import os
from mne.io import read_raw_cnt
from glob import glob
import scipy.io as sio
import datetime
import math
import concurrent.futures
import threading
import argparse
from tqdm import tqdm
import logging
import time

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eeg_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EEG_Preprocessing")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='脑电情绪识别预处理')
    parser.add_argument('--threads', type=int, default=1,
                        help='用于并行处理的线程数量')
    parser.add_argument('--raw_dir', type=str, default='/data/coding/EEG_raw',
                        help='原始EEG数据目录')
    parser.add_argument('--trigger_dir', type=str, default='/data/coding/save_info',
                        help='触发器信息目录')
    parser.add_argument('--output_dir', type=str, default='/data/coding/EEG_preprocessed',
                        help='预处理后数据保存目录')
    parser.add_argument('--ica_components', type=int, default=15,
                        help='ICA组件数量')
    return parser.parse_args()


def preprocess_file(cnt_file, trigger_info_dir, output_base_dir, lock, ica_components=15):
    """预处理单个.cnt文件，应用ICA去伪迹并分段"""
    start_time = time.time()

    try:
        # 提取文件名信息
        base_name = os.path.basename(cnt_file)
        file_parts = base_name.split('_')
        subject_id = file_parts[0]
        date = file_parts[1]
        session_id = file_parts[2].split('.')[0]

        logger.info(f"处理文件: {base_name}")
        logger.info(f"被试ID: {subject_id}, 日期: {date}, 会话ID: {session_id}")

        # 加载.cnt文件
        raw = read_raw_cnt(cnt_file, preload=True)
        logger.info(f"原始数据信息: 采样率: {raw.info['sfreq']} Hz, 通道数: {len(raw.ch_names)}")
        logger.info(f"原始通道列表: {raw.ch_names}")

        # 定义EOG通道用于ICA
        eog_channels = ['HEO', 'VEO']

        # 检查EOG通道是否存在
        eog_exists = all(ch in raw.ch_names for ch in eog_channels)

        if not eog_exists:
            logger.warning(f"数据中未找到EOG通道 {eog_channels}，无法应用ICA。")
            return None

        # 应用带通滤波器
        raw.filter(l_freq=0.1, h_freq=70.0)
        logger.info("已应用0.1-70 Hz带通滤波")

        # 应用陷波滤波器
        raw.notch_filter(freqs=[50])
        logger.info("已应用50 Hz陷波滤波")

        # 重采样
        raw.resample(200)
        logger.info(f"已降采样至200 Hz。新采样率: {raw.info['sfreq']} Hz")

        # 应用ICA去除伪迹
        logger.info("正在应用ICA去除EOG伪迹")

        # 设置ICA
        ica = mne.preprocessing.ICA(n_components=ica_components, random_state=42, max_iter='auto')

        # 拟合ICA
        ica.fit(raw)
        logger.info(f"ICA已拟合，使用{ica_components}个组件")

        # 查找并排除EOG伪迹
        for eog_ch in eog_channels:
            indices, _ = ica.find_bads_eog(raw, ch_name=eog_ch)
            ica.exclude.extend(indices)

        logger.info(f"已识别要排除的ICA组件: {ica.exclude}")

        # 应用ICA去除伪迹
        raw_cleaned = raw.copy()
        ica.apply(raw_cleaned)
        logger.info("已应用ICA去除伪迹")

        # 在ICA后排除通道
        excluded_channels = ['M1', 'M2', 'ECG', 'HEO', 'VEO']
        include_channels = [ch for ch in raw_cleaned.ch_names if ch not in excluded_channels]
        raw_cleaned.pick_channels(include_channels)

        logger.info(f"已排除通道: {excluded_channels}")
        logger.info(f"排除后的通道数: {len(raw_cleaned.ch_names)}")
        logger.info(f"保留的通道: {raw_cleaned.ch_names}")

        # 加载触发器信息
        trigger_file = os.path.join(trigger_info_dir, f"{subject_id}_{date}_{session_id}_trigger_info.csv")

        if not os.path.exists(trigger_file):
            logger.warning(f"未找到触发器信息文件 {trigger_file}")
            return None

        logger.info(f"加载触发器信息: {trigger_file}")

        # 加载触发器CSV文件
        trigger_df = pd.read_csv(trigger_file, header=None, names=['trigger_type', 'timestamp'])

        if len(trigger_df) == 0:
            logger.warning("CSV文件为空，没有触发器信息。")
            return None

        # 将时间戳转换为datetime对象
        trigger_df['datetime'] = pd.to_datetime(trigger_df['timestamp'])

        # 计算相对于第一个时间戳的时间差（秒），并向下取整
        start_time_dt = trigger_df['datetime'].iloc[0]
        trigger_df['time_seconds'] = (trigger_df['datetime'] - start_time_dt).dt.total_seconds().apply(math.floor)

        logger.info(f"已处理 {len(trigger_df)} 个触发点")

        # 创建事件列表
        events = []
        for idx, row in trigger_df.iterrows():
            # 获取触发器类型和时间（秒）
            trigger_type = row['trigger_type']
            time_seconds = row['time_seconds']

            # 将时间转换为采样点
            sample = int(time_seconds * raw_cleaned.info['sfreq'])

            # 添加事件
            events.append([sample, 0, trigger_type])

        events = np.array(events)
        logger.info(f"从CSV文件创建了 {len(events)} 个事件")

        # 找到所有触发器1的索引（开始标记）
        start_indices = np.where(events[:, 2] == 1)[0]

        segments_saved = 0
        segment_data = {}

        # 创建被试输出目录
        subject_temp_dir = os.path.join(output_base_dir, f"subject_{subject_id}")
        with lock:
            if not os.path.exists(subject_temp_dir):
                os.makedirs(subject_temp_dir)

        # 处理每对开始/结束触发器
        for start_idx in start_indices:
            # 查找下一个触发器2（结束标记）
            end_indices = np.where((events[:, 2] == 2) & (events[:, 0] > events[start_idx, 0]))[0]

            if len(end_indices) == 0:
                logger.warning(f"未找到索引 {start_idx} 处开始标记的结束标记")
                continue

            # 使用最近的触发器2
            end_idx = end_indices[0]

            # 计算开始和结束时间（秒）
            start_time_sec = events[start_idx, 0] / raw_cleaned.info['sfreq']
            end_time_sec = events[end_idx, 0] / raw_cleaned.info['sfreq']
            duration = end_time_sec - start_time_sec

            # 截取这一段数据
            try:
                segment_raw = raw_cleaned.copy()
                segment_raw.crop(tmin=start_time_sec, tmax=end_time_sec)

                # 获取数据
                data = segment_raw.get_data()

                # 修改：直接保存数据数组，而不是字典
                segment_data[f"video_{segments_saved + 1}"] = data

                # 也保存该会话的单独分段
                segment_info = {
                    f"segment_{segments_saved + 1}": data
                }

                mat_file = os.path.join(subject_temp_dir,
                                        f"{subject_id}_{date}_{session_id}_segment_{segments_saved + 1}.mat")
                sio.savemat(mat_file, segment_info)

                logger.info(f"分段 {segments_saved + 1} 已处理并保存，持续时间: {duration:.2f} 秒")
                segments_saved += 1
            except Exception as e:
                logger.error(f"处理分段时出错: {e}")

        # 计算处理时间
        elapsed_time = time.time() - start_time
        logger.info(f"处理文件 {base_name} 用时 {elapsed_time:.2f} 秒")
        logger.info(f"为被试 {subject_id}, 会话 {session_id} 共处理了 {segments_saved} 个分段")

        # 返回分段数据以便后续合并
        return {
            'subject_id': subject_id,
            'session_id': session_id,
            'segments': segment_data,
            'segments_count': segments_saved
        }

    except Exception as e:
        logger.error(f"处理文件 {cnt_file} 时出错: {e}", exc_info=True)
        return None


def combine_sessions_data(subject_data_list, output_base_dir):
    """将不同会话的数据合并为单个.mat文件"""
    if not subject_data_list:
        logger.warning("没有数据可合并")
        return

    # 按被试分组数据
    subjects = {}
    for data in subject_data_list:
        if data is None:
            continue

        subject_id = data['subject_id']
        if subject_id not in subjects:
            subjects[subject_id] = []

        subjects[subject_id].append(data)

    # 处理每个被试
    for subject_id, sessions_data in subjects.items():
        logger.info(f"合并被试 {subject_id} 的会话数据")

        # 按会话ID排序
        sessions_data.sort(key=lambda x: int(x['session_id']))

        # 初始化合并数据字典
        combined_data = {}
        video_counter = 1

        # 合并所有会话
        for session_data in sessions_data:
            session_id = session_data['session_id']
            logger.info(f"处理会话 {session_id}，包含 {session_data['segments_count']} 个分段")

            # 按键排序分段(video_1, video_2等)
            segment_keys = sorted(session_data['segments'].keys(),
                                  key=lambda x: int(x.split('_')[1]))

            for segment_key in segment_keys:
                # 修改：直接将数据数组赋值给video_键
                combined_data[f"video_{video_counter}"] = session_data['segments'][segment_key]
                video_counter += 1

        # 保存合并数据
        output_file = os.path.join(output_base_dir, f"{subject_id}.mat")
        logger.info(f"保存包含 {len(combined_data)} 个视频的合并数据到 {output_file}")

        # 保存为MAT文件
        sio.savemat(output_file, combined_data)
        logger.info(f"被试 {subject_id} 的数据已成功保存")


def main():
    """主函数，使用多线程和ICA预处理EEG数据"""
    start_time = time.time()

    # 解析命令行参数
    args = parse_arguments()

    # 设置路径
    raw_data_dir = args.raw_dir
    trigger_info_dir = args.trigger_dir
    output_base_dir = args.output_dir

    logger.info("=== 脑电情绪识别预处理 ===")
    logger.info(f"原始数据目录: {raw_data_dir}")
    logger.info(f"触发器信息目录: {trigger_info_dir}")
    logger.info(f"输出目录: {output_base_dir}")
    logger.info(f"使用 {args.threads} 个线程")
    logger.info(f"ICA组件数量: {args.ica_components}")

    # 要排除的通道列表
    excluded_channels = ['M1', 'M2', 'ECG', 'HEO', 'VEO']
    logger.info(f"将排除这些通道: {excluded_channels}")

    # 如果输出基础目录不存在则创建
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        logger.info(f"已创建输出目录: {output_base_dir}")

    # 获取所有.cnt文件
    cnt_files = glob(os.path.join(raw_data_dir, "*.cnt"))
    logger.info(f"找到 {len(cnt_files)} 个.cnt文件")

    if len(cnt_files) == 0:
        logger.error(f"在 {raw_data_dir} 中未找到.cnt文件")
        return

    # 创建线程锁用于目录创建
    lock = threading.Lock()

    # 并行处理文件
    all_data = []
    logger.info(f"开始使用 {args.threads} 个线程进行并行处理")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        # 创建futures字典
        future_to_file = {
            executor.submit(
                preprocess_file,
                file,
                trigger_info_dir,
                output_base_dir,
                lock,
                args.ica_components
            ): file for file in cnt_files
        }

        # 处理完成的任务，带进度条
        for future in tqdm(concurrent.futures.as_completed(future_to_file),
                           total=len(cnt_files),
                           desc="处理文件"):
            file = future_to_file[future]
            try:
                data = future.result()
                if data:
                    all_data.append(data)
                    logger.info(f"成功处理 {file}")
                else:
                    logger.warning(f"{file} 未返回数据")
            except Exception as e:
                logger.error(f"处理 {file} 出错: {e}")

    # 合并所有被试的会话数据
    logger.info("合并所有被试的会话数据")
    combine_sessions_data(all_data, output_base_dir)

    # 计算总处理时间
    elapsed_time = time.time() - start_time
    logger.info(f"总处理时间: {elapsed_time:.2f} 秒")

    logger.info("所有EEG数据处理完成！")
    logger.info(f"数据已按被试ID以MAT格式保存在: {output_base_dir}")
    logger.info("每个被试的数据以video_1到video_N键组织")


if __name__ == "__main__":
    main()
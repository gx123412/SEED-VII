import os
import json
import argparse
import random
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import scipy.io as sio

# 设置matplotlib
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.formatter.use_locale'] = False

# 情绪标签映射 (6分类，排除中性情绪)
EMOTION_LABELS = {
    0: "Sad",
    1: "Disgust",
    2: "Fear",
    3: "Anger",
    4: "Surprise",
    5: "Happy"
}


class MultiFeatureEEGDataset(Dataset):
    """多特征脑电数据集，支持早期特征融合"""

    def __init__(self, mat_file, feature_types=['de', 'de_lds', 'psd', 'psd_lds', 'faa', 'faa_lds'],
                 sequence_length=5):
        """
        初始化数据集

        参数:
            mat_file (str): .mat文件路径
            feature_types (list): 要使用的特征类型列表
            sequence_length (int): 时序序列长度
        """
        # 加载.mat文件
        data = sio.loadmat(mat_file)

        # 初始化特征和标签字典
        features_dict = {}
        labels_dict = {}

        # 加载各类特征
        for feature_type in feature_types:
            feature_key = f'{feature_type}_features'
            label_key = f'{feature_type}_labels'

            if feature_key in data and label_key in data:
                features = data[feature_key]
                labels = data[label_key].squeeze()

                # 保存特征和标签
                features_dict[feature_type] = features
                labels_dict[feature_type] = labels

        if not features_dict:
            raise ValueError(f"在文件 {mat_file} 中未找到任何指定的特征类型")

        # 提取DE和PSD特征 (形状为 [samples, 62, 5])
        de_psd_features = []
        for feature_type in feature_types:
            if feature_type.startswith('de') or feature_type.startswith('psd'):
                if feature_type in features_dict:
                    de_psd_features.append(features_dict[feature_type])

        # 提取FAA特征 (形状为 [samples, 4])
        faa_features = []
        for feature_type in feature_types:
            if feature_type.startswith('faa'):
                if feature_type in features_dict:
                    # 扩展维度 [samples, 4] -> [samples, 4, 1]
                    features = np.expand_dims(features_dict[feature_type], axis=2)
                    faa_features.append(features)

        # 合并特征
        if de_psd_features:
            # 在频段维度上合并DE/PSD特征
            self.de_psd_features = np.concatenate(de_psd_features, axis=2)
            self.has_de_psd = True
        else:
            self.has_de_psd = False

        if faa_features:
            # 在频段维度上合并FAA特征
            self.faa_features = np.concatenate(faa_features, axis=2)
            self.has_faa = True
        else:
            self.has_faa = False

        # 确保至少有一种特征
        if not self.has_de_psd and not self.has_faa:
            raise ValueError("没有找到有效特征")

        # 使用第一个特征类型的标签作为最终标签
        first_feature = next(iter(labels_dict))
        self.labels = labels_dict[first_feature]

        # 检查标签是否为6分类 (0-5)
        unique_labels = np.unique(self.labels)
        if not np.all(np.isin(unique_labels, np.arange(6))):
            print(f"警告: 标签包含非0-5范围的值: {unique_labels}")

        # 转换为PyTorch张量
        if self.has_de_psd:
            self.de_psd_features = torch.FloatTensor(self.de_psd_features)
        if self.has_faa:
            self.faa_features = torch.FloatTensor(self.faa_features)

        self.labels = torch.LongTensor(self.labels)

        # 序列长度
        self.sequence_length = sequence_length

        # 计算有效样本数
        self.num_samples = len(self.labels) - self.sequence_length + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """获取指定索引的序列样本"""
        # 创建一个时间序列窗口
        sequence_indices = range(idx, idx + self.sequence_length)

        if self.has_de_psd and self.has_faa:
            # 获取DE/PSD和FAA序列
            de_psd_sequence = self.de_psd_features[sequence_indices]
            faa_sequence = self.faa_features[sequence_indices]

            # 返回序列特征和标签
            return (de_psd_sequence, faa_sequence), self.labels[idx + self.sequence_length - 1]
        elif self.has_de_psd:
            # 只有DE/PSD特征
            de_psd_sequence = self.de_psd_features[sequence_indices]
            return de_psd_sequence, self.labels[idx + self.sequence_length - 1]
        elif self.has_faa:
            # 只有FAA特征
            faa_sequence = self.faa_features[sequence_indices]
            return faa_sequence, self.labels[idx + self.sequence_length - 1]


class CNNLSTM(nn.Module):
    """CNN-LSTM模型用于脑电情绪识别"""

    def __init__(self, de_psd_channels=62, de_psd_bands=10, faa_channels=4, faa_bands=2,
                 num_classes=6, use_de_psd=True, use_faa=True, sequence_length=5,
                 cnn_filters=[32, 64], lstm_hidden=128, dropout=0.5):
        """
        初始化模型

        参数:
            de_psd_channels (int): DE/PSD特征通道数
            de_psd_bands (int): DE/PSD特征频段数 (考虑多个特征类型合并后的总数)
            faa_channels (int): FAA特征通道数
            faa_bands (int): FAA特征频段数 (考虑多个特征类型合并后的总数)
            num_classes (int): 类别数
            use_de_psd (bool): 是否使用DE/PSD特征
            use_faa (bool): 是否使用FAA特征
            sequence_length (int): 序列长度
            cnn_filters (list): CNN层过滤器数量列表
            lstm_hidden (int): LSTM隐藏层大小
            dropout (float): Dropout比例
        """
        super(CNNLSTM, self).__init__()

        self.use_de_psd = use_de_psd
        self.use_faa = use_faa

        # 确保至少使用一种特征
        assert use_de_psd or use_faa, "必须至少使用一种特征类型"

        # 特征提取网络
        if use_de_psd:
            # DE/PSD特征的CNN部分
            self.conv1_de_psd = nn.Conv2d(1, cnn_filters[0], kernel_size=(3, 3), padding=(1, 1))
            self.bn1_de_psd = nn.BatchNorm2d(cnn_filters[0])
            self.conv2_de_psd = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(3, 3), padding=(1, 1))
            self.bn2_de_psd = nn.BatchNorm2d(cnn_filters[1])
            self.pool_de_psd = nn.MaxPool2d(kernel_size=(2, 1))  # 只在通道维度池化

            # 计算CNN输出特征维度
            cnn_output_channels = de_psd_channels // 2 + (1 if de_psd_channels % 2 != 0 else 0)
            self.cnn_output_dim_de_psd = cnn_filters[1] * cnn_output_channels * de_psd_bands

        if use_faa:
            # FAA特征的CNN部分
            self.conv1_faa = nn.Conv2d(1, cnn_filters[0], kernel_size=(2, 1), padding=(1, 0))
            self.bn1_faa = nn.BatchNorm2d(cnn_filters[0])
            self.conv2_faa = nn.Conv2d(cnn_filters[0], cnn_filters[1], kernel_size=(2, 1), padding=(1, 0))
            self.bn2_faa = nn.BatchNorm2d(cnn_filters[1])
            self.pool_faa = nn.AdaptiveMaxPool2d((2, faa_bands))  # 自适应池化到固定大小

            # FAA特征CNN输出维度
            self.cnn_output_dim_faa = cnn_filters[1] * 2 * faa_bands

        # 计算融合后的总特征维度
        self.total_feature_dim = 0
        if use_de_psd:
            self.total_feature_dim += self.cnn_output_dim_de_psd
        if use_faa:
            self.total_feature_dim += self.cnn_output_dim_faa

        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=self.total_feature_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True
        )

        # 注意力机制
        self.attention = nn.Linear(lstm_hidden * 2, 1)

        # 分类器
        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        """前向传播"""
        # 检查输入类型
        if isinstance(x, tuple) and len(x) == 2:
            # 同时有DE/PSD和FAA特征
            de_psd_x, faa_x = x

            # 处理DE/PSD特征
            de_psd_features = self._process_de_psd(de_psd_x)

            # 处理FAA特征
            faa_features = self._process_faa(faa_x)

            # 融合特征
            combined_features = torch.cat([de_psd_features, faa_features], dim=2)

        elif self.use_de_psd and not self.use_faa:
            # 只有DE/PSD特征
            combined_features = self._process_de_psd(x)

        elif self.use_faa and not self.use_de_psd:
            # 只有FAA特征
            combined_features = self._process_faa(x)

        else:
            raise ValueError("输入数据类型不匹配模型配置")

        # LSTM前向传播
        lstm_out, _ = self.lstm(combined_features)

        # 注意力机制
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # 分类器
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def _process_de_psd(self, x):
        """处理DE/PSD特征"""
        # 输入形状: [batch, time_steps, channels, freq_bands]
        batch_size, time_steps, channels, freq_bands = x.size()

        # 重塑为CNN可接受的形状
        x = x.view(batch_size * time_steps, 1, channels, freq_bands)

        # CNN前向传播
        x = F.relu(self.bn1_de_psd(self.conv1_de_psd(x)))
        x = F.relu(self.bn2_de_psd(self.conv2_de_psd(x)))
        x = self.pool_de_psd(x)

        # 重塑为LSTM可接受的形状
        x = x.view(batch_size, time_steps, -1)

        return x

    def _process_faa(self, x):
        """处理FAA特征"""
        # 输入形状: [batch, time_steps, channels, bands]
        batch_size, time_steps, channels, bands = x.size()

        # 重塑为CNN可接受的形状
        x = x.view(batch_size * time_steps, 1, channels, bands)

        # CNN前向传播
        x = F.relu(self.bn1_faa(self.conv1_faa(x)))
        x = F.relu(self.bn2_faa(self.conv2_faa(x)))
        x = self.pool_faa(x)

        # 重塑为LSTM可接受的形状
        x = x.view(batch_size, time_steps, -1)

        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                num_epochs=50, patience=10, scheduler=None, fold_idx=None):
    """训练模型"""
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # 早停参数
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for inputs, targets in train_pbar:
            # 将数据移到设备
            if isinstance(inputs, tuple):
                inputs = tuple(x.to(device) for x in inputs)
            else:
                inputs = inputs.to(device)

            targets = targets.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

            # 更新进度条
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * train_correct / train_total:.2f}%"
            })

        # 计算训练集损失和准确率
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
            for inputs, targets in val_pbar:
                # 将数据移到设备
                if isinstance(inputs, tuple):
                    inputs = tuple(x.to(device) for x in inputs)
                else:
                    inputs = inputs.to(device)

                targets = targets.to(device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # 统计
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

                # 更新进度条
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100. * val_correct / val_total:.2f}%"
                })

        # 计算验证集损失和准确率
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # 更新历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 打印统计信息
        fold_str = f" Fold {fold_idx}" if fold_idx is not None else ""
        print(f'Epoch {epoch + 1}/{num_epochs}{fold_str} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    return model, history


def evaluate_model(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for inputs, targets in test_pbar:
            # 将数据移到设备
            if isinstance(inputs, tuple):
                inputs = tuple(x.to(device) for x in inputs)
            else:
                inputs = inputs.to(device)

            targets = targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 统计
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            # 保存预测和目标
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # 更新进度条
            test_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * test_correct / test_total:.2f}%"
            })

    # 计算测试集损失和准确率
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * test_correct / test_total

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    # 生成分类报告
    class_report = classification_report(
        all_targets, all_predictions,
        target_names=[EMOTION_LABELS[i] for i in range(6)],
        digits=4
    )

    return test_loss, test_acc, conf_matrix, class_report, all_predictions, all_targets


def plot_results(history, conf_matrix, class_report, output_path, experiment_type, cv_type, fold_idx=None):
    """绘制训练结果"""
    # 创建图形
    plt.figure(figsize=(20, 15))

    # 绘制损失
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    # 绘制准确率
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    # 绘制混淆矩阵
    plt.subplot(2, 2, 3)
    cm_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=[EMOTION_LABELS[i] for i in range(6)],
                yticklabels=[EMOTION_LABELS[i] for i in range(6)])
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # 添加分类报告
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0, 0.5, f"Classification Report:\n\n{class_report}",
             fontsize=10, family='monospace')

    # 保存图形
    plt.tight_layout()
    if fold_idx is not None:
        filename = f'{experiment_type}_{cv_type}_fold_{fold_idx}_results.png'
    else:
        filename = f'{experiment_type}_{cv_type}_results.png'
    plt.savefig(os.path.join(output_path, filename))
    plt.close()


def run_experiment(args, experiment_type, cv_type):
    """运行单个实验"""
    print(f"\n=== 运行 {experiment_type} {cv_type} 实验 ===\n")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 结果目录
    results_dir = os.path.join(args.output_dir, f'{experiment_type}_{cv_type}')
    os.makedirs(results_dir, exist_ok=True)

    if experiment_type == 'intra':
        run_intra_subject_experiment(args, cv_type, results_dir, device)
    else:
        run_cross_subject_experiment(args, cv_type, results_dir, device)


def run_intra_subject_experiment(args, cv_type, results_dir, device):
    """运行被试内实验"""
    # 获取所有被试目录
    subject_dirs = [d for d in os.listdir(os.path.join(args.data_dir, f'intra_{cv_type}'))
                    if d.startswith('subject_')]

    # 记录结果
    all_subjects_results = []

    # 处理每个被试
    for subject_dir in subject_dirs:
        subject_id = subject_dir.split('_')[1]
        print(f'\n处理被试 {subject_id}')

        # 被试结果目录
        subject_results_dir = os.path.join(results_dir, subject_dir)
        os.makedirs(subject_results_dir, exist_ok=True)

        # 找出所有训练和验证折
        subject_path = os.path.join(args.data_dir, f'intra_{cv_type}', subject_dir)
        train_files = sorted([f for f in os.listdir(subject_path) if f.startswith('train_fold_')])
        val_files = sorted([f for f in os.listdir(subject_path) if f.startswith('val_fold_')])

        # 记录该被试的所有折结果
        subject_folds_results = []

        # 处理每个折
        for fold_idx, (train_file, val_file) in enumerate(zip(train_files, val_files)):
            fold_num = fold_idx + 1
            print(f'\n处理被试 {subject_id} 的折 {fold_num}')

            # 创建数据集
            train_dataset = MultiFeatureEEGDataset(
                os.path.join(subject_path, train_file),
                feature_types=args.feature_types.split(','),
                sequence_length=args.sequence_length
            )

            val_dataset = MultiFeatureEEGDataset(
                os.path.join(subject_path, val_file),
                feature_types=args.feature_types.split(','),
                sequence_length=args.sequence_length
            )

            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )

            # 检查特征类型
            use_de_psd = any(f.startswith('de') or f.startswith('psd') for f in args.feature_types.split(','))
            use_faa = any(f.startswith('faa') for f in args.feature_types.split(','))

            # 计算频段数
            num_de_psd_bands = 0
            if use_de_psd:
                de_bands = 5 * len([f for f in args.feature_types.split(',') if f.startswith('de')])
                psd_bands = 5 * len([f for f in args.feature_types.split(',') if f.startswith('psd')])
                num_de_psd_bands = de_bands + psd_bands

            num_faa_bands = 0
            if use_faa:
                num_faa_bands = len([f for f in args.feature_types.split(',') if f.startswith('faa')])

            # 创建模型
            model = CNNLSTM(
                de_psd_channels=62,
                de_psd_bands=num_de_psd_bands,
                faa_channels=4,
                faa_bands=num_faa_bands,
                num_classes=6,
                use_de_psd=use_de_psd,
                use_faa=use_faa,
                sequence_length=args.sequence_length,
                cnn_filters=args.cnn_filters,
                lstm_hidden=args.lstm_hidden,
                dropout=args.dropout
            ).to(device)

            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # 学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )

            # 训练模型
            model, history = train_model(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                num_epochs=args.epochs,
                patience=args.patience,
                scheduler=scheduler,
                fold_idx=fold_num
            )

            # 测试模型
            test_dataset = MultiFeatureEEGDataset(
                os.path.join(subject_path, 'test.mat'),
                feature_types=args.feature_types.split(','),
                sequence_length=args.sequence_length
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )

            test_loss, test_acc, conf_matrix, class_report, _, _ = evaluate_model(
                model, test_loader, criterion, device
            )

            print(f'\n被试 {subject_id} 折 {fold_num} 测试结果:\n损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%')
            print(f'\n分类报告:\n{class_report}\n')

            # 绘制结果
            plot_results(
                history,
                conf_matrix,
                class_report,
                subject_results_dir,
                'intra',
                cv_type,
                fold_num
            )

            # 保存模型
            torch.save(model.state_dict(), os.path.join(subject_results_dir, f'fold_{fold_num}_model.pth'))

            # 记录该折的结果
            fold_result = {
                'fold': fold_num,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report,
                'history': history
            }

            subject_folds_results.append(fold_result)

        # 计算该被试的平均结果
        avg_test_acc = np.mean([fold['test_acc'] for fold in subject_folds_results])
        avg_test_loss = np.mean([fold['test_loss'] for fold in subject_folds_results])

        print(f'\n被试 {subject_id} 平均测试准确率: {avg_test_acc:.2f}%')

        # 记录该被试的结果
        subject_result = {
            'subject_id': subject_id,
            'avg_test_acc': float(avg_test_acc),
            'avg_test_loss': float(avg_test_loss),
            'folds': subject_folds_results
        }

        all_subjects_results.append(subject_result)

        # 保存该被试的结果
        with open(os.path.join(subject_results_dir, 'results.json'), 'w') as f:
            json.dump(subject_result, f, indent=4)

    # 计算所有被试的平均结果
    overall_avg_acc = np.mean([subject['avg_test_acc'] for subject in all_subjects_results])
    overall_avg_loss = np.mean([subject['avg_test_loss'] for subject in all_subjects_results])

    print(f'\n所有被试平均测试准确率: {overall_avg_acc:.2f}%')

    # 保存总体结果
    overall_result = {
        'overall_avg_acc': float(overall_avg_acc),
        'overall_avg_loss': float(overall_avg_loss),
        'subjects': all_subjects_results
    }

    with open(os.path.join(results_dir, 'overall_results.json'), 'w') as f:
        json.dump(overall_result, f, indent=4)


def run_cross_subject_experiment(args, cv_type, results_dir, device):
    """运行跨被试实验"""
    # 数据路径
    data_path = os.path.join(args.data_dir, f'cross_{cv_type}')

    # 找出所有训练和验证折
    train_files = sorted([f for f in os.listdir(data_path) if f.startswith('train_fold_')])
    val_files = sorted([f for f in os.listdir(data_path) if f.startswith('val_fold_')])

    # 记录所有折结果
    all_folds_results = []

    # 处理每个折
    for fold_idx, (train_file, val_file) in enumerate(zip(train_files, val_files)):
        fold_num = fold_idx + 1
        print(f'\n处理跨被试折 {fold_num}')

        # 创建数据集
        train_dataset = MultiFeatureEEGDataset(
            os.path.join(data_path, train_file),
            feature_types=args.feature_types.split(','),
            sequence_length=args.sequence_length
        )

        val_dataset = MultiFeatureEEGDataset(
            os.path.join(data_path, val_file),
            feature_types=args.feature_types.split(','),
            sequence_length=args.sequence_length
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        # 检查特征类型
        use_de_psd = any(f.startswith('de') or f.startswith('psd') for f in args.feature_types.split(','))
        use_faa = any(f.startswith('faa') for f in args.feature_types.split(','))

        # 计算频段数
        num_de_psd_bands = 0
        if use_de_psd:
            de_bands = 5 * len([f for f in args.feature_types.split(',') if f.startswith('de')])
            psd_bands = 5 * len([f for f in args.feature_types.split(',') if f.startswith('psd')])
            num_de_psd_bands = de_bands + psd_bands

        num_faa_bands = 0
        if use_faa:
            num_faa_bands = len([f for f in args.feature_types.split(',') if f.startswith('faa')])

        # 创建模型
        model = CNNLSTM(
            de_psd_channels=62,
            de_psd_bands=num_de_psd_bands,
            faa_channels=4,
            faa_bands=num_faa_bands,
            num_classes=6,
            use_de_psd=use_de_psd,
            use_faa=use_faa,
            sequence_length=args.sequence_length,
            cnn_filters=args.cnn_filters,
            lstm_hidden=args.lstm_hidden,
            dropout=args.dropout
        ).to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 训练模型
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=args.epochs,
            patience=args.patience,
            scheduler=scheduler,
            fold_idx=fold_num
        )

        # 测试模型
        test_dataset = MultiFeatureEEGDataset(
            os.path.join(data_path, 'test.mat'),
            feature_types=args.feature_types.split(','),
            sequence_length=args.sequence_length
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        test_loss, test_acc, conf_matrix, class_report, _, _ = evaluate_model(
            model, test_loader, criterion, device
        )

        print(f'\n跨被试折 {fold_num} 测试结果:\n损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%')
        print(f'\n分类报告:\n{class_report}\n')

        # 绘制结果
        plot_results(
            history,
            conf_matrix,
            class_report,
            results_dir,
            'cross',
            cv_type,
            fold_num
        )

        # 保存模型
        torch.save(model.state_dict(), os.path.join(results_dir, f'fold_{fold_num}_model.pth'))

        # 记录该折的结果
        fold_result = {
            'fold': fold_num,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'history': history
        }

        all_folds_results.append(fold_result)

    # 计算所有折的平均结果
    overall_avg_acc = np.mean([fold['test_acc'] for fold in all_folds_results])
    overall_avg_loss = np.mean([fold['test_loss'] for fold in all_folds_results])

    print(f'\n跨被试平均测试准确率: {overall_avg_acc:.2f}%')

    # 保存总体结果
    overall_result = {
        'overall_avg_acc': float(overall_avg_acc),
        'overall_avg_loss': float(overall_avg_loss),
        'folds': all_folds_results
    }

    with open(os.path.join(results_dir, 'overall_results.json'), 'w') as f:
        json.dump(overall_result, f, indent=4)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SEED VII 脑电情绪识别CNN-LSTM模型')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='/data/coding/dataset_split',
                        help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='/data/coding/results',
                        help='结果保存目录')
    parser.add_argument('--feature_types', type=str,
                        default='de_lds',
                        help='要使用的特征类型，逗号分隔')
    parser.add_argument('--experiment_type', type=str, choices=['intra', 'cross', 'both'],
                        default='cross', help='实验类型：被试内，跨被试，或两者')
    parser.add_argument('--cv_type', type=str, choices=['loo', 'kfold', 'both'],
                        default='kfold', help='交叉验证类型：留一法，k折，或两者')

    # 模型参数
    parser.add_argument('--sequence_length', type=int, default=5,
                        help='序列长度 (时间步)')
    parser.add_argument('--cnn_filters', type=int, nargs='+', default=[32, 64],
                        help='CNN过滤器数量')
    parser.add_argument('--lstm_hidden', type=int, default=128,
                        help='LSTM隐藏层大小')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout比例')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作线程数')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存参数
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 打印实验设置
    print("\n=== SEED VII 脑电情绪识别实验 ===")
    print(f"特征类型: {args.feature_types}")
    print(f"序列长度: {args.sequence_length}")
    print(f"CNN过滤器: {args.cnn_filters}")
    print(f"LSTM隐藏层: {args.lstm_hidden}")
    print(f"Dropout: {args.dropout}")
    print(f"批量大小: {args.batch_size}")
    print(f"最大训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"权重衰减: {args.weight_decay}")
    print(f"随机种子: {args.seed}")

    # 运行实验
    if args.experiment_type in ['intra', 'both']:
        if args.cv_type in ['loo', 'both']:
            run_experiment(args, 'intra', 'loo')
        if args.cv_type in ['kfold', 'both']:
            run_experiment(args, 'intra', 'kfold')

    if args.experiment_type in ['cross', 'both']:
        if args.cv_type in ['loo', 'both']:
            run_experiment(args, 'cross', 'loo')
        if args.cv_type in ['kfold', 'both']:
            run_experiment(args, 'cross', 'kfold')

    print("\n所有实验完成!")


if __name__ == "__main__":
    main()
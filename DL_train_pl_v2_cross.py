import os
import numpy as np
from typing import Tuple, List, Dict
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if device != 'cpu' and torch.cuda.is_bf16_supported() else torch.float32
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
seed = 1
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
print(f'device: {device}\ndtype: {dtype}\nSeed:{seed} GPU:{os.environ["CUDA_VISIBLE_DEVICES"]}')

# EA 预处理函数
from scipy.linalg import fractional_matrix_power


def EA(x):
    cov = np.zeros((x.shape[0], 24, 24))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (0.00000001) * np.eye(24)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


# 1. 卷积模块（用于提取局部特征）
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),  # 时间卷积
            nn.Conv2d(40, 40, (22, 1), (1, 1)),  # 空间卷积
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # 池化，沿时间维度切片得到"patch"
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 调整通道数
            Rearrange('b e (h) (w) -> b (h w) e'),  # 重排为序列形式
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x

# 2. 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # 注意力能量
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)  # 注意力权重
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)  # 加权求和
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

# 3. 残差连接模块
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

# 4. 前馈网络模块
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# 5. Transformer 编码器块
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

# 6. Transformer 编码器（多层堆叠）
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

# 7. 分类头
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        # 全局平均池化
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        # 全连接层分类器（备用，实际使用 clshea d）
        self.fc = nn.Sequential(
            nn.Linear(1680, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)  # 展平
        out = self.fc(x)  # 通过全连接层分类
        return out  # 返回特征和分类结果

# 8. 完整的 Conformer 模型
class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),      # 1. 卷积提取局部特征
            TransformerEncoder(depth, emb_size),  # 2. Transformer 编码器
            ClassificationHead(emb_size, n_classes)  # 3. 分类头
        )


class EEGNet(nn.Module):
    def __init__(self, in_chan=0, fc_num=0, out_chann=0):
        super(EEGNet, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, in_chan), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        self.fc1 = nn.Linear(fc_num, out_chann)

    def forward(self, x):
        #print('x',x.shape)
        x = x.permute(0, 1, 3, 2)
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(x.size()[0], -1)
        #print('x',x.shape)
        x = self.fc1(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepConvNet(nn.Module):
    def __init__(self, n_classes, input_ch, input_time, batch_norm=True, batch_norm_alpha=0.1):
        super(DeepConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1),  # 10 -> 5
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1, bias=False),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1),
                # nn.InstanceNorm2d(n_ch1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(self.n_ch4),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes),
                                 nn.Dropout(p=0.2))  ####################### classifier
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.l2normalize(output)
        output = self.clf(output)

        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowConvNet(nn.Module):
    def __init__(self, n_classes, input_ch, fc_ch, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = input_ch

        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 13), stride=1, padding=(6, 7)),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5))

        self.fc = nn.Linear(fc_ch, n_classes)

    def forward(self, x):
        #x = x.permute(0, 1, 3, 2)  #跑pretrian的时候关一下
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1, 35), (1, 7))
        x = torch.log(x)
        x = x.flatten(1)
        x = torch.nn.functional.dropout(x)
        #print('x',x.shape)
        x = self.fc(x)
        return x




# 假设有多个被试的数据文件，命名为类似格式：
# 原始数据文件结构：
# ./processed_data/subject_01_sad_car_2D.npy
# ./processed_data/subject_01_happy_car_2D.npy
# ./processed_data/subject_02_sad_car_2D.npy
# ./processed_data/subject_02_happy_car_2D.npy
# 等等...

def get_all_subject_ids():
    """获取所有被试的ID"""
    import glob
    import re

    sad_files = glob.glob("./processed_data/*sad_car_2D.npy")
    subject_ids = []

    for file_path in sad_files:
        # 提取被试ID，假设文件名格式为: subject_XX_sad_car_2D.npy
        match = re.search(r'subject_(\d+)_sad_car_2D', file_path)
        if match:
            subject_ids.append(match.group(1))

    return sorted(subject_ids)


def load_subject_data(subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载单个被试的数据"""
    sad_path = f"./processed_data/subject_{subject_id}_sad_car_2D.npy"
    happy_path = f"./processed_data/subject_{subject_id}_happy_car_2D.npy"

    sad_data = np.load(sad_path)  # [N_sad, C, T]
    happy_data = np.load(happy_path)  # [N_happy, C, T]

    # 合并sad和happy数据
    X = np.concatenate([sad_data, happy_data], axis=0)
    y = np.concatenate([
        np.zeros(sad_data.shape[0], dtype=np.int64),  # 0表示sad
        np.ones(happy_data.shape[0], dtype=np.int64)  # 1表示happy
    ], axis=0)

    return X, y


def load_all_subjects_data() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """加载所有被试的数据"""
    subject_ids = get_all_subject_ids()
    all_data = {}

    for sub_id in subject_ids:
        X, y = load_subject_data(sub_id)
        all_data[sub_id] = (X, y)
        print(f"Loaded subject {sub_id}: {X.shape} samples")

    return all_data


def create_cross_subject_split(target_subject_id: str, apply_ea: bool = False) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建跨被试的数据划分：
    - 训练集：除了目标被试之外的所有被试数据
    - 测试集：目标被试的所有数据

    参数:
        target_subject_id: 作为测试集的被试ID
        apply_ea: 是否应用EA预处理

    返回:
        X_train, y_train, X_test, y_test
    """
    # 加载所有被试数据
    all_data = load_all_subjects_data()

    if target_subject_id not in all_data:
        raise ValueError(f"Target subject {target_subject_id} not found in data")

    # 分离训练集和测试集
    train_data = []
    train_labels = []

    for sub_id, (X, y) in all_data.items():
        if sub_id == target_subject_id:
            # 目标被试作为测试集
            X_test, y_test = X, y
        else:
            # 其他被试作为训练集
            train_data.append(X)
            train_labels.append(y)

    # 合并所有训练被试的数据
    if train_data:
        X_train = np.concatenate(train_data, axis=0)
        y_train = np.concatenate(train_labels, axis=0)
    else:
        raise ValueError("No training subjects found")

    # 应用EA预处理（如果需要）
    if apply_ea:
        X_train = EA(X_train)
        X_test = EA(X_test)

    print(f"Training set: {X_train.shape} samples")
    print(f"Test set: {X_test.shape} samples")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")

    return X_train, y_train, X_test, y_test


# 保留原来的数据集和模型定义
class EEGWindowDataset(Dataset):
    """简单 EEG 窗口数据集"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # [C, T]
        y = self.y[idx]
        x = np.expand_dims(x, axis=0)  # 增加通道维度 [1, C, T]
        return torch.from_numpy(x), torch.tensor(y)


# 保留所有的模型定义（EEGNet, DeepConvNet, ShallowConvNet, Conformer等）
# ... [所有模型定义代码保持不变] ...

class Trainer:
    """自定义训练器"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 lr=1e-3, weight_decay=1e-2, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=20, verbose=True)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }

        self.best_val_acc = 0.0
        self.best_model_state = None

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += data.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def validate(self, loader):
        """验证/测试"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)
                total_correct += (preds == target).sum().item()
                total_samples += data.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def fit(self, max_epochs=1000, patience=100, save_path='best_model.pth'):
        """训练模型"""
        print(f"Training on device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

        no_improve = 0
        best_epoch = 0

        for epoch in range(max_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate(self.val_loader)

            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                }, save_path)
                no_improve = 0
                best_epoch = epoch
            else:
                no_improve += 1

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1:3d}/{max_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nBest model from epoch {best_epoch + 1} with val_acc: {self.best_val_acc:.4f}")

        return self.history

    def test(self):
        """测试最佳模型"""
        print("\n" + "=" * 50)
        print("Testing best model...")

        test_loss, test_acc = self.validate(self.test_loader)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("=" * 50)

        return test_loss, test_acc

    def plot_training_history(self, save_path='training_history.png'):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        epochs = range(1, len(self.history['train_loss']) + 1)

        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.history['lr'], 'g-', label='Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()


def build_dataloaders_cross_subject(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
        val_ratio: float = 0.1
):
    """
    为跨被试实验构建数据加载器
    参数:
        val_ratio: 从训练集中划分验证集的比例
    """
    # 训练集
    train_dataset = EEGWindowDataset(X_train, y_train)

    # 从训练集中划分验证集
    n_train = len(train_dataset)
    n_val = int(n_train * val_ratio)
    n_train_final = n_train - n_val

    train_set, val_set = random_split(
        train_dataset,
        lengths=[n_train_final, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # 测试集
    test_dataset = EEGWindowDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train set: {len(train_set)} samples")
    print(f"Val set: {len(val_set)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


def cross_subject_evaluation(target_subject_id: str, model_name: str = 'DeepConvNet', apply_ea: bool = False):
    """
    对指定被试进行跨被试评估
    """
    print(f"\n{'=' * 60}")
    print(f"Cross-subject evaluation for subject {target_subject_id}")
    print(f"{'=' * 60}")

    # 创建跨被试数据划分
    X_train, y_train, X_test, y_test = create_cross_subject_split(
        target_subject_id,
        apply_ea=apply_ea
    )

    n_channels = X_train.shape[1]
    nTime = X_train.shape[2]

    # 创建数据加载器
    train_loader, val_loader, test_loader = build_dataloaders_cross_subject(
        X_train, y_train, X_test, y_test,
        batch_size=32,
        val_ratio=0.1
    )

    # 选择模型
    if model_name == 'DeepConvNet':
        model = DeepConvNet(2, 24, 300)
    elif model_name == 'ShallowConvNet':
        # 需要根据实际数据计算fc_ch参数
        # 这里假设和之前一样
        model = ShallowConvNet(2, 24, 12168)
    elif model_name == 'EEGNet':
        model = EEGNet(in_chan=24, fc_num=152, out_chann=2)
    elif model_name == 'EEGConformer':
        model = Conformer(emb_size=40, depth=6, n_classes=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\nModel: {model_name}")
    print(f"Input shape: {n_channels} channels, {nTime} time points")

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=1e-3,
        weight_decay=1e-2
    )

    # 训练模型
    print("\nStarting training...")
    history = trainer.fit(
        max_epochs=1000,
        patience=100,
        save_path=f'best_model_subject_{target_subject_id}.pth'
    )

    # 测试模型
    test_loss, test_acc = trainer.test()

    # 保存训练历史图
    trainer.plot_training_history(f'training_history_subject_{target_subject_id}.png')

    return test_acc, trainer, history


def run_all_cross_subject_evaluations(model_name: str = 'DeepConvNet', apply_ea: bool = False):
    """
    对所有被试进行留一被试交叉验证
    """
    # 获取所有被试ID
    subject_ids = get_all_subject_ids()
    print(f"Found {len(subject_ids)} subjects: {subject_ids}")

    results = {}

    for target_subject in subject_ids:
        print(f"\n{'#' * 60}")
        print(f"Evaluating with subject {target_subject} as test set")
        print(f"{'#' * 60}")

        try:
            test_acc, trainer, history = cross_subject_evaluation(
                target_subject,
                model_name=model_name,
                apply_ea=apply_ea
            )
            results[target_subject] = test_acc
        except Exception as e:
            print(f"Error evaluating subject {target_subject}: {e}")
            results[target_subject] = None

    # 打印结果汇总
    print(f"\n{'=' * 60}")
    print("CROSS-SUBJECT EVALUATION RESULTS SUMMARY")
    print(f"{'=' * 60}")

    valid_results = [acc for acc in results.values() if acc is not None]

    for subject, acc in results.items():
        status = f"{acc:.4f}" if acc is not None else "Failed"
        print(f"Subject {subject}: {status}")

    if valid_results:
        mean_acc = np.mean(valid_results)
        std_acc = np.std(valid_results)
        print(f"\nMean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Number of valid evaluations: {len(valid_results)}/{len(results)}")

    return results


if __name__ == "__main__":
    # 示例1: 对单个被试进行跨被试评估
    target_subject = "03"  # 假设被试ID是"01"
    cross_subject_evaluation(target_subject, model_name='EEGConformer', apply_ea=False)

    # 示例2: 对所有被试进行留一被试交叉验证
    # results = run_all_cross_subject_evaluations(
    #     model_name='DeepConvNet',
    #     apply_ea=False
    # )
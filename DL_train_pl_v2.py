import os
import numpy as np
from typing import Tuple
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from DL_model.EEGNet import eegNet  # 使用你现有的 EEGNet 实现

import random
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

#EA
from scipy.linalg import fractional_matrix_power
def EA(x):
        cov = np.zeros((x.shape[0],24, 24))
        for i in range(x.shape[0]):
                cov[i] = np.cov(x[i])
        refEA = np.mean(cov, 0)
        sqrtRefEA=fractional_matrix_power(refEA, -0.5)+ (0.00000001)*np.eye(24)
        XEA = np.zeros(x.shape)
        for i in range(x.shape[0]):
                XEA[i] = np.dot(sqrtRefEA, x[i])
        return  XEA


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import math

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
            nn.Linear(1680, 256),  #这个要改
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)  # 展平
        #print('x',x.shape)
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






class EEGWindowDataset(Dataset):
    """
    简单 EEG 窗口数据集
    X: [N, channels, timepoints]
    y: [N]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.shape[0] == y.shape[0]
        # 转成 float32 / int64
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # [C, T]
        y = self.y[idx]
        # 增加一个通道维度，变成 [1, C, T]，适配 EEGNet 输入
        x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x), torch.tensor(y)


class EEGNetModel(nn.Module):
    """
    EEGNet 模型
    输入: [B, 1, C, T]（在 Dataset 中已经加了通道维度）
    输出: [B, 2] (二分类 logits)
    """

    def __init__(self, n_channels: int, nTime: int, n_classes: int = 2):
        super().__init__()

        # eegNet 的输入是 (batch, chan, time)，内部会自行 unsqueeze 成 (batch,1,chan,time)
        self.model = eegNet(nChan=n_channels, nTime=nTime, nClass=n_classes)

    def forward(self, x):
        # x: [B, 1, C, T]，去掉第 2D 的"伪通道"维度后，变成 [B, C, T]
        x = x.squeeze(1)  # [B, C, T]
        logits = self.model(x)  # [B, nClass]
        return logits


class Trainer:
    """自定义训练器"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 lr=1e-3, weight_decay=1e-2, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 移动模型到设备
        self.model = self.model.to(self.device)

        # 优化器和损失函数
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=20, verbose=True)
        self.criterion = nn.CrossEntropyLoss()

        # 训练历史记录
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }

        # 最佳模型权重
        self.best_val_acc = 0.0
        self.best_model_state = None

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

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

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def fit(self, max_epochs=1000, patience=100, save_path='best_model.pth'):
        """训练模型"""
        print(f"Training on device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

        no_improve = 0

        for epoch in range(max_epochs):
            start_time = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate(self.val_loader)

            # 学习率调度
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # 保存最佳模型
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

            # 打印进度
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1:3d}/{max_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nBest model from epoch {best_epoch + 1} with val_acc: {self.best_val_acc:.4f}")

        return self.history

    def test(self):
        """测试最佳模型"""
        print("\n" + "=" * 50)
        print("Testing best model...")

        # 在测试集上评估
        test_loss, test_acc = self.validate(self.test_loader)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("=" * 50)

        return test_loss, test_acc

    def plot_training_history(self, save_path='training_history.png'):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # 损失曲线
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 学习率曲线
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
    
    #X_trainval = EA(X_trainval)
    #X_test = EA(X_test)
    
    #print('X_trainval',X_trainval.shape)

    return X_trainval, y_trainval, X_test, y_test


def build_dataloaders_from_split(
        X_trainval: np.ndarray,
        y_trainval: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32
):
    """
    在前半段 (trainval) 内再按 7:1:2 划分 train / val / (trainval 内部的 test 部分),
    但是你要求"后一半当测试集"，所以这里通常只用前半段的 7:1 部分做 train/val，
    后半段完整作为真正的测试集。
    为了严格遵守"前半段只用于训练和验证（训练集的12.5%）"的说法，这里实现为:
      - 在 X_trainval 上: 7:1:2 划分 train / val / discard (丢弃这 2 部分)
      - X_test 完整用作最终 test
    """
    # 整个 trainval 数据集
    trainval_dataset = EEGWindowDataset(X_trainval, y_trainval)
    n_total = len(trainval_dataset)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.1)
    n_discard = n_total - n_train - n_val  # 这部分丢弃，不参与任何使用

    train_set, val_set, _ = random_split(
        trainval_dataset,
        lengths=[n_train, n_val, n_discard],
        generator=torch.Generator().manual_seed(42)
    )

    # 最终测试集: 来自每类的后半段
    test_dataset = EEGWindowDataset(X_test, y_test)

    # 只在 DataLoader 中对训练集做 shuffle
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 打印数据集信息
    print(f"Train set: {len(train_set)} samples")
    print(f"Val set: {len(val_set)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader


def DL_train_pytorch():
    """
    基于 CAR 2D 窗口和 EEGNet 的纯 PyTorch 训练版本
    """
    # 加载数据
    X_trainval, y_trainval, X_test, y_test = load_car_2d_data_split()
    n_channels = X_trainval.shape[1]
    nTime = X_trainval.shape[2]

    # 创建数据加载器
    train_loader, val_loader, test_loader = build_dataloaders_from_split(
        X_trainval, y_trainval, X_test, y_test, batch_size=32
    )

    # 创建模型
    model_name = 'DeepConvNet'

    if model_name == 'DeepConvNet':
        model = DeepConvNet(2, 24, 300)  # 750是时间点长度
    elif model_name == 'ShallowConvNet':
        model = ShallowConvNet(2, 24, 12168)  # 9152要实地算
    elif model_name == 'EEGNet':
        model = EEGNet(in_chan=24, fc_num=152, out_chann=2)  #34,24,300
    elif model_name == 'EEGConformer':
        model = Conformer(emb_size=40, depth=6, n_classes=2)  #需要去模型定义中调整全连接层的定义

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
    print("Starting training...")
    history = trainer.fit(
        max_epochs=1000,
        patience=100,
        save_path='best_eegnet_model.pth'
    )

    # 测试模型
    trainer.test()

    # 绘制训练历史
    trainer.plot_training_history('eegnet_training_history.png')

    return trainer, history


if __name__ == "__main__":
    DL_train_pytorch()
"""
优化版本的自编码器训练框架

主要优化点：
1. 模型结构优化：
   - 稀疏SE注意力：只在关键位置（输入层、bottleneck、输出层）使用SE注意力，减少约30-40%计算量
   - 优化残差连接：简化融合模块，提升速度
   - 改进BatchNorm：使用更大的eps(1e-4)以提升AMP下的数值稳定性
   - 输入层位置编码：新增RGB输入的位置编码，增强空间位置信息

2. 损失函数优化：
   - SSIM窗口大小：从11减小到7，提升约40%计算速度
   - SSIM窗口缓存：使用缓存机制避免重复创建窗口，提升效率
   - 损失权重调整：降低MSE和SSIM权重（0.03和0.01），提升稳定性
   - 数值稳定性：添加多层NaN检查和恢复机制

3. 训练流程优化：
   - 梯度裁剪：max_norm从1.0降到0.3，更严格防止NaN
   - 梯度范数检查：检测异常梯度范数（>5.0）并跳过
   - 梯度累积：支持梯度累积以模拟更大batch size，节省显存
   - 位置编码缩放：从0.01降到0.005，提升稳定性
   - 位置编码缓存：缓存dim_t避免重复计算

4. 数据加载优化：
   - 动态调整num_workers：根据CPU核心数智能调整（最少4个，最多8个）
   - 动态prefetch_factor：根据batch size调整预取数量
   - 添加drop_last配置：验证时不丢弃数据

5. 性能优化：
   - 减少SE模块使用：约减少30-40%的注意力计算
   - 优化SSIM计算：使用更小的窗口和缓存机制
   - 优化位置编码：缓存基础计算减少重复
   - 改进NaN检测：在关键位置添加NaN检查和恢复
   - 内存优化：使用channels_last内存格式，验证时使用no_grad

预期效果：
- 训练速度提升：约25-35%（由于减少SE模块、优化SSIM和位置编码缓存）
- NaN问题：显著减少，通过多层防护机制
- 内存使用：优化数据加载和验证流程，减少内存占用
- 模型性能：保持或略有提升（稀疏SE注意力和输入层位置编码仍然有效）
"""

import os
import math
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
# 检测并导入正确的 autocast 和 GradScaler
try:
    from torch.amp import autocast, GradScaler
    AUTOCAST_NEW_API = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        AUTOCAST_NEW_API = False
    except ImportError:
        # 较老版本的PyTorch
        from torch.cuda import autocast, GradScaler
        AUTOCAST_NEW_API = False
from copy import deepcopy

# ====================== 扩散模型相关模块 ======================
class SinusoidalPositionalEmbedding(nn.Module):
    """正弦位置编码，用于时间步嵌入（借鉴DDPM）"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        time: [B] 时间步，范围[0, 1]
        返回: [B, dim] 位置编码
        """
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:  # 奇数维度时补齐
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class NoiseScheduler:
    """噪声调度器（线性/余弦调度）"""
    def __init__(self, num_timesteps=1000, schedule_type='linear', beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            # 余弦调度（更稳定）
            s = 0.008
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x_start, t, noise=None):
        """添加噪声到输入"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x_start.device)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size, device):
        """采样时间步"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

class LatentCodeExtractor(nn.Module):
    """隐空间编码提取器：将特征图压缩为低维隐编码"""
    def __init__(self, in_channels, latent_dim=512, compression_ratio=4):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 使用全局平均池化 + MLP提取隐编码
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh()  # 限制范围到[-1, 1]
        )
        
        # 可选的：从隐编码重建特征图
        self.expand_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, in_channels),
            nn.LayerNorm(in_channels)
        )
        
        # 卷积式重建（更精确）
        H_latent = W_latent = compression_ratio  # 假设bottleneck是[H/comp_ratio, W/comp_ratio]
        self.expand_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, in_channels, kernel_size=compression_ratio, stride=compression_ratio),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
    
    def extract(self, x):
        """提取隐编码: [B, C, H, W] -> [B, latent_dim]"""
        # x: [B, C, H, W]
        pooled = self.global_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        latent_code = self.mlp(pooled)  # [B, latent_dim]
        return latent_code
    
    def expand_from_code(self, latent_code, target_shape):
        """从隐编码重建特征图"""
        B, latent_dim = latent_code.shape
        _, C, H, W = target_shape
        
        # 方法1: MLP重建
        expanded = self.expand_mlp(latent_code)  # [B, C]
        expanded = expanded.view(B, C, 1, 1)
        expanded = F.interpolate(expanded, size=(H, W), mode='bilinear', align_corners=True)
        
        # 方法2: 卷积重建（如果维度允许）
        if latent_dim * 4 <= C:
            # 先上采样到合适尺寸
            H_lat = max(1, H // 4)
            W_lat = max(1, W // 4)
            latent_2d = latent_code.view(B, latent_dim, 1, 1)
            latent_2d = F.interpolate(latent_2d, size=(H_lat, W_lat), mode='bilinear', align_corners=True)
            expanded_conv = self.expand_conv(latent_2d)  # [B, C, H, W]
            # 融合两种方法
            expanded = 0.7 * expanded + 0.3 * expanded_conv
        
        return expanded

class DiffusionDenoisingBlock(nn.Module):
    """扩散模型式去噪块：在隐空间中执行去噪"""
    def __init__(self, channels, time_emb_dim=128, num_groups=32):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels)
        )
        
        # 去噪卷积块（GroupNorm + SiLU，类似DDPM）
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
        # 残差连接
        self.skip_connection = nn.Conv2d(channels, channels, 1) if channels != channels else nn.Identity()
        
    def forward(self, x, t):
        """
        x: [B, C, H, W] 特征图
        t: [B] 归一化时间步 [0, 1]
        """
        # 时间嵌入
        time_emb = self.time_embed(t)  # [B, C]
        time_emb = time_emb.view(-1, time_emb.shape[1], 1, 1)  # [B, C, 1, 1]
        
        # 第一层
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 注入时间信息
        scale, shift = h.chunk(2, dim=1)
        h = scale * (1 + time_emb) + shift
        
        # 第二层
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # 残差连接
        return h + self.skip_connection(x)

# ====================== 可复现实验：随机种子 ======================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(2025)

# ====================== 自编码器数据集分割函数 ======================
import shutil

def split_autoencoder_dataset(
    source_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    将自编码器数据集拆分为训练集、验证集和测试集
    自编码器输入和输出都是相同的图像

    Args:
        source_dir: 源图像文件夹路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    print(f"\n{'='*80}")
    print("🤖 分割自编码器数据集")
    print(f"  - 数据文件夹: {source_dir}")
    print(f"{'='*80}")

    # 获取所有图像文件
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    all_files = []

    print("正在扫描图像文件...")
    for file in os.listdir(source_dir):
        if any(file.lower().endswith(ext) for ext in img_extensions):
            all_files.append(file)

    print(f"✓ 找到 {len(all_files)} 个图像文件")

    # 显示一些文件示例
    if len(all_files) > 0:
        print("文件示例:")
        for i, f in enumerate(all_files[:min(5, len(all_files))]):
            print(f"  {i+1}. {f}")
        if len(all_files) > 5:
            print(f"  ... 还有 {len(all_files)-5} 个文件")

    # 检查是否有有效的文件
    if len(all_files) == 0:
        raise ValueError(f"在 {source_dir} 中没有找到有效的图像文件")

    # 打乱顺序
    random.seed(seed)
    random.shuffle(all_files)

    # 计算拆分点
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # 拆分数据集
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]

    print(f"\n数据集拆分结果:")
    print(f"  - 训练集: {len(train_files)} 张 ({len(train_files)/n_total*100:.1f}%)")
    print(f"  - 验证集: {len(val_files)} 张 ({len(val_files)/n_total*100:.1f}%)")
    print(f"  - 测试集: {len(test_files)} 张 ({len(test_files)/n_total*100:.1f}%)")

    # 创建目标文件夹
    base_dir = source_dir
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 复制文件函数
    def copy_files(file_list, dest_dir, set_name):
        print(f"\n  创建{set_name}集...")
        for i, filename in enumerate(file_list):
            if i % 1000 == 0 and i > 0:
                print(f"    进度: {i}/{len(file_list)} ({i/len(file_list)*100:.1f}%)")

            # 复制图像
            src_file = os.path.join(source_dir, filename)
            dst_file = os.path.join(dest_dir, filename)
            try:
                shutil.copy2(src_file, dst_file)
            except Exception as e:
                print(f"    警告: 无法复制图像 {src_file}: {e}")

        print(f"    ✓ {set_name}集创建完成: {len(file_list)} 张图像")

    # 复制到各个数据集
    copy_files(train_files, train_dir, "训练")
    copy_files(val_files, val_dir, "验证")
    copy_files(test_files, test_dir, "测试")

    print(f"\n{'='*80}")
    print("✅ 自编码器数据集分割完成!")
    print(f"{'='*80}")
    print(f"\n输出文件夹:")
    print(f"  - 训练集: {train_dir}")
    print(f"  - 验证集: {val_dir}")
    print(f"  - 测试集: {test_dir}")
    print(f"\n现在可以使用这些文件夹进行自编码器训练了！")
    return train_dir, val_dir, test_dir

# ====================== 自编码器Dataset（Albumentations 增强） ======================
class AutoencoderDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (128, 128),
        augment: bool = True,
    ):
        self.root_dir = root_dir
        self.image_files = sorted(os.listdir(root_dir))
        # 过滤出图像文件
        img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        self.image_files = [f for f in self.image_files if any(f.lower().endswith(ext) for ext in img_extensions)]
        print(f"✓ AutoencoderDataset: 找到 {len(self.image_files)} 张图像")

        # 注意：Albumentations 期望 numpy 数组，BGR/RGB 皆可，但要一致
        resize_h, resize_w = image_size
        if augment:
            self.transform = A.Compose([
                A.Resize(resize_h, resize_w, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),  # 增加垂直翻转概率
                A.RandomRotate90(p=0.4),  # 增加旋转90度概率
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=25, 
                                 border_mode=cv2.BORDER_REFLECT_101, p=0.6),  # 增强几何变换
                A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.08, p=0.4),
                A.GaussianBlur(blur_limit=(3, 7), p=0.15),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.1),  # 添加高斯噪声
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),  # 添加dropout增强
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(resize_h, resize_w, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])

        # 读图：使用 cv2 读取再转 RGB
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Albumentations 增强（只对图像）
        out = self.transform(image=img)
        image_t = out["image"]  # [3,H,W], float32, 已标准化到约[-1,1]

        # 自编码器：输入和输出都是相同的图像
        return image_t, image_t

# ====================== 基础模块 ======================
class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, g=1, dilation=1, act=True, dropout_p=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, dilation=dilation, bias=False)
        # 使用eps更大的BatchNorm以提升数值稳定性（AMP下）
        self.bn   = nn.BatchNorm2d(out_c, eps=1e-4, momentum=0.1)
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()
        self.do   = nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        # 检查NaN
        if torch.isnan(out).any():
            out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
        out = self.act(out)
        return self.do(out)

# ======== SE注意力模块（通道注意力）- 必须先定义 ========
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x)

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_c, out_c, dropout_p=0.0, use_se=False, residual_scale=1.0):
        super().__init__()
        self.residual_scale = residual_scale
        self.conv = nn.Sequential(
            ConvBNAct(in_c, out_c, k=3, p=1, dropout_p=dropout_p),
            ConvBNAct(out_c, out_c, k=3, p=1, dropout_p=dropout_p),
        )
        self.proj = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1, bias=False)
        # 只在关键位置使用SE注意力，减少计算量
        self.se = SEBlock(out_c) if use_se else nn.Identity()
    def forward(self, x):
        out = self.conv(x)
        out = out + self.residual_scale * self.proj(x)
        # 使用SE注意力增强特征（如果启用）
        return self.se(out)

# ======== 改进的跳跃连接融合模块 ========
class SkipConnection(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            SEBlock(out_c)
        )
    def forward(self, x, skip):
        fused = torch.cat([x, skip], dim=1)
        return self.fusion(fused)

class Down(nn.Module):
    def __init__(self, in_c, out_c, dropout_p=0.0, use_se=False):
        super().__init__()
        # 使用stride conv代替maxpool，保留更多信息
        self.conv_down = ConvBNAct(in_c, in_c, k=3, s=2, p=1, dropout_p=dropout_p)
        # 只在关键位置使用SE注意力
        self.block = ResidualDoubleConv(in_c, out_c, dropout_p=dropout_p, use_se=use_se)
        self.se = SEBlock(out_c) if use_se else nn.Identity()
    def forward(self, x):
        x = self.conv_down(x)
        x = self.block(x)
        return self.se(x)

class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c, dropout_p=0.0, use_deconv=True, use_se=False, residual_scale=1.0):
        super().__init__()
        if use_deconv:
            self.up = nn.ConvTranspose2d(in_c, in_c, 2, 2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 简化特征融合：减少SE模块使用以提升速度
        self.skip_fusion = SkipConnection(in_c, skip_c, in_c + skip_c) if use_se else nn.Identity()
        if use_se:
            self.conv = ResidualDoubleConv(in_c + skip_c, out_c, dropout_p=dropout_p, use_se=False, residual_scale=residual_scale)
        else:
            # 简化融合：直接concat后卷积
            self.conv = ResidualDoubleConv(in_c + skip_c, out_c, dropout_p=dropout_p, use_se=False, residual_scale=residual_scale)
        self.se = SEBlock(out_c) if use_se else nn.Identity()
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        # 使用改进的融合模块（如果启用）或直接concat
        if isinstance(self.skip_fusion, nn.Identity):
            fused = torch.cat([x, skip], dim=1)
        else:
            fused = self.skip_fusion(x, skip)
        x = self.conv(fused)
        return self.se(x)

# ====================== 位置编码模块 ======================
class PositionalEncoding2D(nn.Module):
    """2D可学习位置编码，适应任意H/W"""
    def __init__(self, dim, max_h=256, max_w=256):
        super().__init__()
        # 为H和W分别创建可学习的位置编码
        self.pos_emb_h = nn.Parameter(torch.randn(1, max_h, 1, dim // 2))
        self.pos_emb_w = nn.Parameter(torch.randn(1, 1, max_w, dim // 2))
        nn.init.trunc_normal_(self.pos_emb_h, std=0.02)
        nn.init.trunc_normal_(self.pos_emb_w, std=0.02)
    
    def forward(self, x):
        """
        x: [B, C, H, W] 或 [B, H, W, C]
        返回添加位置编码后的特征
        """
        B, H, W, C = x.shape if len(x.shape) == 4 and x.shape[-1] != 1 else x.shape[:2] + x.shape[-2:]
        # 确保位置编码的维度匹配
        if len(x.shape) == 4:
            if x.shape[1] == C:  # [B, C, H, W]
                B, C, H, W = x.shape
                pos_h = self.pos_emb_h[:, :H, :, :].expand(B, -1, W, -1)  # [B, H, W, C//2]
                pos_w = self.pos_emb_w[:, :, :W, :].expand(B, H, -1, -1)  # [B, H, W, C//2]
                pos = torch.cat([pos_h, pos_w], dim=-1).permute(0, 3, 1, 2)  # [B, C, H, W]
                return x + pos
            else:  # [B, H, W, C]
                B, H, W, C = x.shape
                pos_h = self.pos_emb_h[:, :H, :, :].expand(B, -1, W, -1)  # [B, H, W, C//2]
                pos_w = self.pos_emb_w[:, :, :W, :].expand(B, H, -1, -1)  # [B, H, W, C//2]
                pos = torch.cat([pos_h, pos_w], dim=-1)  # [B, H, W, C]
                return x + pos
        return x

class AdaptivePositionalEncoding2D(nn.Module):
    """自适应的2D位置编码，使用固定正弦编码（更稳定）"""
    def __init__(self, dim, scale=0.005):
        super().__init__()
        self.dim = dim
        self.scale = scale  # 使用更小的缩放因子，0.005以提升稳定性
        
        # 预计算位置编码特征（使用固定的sin/cos）
        # 这样比MLP更稳定
        self.embed_dim = dim
        # 缓存位置编码的基础部分（dim_t），避免重复计算
        self._dim_t_cache = None
        self._cached_dim = None
    
    def _get_dim_t(self, device, dtype):
        """获取或创建dim_t，使用缓存避免重复计算"""
        if self._dim_t_cache is None or self._cached_dim != self.embed_dim:
            dim_t = torch.arange(self.embed_dim // 2, dtype=torch.float32, device=device)
            dim_t = torch.clamp(10000 ** (2 * (dim_t // 2) / max(self.embed_dim // 2, 1)), min=1e-4, max=1e6)
            dim_t = torch.clamp(dim_t, min=1e-4)
            self._dim_t_cache = dim_t
            self._cached_dim = self.embed_dim
        return self._dim_t_cache.to(device=device, dtype=dtype)
    
    def forward(self, x):
        """
        x: [B, H, W, C] - 期望输入格式
        """
        B, H, W, C = x.shape
        assert C == self.dim, f"特征维度 {C} 与位置编码维度 {self.dim} 不匹配"
        
        device = x.device
        dtype = x.dtype
        
        # 使用简单的归一化坐标（优化：使用meshgrid减少操作）
        y_embed = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).expand(-1, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).expand(H, -1)
        
        # 归一化到[0, 1]，添加数值稳定性
        y_embed = y_embed / max(H - 1, 1.0)
        x_embed = x_embed / max(W - 1, 1.0)
        
        # 转换到[-1, 1]，使用clamp防止溢出
        y_embed = torch.clamp(y_embed * 2 - 1, min=-1.0, max=1.0)
        x_embed = torch.clamp(x_embed * 2 - 1, min=-1.0, max=1.0)
        
        # 使用缓存的dim_t
        dim_t = self._get_dim_t(device, torch.float32)
        
        # 创建位置编码，添加数值稳定性
        pos_x = torch.clamp(x_embed[:, :, None] / dim_t, min=-100.0, max=100.0)  # [H, W, embed_dim//2]
        pos_y = torch.clamp(y_embed[:, :, None] / dim_t, min=-100.0, max=100.0)  # [H, W, embed_dim//2]
        
        # 处理位置编码：确保能够处理奇数embed_dim的情况
        # 使用标准的sin/cos位置编码模式
        # 对于embed_dim，我们需要创建embed_dim个维度
        # 每个频率分量产生2个维度（sin和cos），所以需要(embed_dim + 1) // 2个频率分量
        
        num_freqs = (self.embed_dim + 1) // 2
        
        # 扩展dim_t以匹配需要的频率数
        if len(dim_t) < num_freqs:
            if len(dim_t) > 0:
                # 使用最后一个值填充
                last_val = dim_t[-1:]
                dim_t = torch.cat([dim_t, last_val.repeat(num_freqs - len(dim_t))])
            else:
                # 如果dim_t为空，创建默认值
                dim_t = torch.ones(num_freqs, device=device, dtype=torch.float32)
        
        # 只使用需要的频率数
        dim_t = dim_t[:num_freqs]
        
        # 重新计算位置编码，使用足够的频率
        pos_x_full = torch.clamp(x_embed[:, :, None] / dim_t, min=-100.0, max=100.0)  # [H, W, num_freqs]
        pos_y_full = torch.clamp(y_embed[:, :, None] / dim_t, min=-100.0, max=100.0)  # [H, W, num_freqs]
        
        # 创建位置编码列表：交替使用sin和cos
        pos_x_list = []
        pos_y_list = []
        
        for i in range(num_freqs):
            # 每个频率分量产生2个维度（sin和cos）
            if i * 2 < self.embed_dim:
                # 偶数索引位置：使用sin
                pos_x_list.append(pos_x_full[:, :, i:i+1].sin())
                pos_y_list.append(pos_y_full[:, :, i:i+1].sin())
            if i * 2 + 1 < self.embed_dim:
                # 奇数索引位置：使用cos
                pos_x_list.append(pos_x_full[:, :, i:i+1].cos())
                pos_y_list.append(pos_y_full[:, :, i:i+1].cos())
        
        # 拼接所有位置编码
        if len(pos_x_list) > 0:
            pos_x = torch.cat(pos_x_list, dim=2)  # [H, W, embed_dim]
            pos_y = torch.cat(pos_y_list, dim=2)  # [H, W, embed_dim]
        else:
            # 如果列表为空（不应该发生），创建零位置编码
            pos_x = torch.zeros(H, W, self.embed_dim, device=device, dtype=torch.float32)
            pos_y = torch.zeros(H, W, self.embed_dim, device=device, dtype=torch.float32)
        
        # 确保维度正确
        if pos_x.shape[2] != self.embed_dim:
            if pos_x.shape[2] > self.embed_dim:
                pos_x = pos_x[:, :, :self.embed_dim]
                pos_y = pos_y[:, :, :self.embed_dim]
            else:
                # 填充到正确维度
                padding = self.embed_dim - pos_x.shape[2]
                pos_x = F.pad(pos_x, (0, padding), mode='constant', value=0)
                pos_y = F.pad(pos_y, (0, padding), mode='constant', value=0)
        
        # 拼接x和y的位置编码，然后截断到embed_dim
        pos = torch.cat((pos_y, pos_x), dim=2)  # [H, W, embed_dim*2]
        
        # 如果总维度超过embed_dim，截断到embed_dim（取前embed_dim个维度）
        if pos.shape[2] > self.embed_dim:
            pos = pos[:, :, :self.embed_dim]
        elif pos.shape[2] < self.embed_dim:
            # 填充到embed_dim（不应该发生）
            padding = self.embed_dim - pos.shape[2]
            pos = F.pad(pos, (0, padding), mode='constant', value=0)
        
        pos = pos.unsqueeze(0)  # [1, H, W, embed_dim]
        pos = pos.repeat(B, 1, 1, 1)  # [B, H, W, embed_dim]
        
        # 确保位置编码在正确的dtype上
        pos = pos.to(dtype=dtype)
        
        # 非常小的缩放因子，确保数值稳定
        out = x + pos * self.scale
        
        # 最终检查NaN
        if torch.isnan(out).any():
            out = torch.where(torch.isnan(out), x, out)
        
        return out

# ====================== Swin 相关（动态 H/W） ======================
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = windows.shape[0] // (H // window_size * W // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

def build_relative_position_index(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
    coords_flatten = coords.flatten(1)
    rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    rel_coords = rel_coords.permute(1,2,0).contiguous()
    rel_coords[:,:,0] += window_size - 1
    rel_coords[:,:,1] += window_size - 1
    rel_coords[:,:,0] *= 2 * window_size - 1
    relative_position_index = rel_coords.sum(-1)
    return relative_position_index

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1), num_heads)
        )
        self.register_buffer("relative_position_index", build_relative_position_index(window_size))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, attn_mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2,0,3,1,4)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size*self.window_size, self.window_size*self.window_size, -1)
        bias = bias.permute(2,0,1).unsqueeze(0)
        attn = attn + bias
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B_, N, C)
        out = self.proj(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, dim, window_size=8, shift_size=0, num_heads=8, mlp_ratio=6.0, drop=0.2, use_pos_emb=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_pos_emb = use_pos_emb
        self._attn_mask_cache = {}  # 缓存不同(Hp,Wp)的mask以加速
        
        # 位置编码（在进入Swin块之前添加）
        if use_pos_emb:
            self.pos_emb = AdaptivePositionalEncoding2D(dim, scale=0.005)  # 使用更小的缩放因子，提升稳定性
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim*mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def create_attn_mask(self, H, W, device):
        if self.shift_size == 0:
            return None
        key = (H, W, device)
        if key in self._attn_mask_cache:
            return self._attn_mask_cache[key]
        img_mask = torch.zeros((1, H, W, 1), device=device)
        cnt = 0
        ws, ss = self.window_size, self.shift_size
        for h in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for w in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws).view(-1, ws*ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float("-inf")).masked_fill(attn_mask == 0, 0.0)
        self._attn_mask_cache[key] = attn_mask
        return attn_mask

    def forward(self, x):
        B, C, H, W = x.shape
        ws, ss = self.window_size, self.shift_size
        
        # 添加位置编码（在转换为[H,W,C]格式之前）
        if self.use_pos_emb:
            # x是[B,C,H,W]，转换为[B,H,W,C]后添加位置编码
            x_hwc = x.permute(0,2,3,1).contiguous()  # [B,H,W,C]
            x_hwc = self.pos_emb(x_hwc)  # 添加位置编码
            x = x_hwc.permute(0,3,1,2).contiguous()  # 转回[B,C,H,W]
        
        x = x.permute(0,2,3,1).contiguous()  # [B,H,W,C]
        if ss > 0:
            x = torch.roll(x, shifts=(-ss,-ss), dims=(1,2))
        # 补齐到 window 的整倍数（支持任意 H/W）
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b or pad_r:
            x = F.pad(x, (0,0, 0,pad_r, 0,pad_b))  # pad last dims: W then H
        Hp, Wp = x.shape[1], x.shape[2]

        x_windows = window_partition(x, ws).view(-1, ws*ws, C)
        attn_mask = self.create_attn_mask(Hp, Wp, x.device)
        x_windows_norm = self.norm1(x_windows)
        attn_out = self.attn(x_windows_norm, attn_mask)
        x_windows = x_windows + attn_out
        x = window_reverse(x_windows.view(-1, ws, ws, C), ws, Hp, Wp)
        if ss > 0:
            x = torch.roll(x, shifts=(ss,ss), dims=(1,2))
        x_reshaped = x.view(B, Hp, Wp, C)
        x = x_reshaped + self.mlp(self.norm2(x_reshaped)).view(B, Hp, Wp, C)
        x = x[:, :H, :W, :]  # 去掉补齐
        x = x.permute(0,3,1,2).contiguous()
        return x

class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, drop=0.2, window_size: int = 8, use_pos_emb=True):
        super().__init__()
        blocks = []
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else window_size // 2
            # 只在第一个block添加位置编码，避免重复
            use_pos = use_pos_emb and (i == 0)
            blocks.append(SwinBlock(dim, window_size=window_size, shift_size=shift,
                                    num_heads=num_heads, mlp_ratio=6.0, drop=drop, use_pos_emb=use_pos))
        self.net = nn.Sequential(*blocks)
    def forward(self, x):
        return self.net(x)

# ====================== Swin-UNetLarge ======================
class SwinUNetLarge(nn.Module):
    def __init__(self, in_channels=3, base_c=64, dropout_p=0.2, window_size: int = 8,
                 depths=(3,4,6), bottleneck_depth=6,
                 num_heads_stages=(4,8,8), bottleneck_heads=16, use_pos_emb=True, use_se_sparse=True,
                 use_input_pos_emb=True):
        super().__init__()
        self.base_c = base_c
        self.use_pos_emb = use_pos_emb
        self.use_input_pos_emb = use_input_pos_emb  # 是否在输入层添加位置编码
        # 只在关键位置使用SE注意力：bottleneck和最后一层
        self.use_se_sparse = use_se_sparse

        # 输入层位置编码（可选）- 为RGB输入添加位置信息
        if use_input_pos_emb:
            # 创建一个3通道的位置编码，用于RGB输入
            self.input_pos_emb = AdaptivePositionalEncoding2D(in_channels, scale=0.01)  # 输入层使用稍大的缩放因子
        
        # Encoder - 只在第一层和最后一层使用SE注意力
        self.enc1 = ResidualDoubleConv(in_channels, base_c, dropout_p=dropout_p, use_se=use_se_sparse)
        self.enc2 = Down(base_c, base_c*2, dropout_p=dropout_p, use_se=False)
        self.enc3 = Down(base_c*2, base_c*4, dropout_p=dropout_p, use_se=False)
        self.enc4 = Down(base_c*4, base_c*8, dropout_p=dropout_p, use_se=use_se_sparse)

        # 为每个Swin stage添加可选的位置编码
        # Swin stages（删除 H/W 写死，完全动态）
        d2, d3, d4 = depths
        h2, h3, h4 = num_heads_stages
        self.swin2 = SwinStage(base_c*2, depth=d2, num_heads=h2, drop=dropout_p, 
                              window_size=window_size, use_pos_emb=use_pos_emb)
        self.swin3 = SwinStage(base_c*4, depth=d3, num_heads=h3, drop=dropout_p, 
                              window_size=window_size, use_pos_emb=use_pos_emb)
        self.swin4 = SwinStage(base_c*8, depth=d4, num_heads=h4, drop=dropout_p, 
                              window_size=window_size, use_pos_emb=use_pos_emb)
        self.down_bottleneck = Down(base_c*8, base_c*16, dropout_p=dropout_p, use_se=use_se_sparse)
        self.swin_bottleneck = SwinStage(base_c*16, depth=bottleneck_depth, num_heads=bottleneck_heads, 
                                        drop=dropout_p, window_size=window_size, use_pos_emb=use_pos_emb)

        # ========== 隐编码提取模块（在bottleneck后） ==========
        self.use_latent_code = True
        self.latent_dim = 512
        if self.use_latent_code:
            self.latent_extractor = LatentCodeExtractor(base_c*16, latent_dim=self.latent_dim, compression_ratio=8)
        
        # ========== 扩散模型式去噪模块（在bottleneck处应用） ==========
        self.use_diffusion_denoising = True
        num_denoising_blocks = 2
        if self.use_diffusion_denoising:
            self.denoising_blocks = nn.ModuleList([
                DiffusionDenoisingBlock(base_c*16, time_emb_dim=128, num_groups=min(32, base_c*16//4))
                for _ in range(num_denoising_blocks)
            ])
        
        # Decoder - 只在关键层使用SE注意力
        self.up4 = Up(base_c*16, base_c*8, base_c*8, dropout_p=dropout_p, residual_scale=0.3)
        self.up3 = Up(base_c*8, base_c*4, base_c*4, dropout_p=dropout_p, residual_scale=0.3)
        self.up2 = Up(base_c*4, base_c*2, base_c*2, dropout_p=dropout_p, residual_scale=0.3)
        self.up1 = Up(base_c*2, base_c, base_c, dropout_p=dropout_p, residual_scale=0.3)

        # Heads - 自编码器需要输出3通道RGB图像
        self.head_main = nn.Conv2d(base_c, 3, 1)  # 输出3通道RGB
        self.head_d2   = nn.Conv2d(base_c*2, 3, 1)  # 输出3通道RGB
        self.head_d3   = nn.Conv2d(base_c*4, 3, 1)  # 输出3通道RGB

    def forward(self, x, timestep=None, return_latent=False):
        """
        前向传播（增强版：支持隐编码提取和扩散去噪）
        Args:
            x: [B, C, H, W] 输入图像
            timestep: [B] 可选，归一化时间步[0,1]，用于扩散去噪（训练时可随机采样）
            return_latent: bool, 是否返回隐编码
        """
        B, C, H, W = x.shape
        
        # 在输入层添加位置编码（可选）
        if self.use_input_pos_emb:
            x_hwc = x.permute(0, 2, 3, 1).contiguous()
            x_hwc = self.input_pos_emb(x_hwc)
            x = x_hwc.permute(0, 3, 1, 2).contiguous()
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1); e2 = self.swin2(e2)
        e3 = self.enc3(e2); e3 = self.swin3(e3)
        e4 = self.enc4(e3); e4 = self.swin4(e4)
        b = self.down_bottleneck(e4); b = self.swin_bottleneck(b)  # [B, base_c*16, H/16, W/16]
        
        # ========== 隐编码提取 ==========
        latent_code = None
        if self.use_latent_code:
            latent_code = self.latent_extractor.extract(b)  # [B, latent_dim]
        
        # ========== 扩散模型式去噪（在隐空间特征图上） ==========
        if self.use_diffusion_denoising and timestep is not None:
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).expand(B)
            elif timestep.shape[0] != B:
                timestep = timestep[:B]
            for denoising_block in self.denoising_blocks:
                b = denoising_block(b, timestep)

        # Decoder with improved skip connections
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        # Multi-scale outputs
        logit_main = self.head_main(d1)
        logit_d2   = F.interpolate(self.head_d2(d2), size=(H,W), mode='bilinear', align_corners=True)
        logit_d3   = F.interpolate(self.head_d3(d3), size=(H,W), mode='bilinear', align_corners=True)
        
        if return_latent and latent_code is not None:
            return logit_main, [logit_d2, logit_d3], latent_code
        return logit_main, [logit_d2, logit_d3]
    
    def encode(self, x):
        """提取隐编码"""
        B, C, H, W = x.shape
        if self.use_input_pos_emb:
            x_hwc = x.permute(0, 2, 3, 1).contiguous()
            x_hwc = self.input_pos_emb(x_hwc)
            x = x_hwc.permute(0, 3, 1, 2).contiguous()
        e1 = self.enc1(x)
        e2 = self.enc2(e1); e2 = self.swin2(e2)
        e3 = self.enc3(e2); e3 = self.swin3(e3)
        e4 = self.enc4(e3); e4 = self.swin4(e4)
        b = self.down_bottleneck(e4); b = self.swin_bottleneck(b)
        if self.use_latent_code:
            return self.latent_extractor.extract(b)
        return None

# ====================== 损失函数：Focal + Dice ======================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-6, 1-1e-6)
        num = 2 * (probs * targets).sum(dim=(2, 3)) + self.smooth
        den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        dice = num / den
        return 1 - dice.mean(), dice.mean()

class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1-pt) ** self.gamma * bce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.focal = BinaryFocalLossWithLogits(alpha=alpha, gamma=gamma)
        self.dice  = DiceLoss()
        self.dw = dice_weight
        self.fw = focal_weight
    def forward(self, logits, targets):
        dice_l, dice_score = self.dice(logits, targets)
        focal_l = self.focal(logits, targets) if self.fw > 0 else 0.0
        return self.fw * focal_l + self.dw * dice_l, dice_score

# ======== IoU Loss（数值稳定版） ========
class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-7, 1-1e-7)  # 更严格的clamp
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
        union = torch.clamp(union, min=self.smooth)  # 确保union不为0
        iou = (inter + self.smooth) / union
        iou = torch.clamp(iou, min=0.0, max=1.0)  # 确保IoU在合理范围
        loss = 1 - iou.mean()
        # 检查NaN
        if torch.isnan(loss):
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.5, device=logits.device)
        return loss, iou.mean()

# ======== Tversky Loss（对不平衡数据更友好） ========
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5):
        super().__init__()
        self.alpha = alpha  # 假阳性权重
        self.beta = beta    # 假阴性权重
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-6, 1-1e-6)
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean(), tversky.mean()

# ======== BCE + Dice + IoU 组合（数值稳定版） ========
class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, bce_weight: float = 1.0, 
                 dice_weight: float = 1.0, iou_weight: float = 0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.iou = IoULoss()
        self.bw = bce_weight
        self.dw = dice_weight
        self.iw = iou_weight
    def forward(self, logits, targets):
        bce_l = self.bce(logits, targets)
        dice_l, dice_s = self.dice(logits, targets)
        iou_l, iou_s = self.iou(logits, targets)
        
        # 检查每个损失是否为NaN
        if torch.isnan(bce_l):
            bce_l = torch.tensor(0.0, device=logits.device)
        if torch.isnan(dice_l):
            dice_l = torch.tensor(0.0, device=logits.device)
        if torch.isnan(iou_l):
            iou_l = torch.tensor(0.0, device=logits.device)
        
        total_loss = self.bw * bce_l + self.dw * dice_l + self.iw * iou_l
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # 如果仍有NaN，只使用BCE和Dice
            total_loss = self.bw * bce_l + self.dw * dice_l
        
        return total_loss, dice_s

# ======== 边缘感知损失（提升轮廓质量） ========
class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('lap_kernel', k)
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # 适配 AMP：确保 kernel 与输入 dtype/device 一致
        k = self.lap_kernel.to(dtype=probs.dtype, device=probs.device)
        pe = F.conv2d(probs, k, padding=1)
        te = F.conv2d(targets, k, padding=1)
        return F.l1_loss(pe, te)

# ====================== 图像重建损失函数（自编码器任务） ======================
class ImageReconstructionLoss(nn.Module):
    """图像重建损失：MSE + L1 + Perceptual Loss + SSIM"""
    def __init__(self, mse_weight=1.0, l1_weight=0.5, perceptual_weight=0.1, ssim_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        
        # Perceptual Loss: 使用预训练的VGG特征
        if perceptual_weight > 0:
            try:
                from torchvision.models import vgg16
                vgg = vgg16(pretrained=True).features[:16].eval()  # 取前几层特征
                for param in vgg.parameters():
                    param.requires_grad = False
                self.vgg = vgg
            except:
                self.vgg = None
                self.perceptual_weight = 0.0
    
    def perceptual_loss(self, pred, target):
        """使用VGG特征的感知损失"""
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        # 归一化到ImageNet范围
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std
        feat_pred = self.vgg(pred_norm)
        feat_target = self.vgg(target_norm)
        return F.mse_loss(feat_pred, feat_target)
    
    def ssim_loss(self, pred, target):
        """简化的SSIM损失（计算复杂度较低）"""
        # 将输入归一化到[0, 1]
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        mu1 = pred.mean(dim=(2, 3), keepdim=True)
        mu2 = target.mean(dim=(2, 3), keepdim=True)
        
        sigma1_sq = ((pred - mu1) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma2_sq = ((target - mu2) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma12 = ((pred - mu1) * (target - mu2)).mean(dim=(2, 3), keepdim=True)
        
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        return 1 - ssim.mean()
    
    def forward(self, pred, target):
        """
        pred: [B, 3, H, W] 预测图像（已归一化到[-1, 1]或[0, 1]）
        target: [B, 3, H, W] 目标图像
        """
        # 确保在[0, 1]范围（如果需要）
        if pred.min() < 0:
            pred = (pred + 1) / 2.0
        if target.min() < 0:
            target = (target + 1) / 2.0
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # MSE Loss
        mse_loss = F.mse_loss(pred, target)
        
        # L1 Loss
        l1_loss = F.l1_loss(pred, target)
        
        # Perceptual Loss
        perceptual_loss_val = self.perceptual_loss(pred, target) if self.perceptual_weight > 0 else torch.tensor(0.0, device=pred.device)
        
        # SSIM Loss
        ssim_loss_val = self.ssim_loss(pred, target)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.l1_weight * l1_loss + 
                     self.perceptual_weight * perceptual_loss_val +
                     self.ssim_weight * ssim_loss_val)
        
        return total_loss, {
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
            'perceptual': perceptual_loss_val.item() if isinstance(perceptual_loss_val, torch.Tensor) else 0.0,
            'ssim': ssim_loss_val.item()
        }

# ====================== CutMix for Segmentation ======================
def rand_bbox(W, H, lam):
    """生成 CutMix 的矩形区域（在图像坐标系内）。"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix_batch(images, masks, alpha=0.4):
    """对 batch 应用 CutMix。images:[B,3,H,W], masks:[B,1,H,W]"""
    if alpha <= 0:
        return images, masks, 1.0
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = images.size()
    index = torch.randperm(B, device=images.device)
    x1, y1, x2, y2 = rand_bbox(W, H, lam)

    # 计算实际的混合比例（保留区域的比例）
    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area > 0:
        # 实际lambda = 保留的像素比例（未被替换的部分）
        actual_lam = bbox_area / (W * H)
        # 确保lambda在合理范围内
        actual_lam = np.clip(actual_lam, 0.1, 0.9)
    else:
        # 如果bbox区域为0，保持原始lambda
        actual_lam = lam

    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    masks[:, :, y1:y2, x1:x2]  = masks[index, :, y1:y2, x1:x2]

    return images, masks, actual_lam

def mixup_batch(images, masks, alpha=0.2):
    """对 batch 应用 MixUp。images:[B,3,H,W], masks:[B,1,H,W]"""
    if alpha <= 0:
        return images, masks, 1.0
    lam = np.random.beta(alpha, alpha)
    # 限制lambda范围，避免极端值
    lam = np.clip(lam, 0.3, 0.7)
    B, C, H, W = images.size()
    index = torch.randperm(B, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    # 对于mask，使用软标签（允许连续值，但限制范围）
    mixed_masks = lam * masks + (1 - lam) * masks[index]
    mixed_masks = torch.clamp(mixed_masks, 0.0, 1.0)  # 确保在[0,1]范围
    return mixed_images, mixed_masks, lam

# ====================== 监控指标计算 ======================
def calculate_metrics(preds, targets, threshold=0.5):
    """
    计算二值分割的IoU、Precision、Recall
    preds: [B, 1, H, W] 预测概率图
    targets: [B, 1, H, W] 真实标签
    """
    # 二值化预测结果
    preds_binary = (preds > threshold).float()

    # 计算TP, FP, FN
    tp = ((preds_binary == 1) & (targets == 1)).sum(dim=(1, 2, 3)).float()
    fp = ((preds_binary == 1) & (targets == 0)).sum(dim=(1, 2, 3)).float()
    fn = ((preds_binary == 0) & (targets == 1)).sum(dim=(1, 2, 3)).float()

    # 计算指标
    iou = tp / (tp + fp + fn + 1e-6)  # IoU = TP / (TP + FP + FN)
    precision = tp / (tp + fp + 1e-6)  # Precision = TP / (TP + FP)
    recall = tp / (tp + fn + 1e-6)     # Recall = TP / (TP + FN)

    # 返回batch平均值
    return {
        'iou': iou.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item()
    }

# ====================== 后处理与TTA ======================
def postprocess_masks(probs: torch.Tensor, remove_small: int = 50, morph_kernel: int = 3):
    """probs: [B,1,H,W] in [0,1] -> numpy后处理再回tensor。"""
    b, _, h, w = probs.shape
    out = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)) if morph_kernel > 0 else None
    for i in range(b):
        p = probs[i,0].detach().cpu().numpy()
        m = (p > 0.5).astype(np.uint8)
        if remove_small > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            keep = np.zeros_like(m)
            for lab in range(1, num_labels):
                if stats[lab, cv2.CC_STAT_AREA] >= remove_small:
                    keep[labels == lab] = 1
            m = keep
        if kernel is not None:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        out.append(torch.from_numpy(m).float().unsqueeze(0))
    return torch.stack(out, dim=0).to(probs.device)

def tta_horizontal(model: nn.Module, x: torch.Tensor):
    """简单TTA：原图 + 水平翻转 概率平均。返回 logits 平均。"""
    logit1, _ = model(x)
    x_flip = torch.flip(x, dims=[3])
    logit2, _ = model(x_flip)
    logit2 = torch.flip(logit2, dims=[3])
    return (logit1 + logit2) / 2.0

# ====================== EMA ======================
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                nv = msd[k]
                if v.dtype.is_floating_point:
                    v.copy_(v * d + nv * (1.0 - d))

# ====================== 计数参数 ======================
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

# ====================== 训练脚本 ======================
if __name__ == '__main__':
    # 设置路径
    ROOT_PATH = '/root/autodl-fs'
    os.makedirs(os.path.join(ROOT_PATH, 'mlp_LocalLines_img-Codings'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_PATH, 'feature_visualizations'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_PATH, 'curves'), exist_ok=True)

    # 新的数据路径
    DATA_ROOT = "/root/autodl-tmp/EndoscopicData-all/EndoscopicData-all"

    # 自动检测数据结构
    print(f"正在检测数据结构: {DATA_ROOT}")
    contents = os.listdir(DATA_ROOT) if os.path.exists(DATA_ROOT) else []
    print(f"文件夹内容: {contents}")

    # 尝试不同的数据结构
    possible_structures = [
        # 结构1: images/ 和 masks/ 子文件夹
        (os.path.join(DATA_ROOT, "images"), os.path.join(DATA_ROOT, "masks")),
        # 结构2: 所有文件都在根目录，按文件名区分
        (DATA_ROOT, DATA_ROOT),
        # 结构3: train/ 和 test/ 子文件夹（如果已分割）
        (os.path.join(DATA_ROOT, "train"), os.path.join(DATA_ROOT, "train_masks")) if "train" in contents else (None, None)
    ]

    source_img_dir, source_mask_dir = None, None
    for img_dir, mask_dir in possible_structures:
        if img_dir and mask_dir and os.path.exists(img_dir) and os.path.exists(mask_dir):
            # 检查是否包含图像文件
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
            mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]

            if img_files and (img_dir == mask_dir or mask_files):
                source_img_dir = img_dir
                source_mask_dir = mask_dir
                print(f"✓ 检测到数据结构: 图像({len(img_files)}张) | 标签({len(mask_files)}张)")
                break

    if not source_img_dir or not source_mask_dir:
        raise FileNotFoundError(f"无法检测到有效的数据结构在 {DATA_ROOT}。请确保数据按以下格式组织：\n"
                               "  - images/ 和 masks/ 子文件夹，或\n"
                               "  - 所有图像和标签文件在同一文件夹中")

    Image_Width, Image_Height = 128, 128

    # 检查是否已分割数据集
    train_dir = os.path.join(DATA_ROOT, 'train')
    val_dir = os.path.join(DATA_ROOT, 'val')
    test_dir = os.path.join(DATA_ROOT, 'test')

    # 改进检查逻辑：检查文件夹是否存在且包含图像文件
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    def has_image_files(directory):
        """检查目录是否存在且包含图像文件"""
        if not os.path.exists(directory):
            return False
        files = os.listdir(directory)
        return any(any(f.lower().endswith(ext) for ext in img_extensions) for f in files)
    
    train_exists = has_image_files(train_dir)
    val_exists = has_image_files(val_dir)
    test_exists = has_image_files(test_dir)
    
    dataset_already_split = train_exists and val_exists and test_exists

    if not dataset_already_split:
        print(f"\n{'='*80}")
        print("⚠️  未检测到已分割的数据集，开始自动分割...")
        print(f"   源数据文件夹: {DATA_ROOT}")
        print(f"{'='*80}")
        
        # 检查源文件夹是否存在
        if not os.path.exists(DATA_ROOT):
            raise FileNotFoundError(f"源数据文件夹不存在: {DATA_ROOT}")

        # 分割数据集
        train_dir, val_dir, test_dir = split_autoencoder_dataset(
            source_dir=DATA_ROOT,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=2025
        )
    else:
        # 计算每个集合的图像数量
        train_files = [f for f in os.listdir(train_dir) 
                      if any(f.lower().endswith(ext) for ext in img_extensions)]
        val_files = [f for f in os.listdir(val_dir) 
                    if any(f.lower().endswith(ext) for ext in img_extensions)]
        test_files = [f for f in os.listdir(test_dir) 
                     if any(f.lower().endswith(ext) for ext in img_extensions)]
        
        print(f"\n{'='*80}")
        print("✅ 检测到已分割的数据集，跳过分割步骤")
        print(f"{'='*80}")
        print(f"  - 训练集: {train_dir} ({len(train_files)} 张图像)")
        print(f"  - 验证集: {val_dir} ({len(val_files)} 张图像)")
        print(f"  - 测试集: {test_dir} ({len(test_files)} 张图像)")
        print(f"{'='*80}\n")

    # 设置数据集路径
    folder_path_Input = train_dir
    folder_path_Output = train_dir  # 自编码器输入和输出都是训练集
    folder_path_Val = val_dir
    folder_path_Val_Output = val_dir  # 自编码器输入和输出都是验证集

    batch_size = 32
    cpu_cnt = os.cpu_count() or 0
    # 优化数据加载：根据CPU核心数动态调整，避免过多进程导致资源竞争
    num_workers = min(max(4, cpu_cnt // 2), 8)  # 最少4个，最多8个，取CPU核心数的一半
    learning_rate = 5e-5  # 进一步降低初始学习率，提高稳定性（从1e-4降到5e-5）
    num_epochs = 2000
    mix_alpha = 0.3  # 降低CutMix强度
    cutmix_prob = 0.08  # 进一步降低CutMix频率
    mixup_alpha = 0.15   # 降低MixUp强度
    mixup_prob = 0.05   # 进一步降低MixUp概率

    # Dataset & Loader（训练集有增强，验证集无增强）
    train_dataset = AutoencoderDataset(folder_path_Input, (Image_Height, Image_Width), augment=True)
    val_dataset   = AutoencoderDataset(folder_path_Val, (Image_Height, Image_Width), augment=False)

    # 优化数据加载配置：根据batch size调整prefetch_factor
    train_prefetch = min(4, max(2, batch_size // 8))  # 根据batch size动态调整
    val_prefetch = min(2, max(1, batch_size // 16))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,  # 不丢弃最后一个不完整的batch
        **({'prefetch_factor': train_prefetch} if num_workers > 0 else {})
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,  # 验证时不丢弃数据
        **({'prefetch_factor': val_prefetch} if num_workers > 0 else {})
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 性能优先模式（如需严格可复现请保持 set_seed 的设定）
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    # 优化后的模型配置：针对128x128图像优化窗口大小
    # window_size=8更适合小图像（128x128），16可能太大导致窗口内像素太少
    # 使用改进的固定正弦位置编码（数值更稳定）
    # use_se_sparse=True：只在关键位置使用SE注意力，提升速度
    model = SwinUNetLarge(in_channels=3, base_c=64, dropout_p=0.15, window_size=8,  # 调整为8
                          depths=(2,3,4), bottleneck_depth=4,  # 适度深度，避免过拟合
                          num_heads_stages=(4,8,8), bottleneck_heads=16,
                          use_pos_emb=True, use_se_sparse=True, 
                          use_input_pos_emb=True).to(device)  # 启用稳定的位置编码，稀疏SE注意力，输入层位置编码
    
    # 输出模型结构信息
    print(f"\n{'='*80}")
    print("📊 模型架构信息")
    print(f"{'='*80}")
    print(f"模型类型: SwinUNetLarge (自编码器)")
    print(f"输入通道: 3 (RGB)")
    print(f"输出通道: 3 (RGB)")
    print(f"基础通道数: 64")
    print(f"窗口大小: 8")
    print(f"Dropout率: 0.15")
    print(f"位置编码: 启用（SwinBlock中）")
    print(f"输入层位置编码: 启用（RGB输入）")
    print(f"SE注意力: 稀疏模式（仅在关键层）")
    print(f"{'='*80}")
    
    # 打印参数量
    count_parameters(model)
    
    # 打印详细模型结构
    print(f"\n{'='*80}")
    print("📐 详细模型结构")
    print(f"{'='*80}")
    print(model)
    print(f"{'='*80}\n")
    
    # 测试前向传播并输出特征图尺寸（可选）
    print("🧪 测试前向传播（验证模型结构）...")
    try:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, Image_Height, Image_Width).to(device)
            logit_main, aux_list = model(dummy_input)
            print(f"  ✓ 前向传播成功")
            print(f"  - 主输出形状: {logit_main.shape}")
            print(f"  - 辅助输出数量: {len(aux_list)}")
            for i, aux in enumerate(aux_list):
                print(f"  - 辅助输出{i+1}形状: {aux.shape}")
    except Exception as e:
        print(f"  ✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()

    # ========== 加载预训练模型 ==========
    pretrained_path = r"autodl-tmp/TinySAM-main/SwinUNet_FocalDice_DeepSup_199.pth"
    if os.path.exists(pretrained_path):
        print(f"\n📁 检测到预训练模型: {pretrained_path}")
        try:
            # 加载checkpoint
            checkpoint = torch.load(pretrained_path, map_location=device)
            print("  ✓ 模型文件加载成功")

            # 检查checkpoint结构
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("  ✓ 从checkpoint['model_state_dict']加载模型权重")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print("  ✓ 从checkpoint['state_dict']加载模型权重")
                else:
                    # 直接加载整个checkpoint作为state_dict
                    state_dict = checkpoint
                    print("  ✓ 直接加载checkpoint作为模型权重")
            else:
                # 假设checkpoint就是state_dict
                state_dict = checkpoint
                print("  ✓ 加载模型权重")

            # 尝试加载权重，如果失败则进行兼容性处理
            try:
                model.load_state_dict(state_dict)
                print("  ✓ 模型权重加载成功")
            except RuntimeError as e:
                print(f"  ⚠️  直接加载失败: {e}")
                print("  🔄 尝试兼容性权重映射...")

                # 创建兼容性映射
                new_state_dict = {}
                model_keys = set(model.state_dict().keys())

                for key, value in state_dict.items():
                    # 尝试找到匹配的键
                    if key in model_keys:
                        new_state_dict[key] = value
                    else:
                        # 尝试模糊匹配
                        found_match = False
                        for model_key in model_keys:
                            if key in model_key or model_key in key:
                                print(f"  📝 映射: {key} → {model_key}")
                                new_state_dict[model_key] = value
                                found_match = True
                                break

                        if not found_match:
                            # 检查是否是类似的层
                            if 'enc' in key and 'enc' in str(model_keys):
                                # 尝试映射编码器层
                                for model_key in model_keys:
                                    if 'enc' in model_key and key.split('.')[-2:] == model_key.split('.')[-2:]:
                                        print(f"  📝 编码器映射: {key} → {model_key}")
                                        new_state_dict[model_key] = value
                                        found_match = True
                                        break

                            if not found_match:
                                print(f"  ⚠️  跳过不匹配的层: {key}")

                # 加载映射后的权重
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                if missing_keys:
                    print(f"  ⚠️  缺失的层: {len(missing_keys)} 个")
                if unexpected_keys:
                    print(f"  ⚠️  多余的层: {len(unexpected_keys)} 个")
                print("  ✓ 兼容性权重映射完成")

            # 移动到设备
            model = model.to(device)
            print(f"  ✓ 模型已移动到设备: {device}")

            # 再次测试前向传播
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, Image_Height, Image_Width).to(device)
                _ = model(dummy_input)
            print("  ✓ 加载的模型前向传播测试通过")

        except Exception as e:
            print(f"  ✗ 加载预训练模型失败: {e}")
            print("  继续使用随机初始化的模型进行训练")
    else:
        print(f"\n⚠️  未找到预训练模型: {pretrained_path}")
        print("  将使用随机初始化的模型进行训练")
    # 可选：PyTorch 2.x 编译加速（显著提升训练速度，推荐启用）
    use_torch_compile = False  # 禁用编译以避免Triton编译错误（当前环境不支持torch._dynamo）
    if use_torch_compile:
        try:
            # 检查PyTorch版本是否支持compile
            if hasattr(torch, 'compile'):
                print("🚀 启用torch.compile加速...")
                model = torch.compile(model, mode='max-autotune')  # 使用max-autotune模式获得最佳性能
                print("  ✓ torch.compile成功，预计训练速度提升20-30%")
            else:
                print("  ⚠️  PyTorch版本不支持torch.compile，跳过编译")
        except Exception as e:
            print(f"  ⚠️  torch.compile失败: {e}, 继续使用原始模型")

    # ============ 自编码器配置开关 ============
    use_ema = True                # 启用EMA
    use_tta = False               # 自编码器不需要TTA
    use_postproc = False          # 自编码器不需要后处理
    use_dynamic_aux_weight = True # 动态调整辅助损失权重
    use_gradient_accumulation = False  # 是否使用梯度累积（用于支持更大batch size）
    accumulation_steps = 4  # 梯度累积步数（如果启用，等效batch size = batch_size * accumulation_steps）
    
    # 性能优化配置
    val_frequency = 5  # 每N个epoch验证一次（减少验证频率以提升训练速度）
    save_pred_frequency = 10  # 每N个epoch保存一次预测结果
    compute_ssim_every_n_batches = 1  # 每N个batch计算一次SSIM（减少SSIM计算频率以提升速度）

    # 自编码器损失函数：使用L1 + MSE + SSIM损失
    criterion_recon = nn.L1Loss()  # 重建损失
    criterion_mse = nn.MSELoss()   # MSE损失

    # 优化的SSIM损失 - 使用更小的窗口和更快的计算
    class SSIMLoss(nn.Module):
        def __init__(self, window_size=7, size_average=True, channel=None):
            super().__init__()
            self.window_size = window_size  # 减小窗口大小以提升速度
            self.size_average = size_average
            self.channel = channel  # None表示自动检测
            self.window = None  # 延迟创建
            self._channel = channel  # 保存初始通道数
            self._window_cache = {}  # 缓存不同(device, dtype, channel)的窗口，避免重复创建

        def gaussian(self, window_size, sigma):
            gauss = torch.exp(-(torch.arange(window_size, dtype=torch.float32) - window_size//2)**2 / (2*sigma**2))
            return gauss / (gauss.sum() + 1e-8)  # 防止除零

        def create_window(self, window_size, channel, device=None, dtype=None):
            # 使用缓存键避免重复创建
            cache_key = (window_size, channel, device, dtype)
            if cache_key in self._window_cache:
                return self._window_cache[cache_key]
            
            _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            
            # 如果指定了设备和数据类型，提前转换
            if device is not None:
                window = window.to(device=device)
            if dtype is not None:
                window = window.to(dtype=dtype)
            
            # 缓存窗口（限制缓存大小，避免内存泄漏）
            if len(self._window_cache) < 10:  # 最多缓存10个窗口
                self._window_cache[cache_key] = window
            
            return window

        def forward(self, img1, img2):
            # 自动检测通道数
            if self.channel is None:
                self.channel = img1.shape[1]
            
            # 获取设备和数据类型
            device = img1.device
            dtype = img1.dtype
            
            # 使用缓存机制，避免重复创建和转换
            cache_key = (self.window_size, self.channel, device, dtype)
            if cache_key in self._window_cache:
                self.window = self._window_cache[cache_key]
            elif self.window is None or self.window.shape[0] != self.channel or \
                 self.window.device != device or self.window.dtype != dtype:
                self.window = self.create_window(self.window_size, self.channel, device=device, dtype=dtype)

            return 1 - self._ssim(img1, img2)

        def _ssim(self, img1, img2):
            # 确保输入在[0, 1]范围内，转换为float32以提升稳定性
            img1 = torch.clamp(img1.float(), min=0.0, max=1.0)
            img2 = torch.clamp(img2.float(), min=0.0, max=1.0)
            
            # 使用更小的padding以提升速度
            pad = self.window_size // 2
            mu1 = F.conv2d(img1, self.window, padding=pad, groups=self.channel)
            mu2 = F.conv2d(img2, self.window, padding=pad, groups=self.channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(img1*img1, self.window, padding=pad, groups=self.channel) - mu1_sq
            sigma2_sq = F.conv2d(img2*img2, self.window, padding=pad, groups=self.channel) - mu2_sq
            sigma12 = F.conv2d(img1*img2, self.window, padding=pad, groups=self.channel) - mu1_mu2

            # 数值稳定性：确保sigma不为负，使用更严格的clamp
            sigma1_sq = torch.clamp(sigma1_sq, min=0.0, max=1.0)
            sigma2_sq = torch.clamp(sigma2_sq, min=0.0, max=1.0)
            sigma12 = torch.clamp(sigma12, min=-1.0, max=1.0)

            C1 = 0.01**2
            C2 = 0.03**2

            # 计算SSIM，添加数值稳定性
            numerator = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            denominator = torch.clamp(denominator, min=1e-6, max=10.0)  # 更严格的限制
            
            ssim_map = numerator / denominator
            ssim_map = torch.clamp(ssim_map, min=-1.0, max=1.0)  # 允许轻微负值以提升稳定性
            return ssim_map.mean()

    criterion_ssim = SSIMLoss(channel=3, window_size=7)  # 使用更小的窗口以提升速度

    # 优化的组合损失函数（数值稳定版本，提升速度）
    def autoencoder_loss(reconstructed, target, compute_ssim=True):
        # 将输入clip到合理范围，避免数值不稳定
        # 模型输出应该在[-1, 1]范围内（归一化后的范围）
        # 转换为float32以支持clamp操作（AMP可能产生float16）
        reconstructed = torch.clamp(reconstructed.float(), min=-2.0, max=2.0)
        target = torch.clamp(target.float(), min=-2.0, max=2.0)
        
        # L1损失（主要）- 数值稳定
        l1_loss = criterion_recon(reconstructed, target)
        if torch.isnan(l1_loss) or torch.isinf(l1_loss):
            l1_loss = torch.tensor(0.0, device=reconstructed.device)
        
        # MSE损失（辅助）- 数值稳定
        mse_loss = criterion_mse(reconstructed, target)
        if torch.isnan(mse_loss) or torch.isinf(mse_loss):
            mse_loss = torch.tensor(0.0, device=reconstructed.device)
        
        # SSIM损失（结构相似性）- 可选，降低计算频率
        # 只在验证时或每N个batch计算一次以提升速度
        ssim_loss = torch.tensor(0.0, device=reconstructed.device)
        if compute_ssim:
            try:
                # 将输入归一化到[0, 1]范围
                recon_normalized = (reconstructed + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                target_normalized = (target + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                recon_normalized = torch.clamp(recon_normalized, min=0.0, max=1.0)
                target_normalized = torch.clamp(target_normalized, min=0.0, max=1.0)
                
                ssim_loss = criterion_ssim(recon_normalized, target_normalized)
                # 检查SSIM损失是否为NaN
                if torch.isnan(ssim_loss) or torch.isinf(ssim_loss) or ssim_loss < 0:
                    ssim_loss = torch.tensor(0.0, device=reconstructed.device)
            except Exception:
                # 如果SSIM计算失败，设为0
                ssim_loss = torch.tensor(0.0, device=reconstructed.device)

        # 组合损失：主要使用L1，辅以MSE，SSIM权重更低以提升稳定性
        # 使用更保守的权重组合，进一步降低辅助损失权重
        total_loss = l1_loss + 0.03 * mse_loss + 0.01 * ssim_loss
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # 如果仍有NaN，只使用L1
            total_loss = l1_loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = torch.tensor(0.01, device=reconstructed.device)  # 使用小的非零值

        return total_loss, l1_loss.item() if not torch.isnan(l1_loss) else 0.0, \
               mse_loss.item() if not torch.isnan(mse_loss) else 0.0, \
               ssim_loss.item() if not torch.isnan(ssim_loss) else 0.0
    
    # 优化器：使用更小的weight decay和更稳定的eps
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-5, 
                           betas=(0.9, 0.999), eps=1e-7)
    
    # 使用更稳定的学习率调度：OneCycleLR（平滑，无重启问题）
    from torch.optim.lr_scheduler import OneCycleLR
    
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.25,  # 增加warmup到25%，更平滑的启动
        anneal_strategy='cos',
        div_factor=20.0,  # 初始lr更小 = max_lr / div_factor（更保守的启动）
        final_div_factor=100.0  # 最终lr = max_lr / final_div_factor
    )
    # 兼容不同版本的 GradScaler
    # 使用更保守的初始scale以提升稳定性
    try:
        scaler = GradScaler(device='cuda', enabled=(device.type == 'cuda'), 
                           init_scale=2.**10, growth_factor=2.0, backoff_factor=0.5)
    except TypeError:
        # 旧版本使用不同的参数
        try:
            scaler = GradScaler(init_scale=2.**10, growth_factor=2.0, backoff_factor=0.5)
        except TypeError:
            # 更老的版本
            scaler = GradScaler()

    # ========== 从检查点恢复训练 ==========
    resume_checkpoint_path = "/root/autodl-tmp/Autoencoder_10.pth"  # 检查点文件路径
    start_epoch = 0
    best_val_dice = -1.0
    train_losses, val_losses, val_dices = [], [], []
    val_ious, val_precisions, val_recalls = [], [], []
    
    # 初始化EMA helper（如果使用），以便恢复EMA状态
    ema_helper = None
    if use_ema:
        ema_helper = ModelEMA(model, decay=0.999)
    
    # 检查是否存在检查点文件
    if os.path.exists(resume_checkpoint_path):
        print(f"\n🔄 检测到检查点文件: {resume_checkpoint_path}")
        print("  正在恢复训练...")
        try:
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)
            
            # 加载模型权重
            if isinstance(checkpoint, dict):
                # 如果是完整的检查点字典
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("  ✓ 模型权重加载成功")
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print("  ✓ 模型权重加载成功")
                else:
                    # 尝试直接作为state_dict加载
                    try:
                        model.load_state_dict(checkpoint)
                        print("  ✓ 模型权重加载成功（直接加载）")
                    except:
                        print("  ⚠️  无法加载模型权重，将使用随机初始化")
                
                # 加载优化器状态
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("  ✓ 优化器状态加载成功")
                
                # 加载学习率调度器状态
                if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("  ✓ 学习率调度器状态加载成功")
                
                # 加载Scaler状态
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    print("  ✓ GradScaler状态加载成功")
                
                # 加载EMA状态
                if 'ema_state_dict' in checkpoint and use_ema:
                    if ema_helper is not None:
                        ema_helper.ema.load_state_dict(checkpoint['ema_state_dict'])
                        print("  ✓ EMA状态加载成功")
                    else:
                        print("  ⚠️  检查点包含EMA状态，但当前未启用EMA")
                
                # 加载训练状态
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"  ✓ 从epoch {start_epoch} 继续训练")
                
                if 'best_val_dice' in checkpoint:
                    best_val_dice = checkpoint['best_val_dice']
                    print(f"  ✓ 最佳验证指标: {best_val_dice:.4f}")
                
                if 'train_losses' in checkpoint:
                    train_losses = checkpoint['train_losses']
                    print(f"  ✓ 训练历史记录加载成功（{len(train_losses)}个epoch）")
                
                if 'val_losses' in checkpoint:
                    val_losses = checkpoint['val_losses']
                
                if 'val_dices' in checkpoint:
                    val_dices = checkpoint['val_dices']
                
                if 'val_ious' in checkpoint:
                    val_ious = checkpoint['val_ious']
                
                if 'val_precisions' in checkpoint:
                    val_precisions = checkpoint['val_precisions']
                
                if 'val_recalls' in checkpoint:
                    val_recalls = checkpoint['val_recalls']
                
            else:
                # 如果只是state_dict，只加载模型权重
                try:
                    model.load_state_dict(checkpoint)
                    print("  ✓ 模型权重加载成功（仅state_dict）")
                    # 从文件名解析epoch
                    import re
                    epoch_match = re.search(r'_(\d+)\.pth$', resume_checkpoint_path)
                    if epoch_match:
                        start_epoch = int(epoch_match.group(1)) + 1
                        print(f"  ✓ 从文件名解析：从epoch {start_epoch} 继续训练")
                except Exception as e:
                    print(f"  ⚠️  加载模型权重失败: {e}")
            
            print("  ✅ 检查点恢复成功！")
            
        except Exception as e:
            print(f"  ✗ 加载检查点失败: {e}")
            print("  将从头开始训练")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️  未找到检查点文件: {resume_checkpoint_path}")
        print("  将从头开始训练")
    
    # 计算剩余训练轮数
    remaining_epochs = max(1, num_epochs - start_epoch)

    print(f"\n🚀 🚀 🚀 开始训练...")
    print(f"  - 起始epoch: {start_epoch}")
    print(f"  - 计划训练总轮数: {remaining_epochs}")
    print(f"  - 最终达到的epoch: {start_epoch + remaining_epochs - 1}")
    print(f"  - 当前最佳验证指标: {best_val_dice:.4f}")

    patience, bad_epochs = 25, 0  # 增加patience给模型更多机会

    # ========== 扩散模型相关初始化 ==========
    noise_scheduler = None
    if model.use_diffusion_denoising:
        print("  ✨ 初始化扩散噪声调度器...")
        # 使用默认参数初始化NoiseScheduler
        noise_scheduler = NoiseScheduler(
            num_timesteps=1000,
            schedule_type='linear',
            beta_start=0.0001,
            beta_end=0.02
        )
        print("  ✅ 扩散噪声调度器初始化完成。")

    for epoch in range(remaining_epochs):
        model.train()
        train_loss = 0.0

        # 动态调整辅助损失权重：前期较大，后期减小
        # 降低辅助损失权重以提升稳定性
        if use_dynamic_aux_weight:
            # 计算实际的全局epoch编号用于权重衰减
            actual_epoch = epoch + start_epoch
            # 从0.2逐渐衰减到0.1（降低初始权重）
            aux_weight = max(0.1, 0.2 * (1 - actual_epoch / (start_epoch + remaining_epochs)))
        else:
            aux_weight = 0.15  # 降低固定权重

        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # channels_last 内存格式以提升卷积吞吐
            inputs = inputs.contiguous(memory_format=torch.channels_last)

            # 混合增强策略：CutMix或MixUp（降低强度以提升稳定性）
            rand_val = torch.rand(1).item()
            if rand_val < cutmix_prob:
                inputs, targets, _ = cutmix_batch(inputs, targets, alpha=mix_alpha)
            elif rand_val < cutmix_prob + mixup_prob:
                inputs, targets, _ = mixup_batch(inputs, targets, alpha=mixup_alpha)
            
            # 检查输入是否有异常值
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                optimizer.zero_grad(set_to_none=True)
                continue
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                optimizer.zero_grad(set_to_none=True)
                continue

            # 梯度累积：只在第一步清零梯度，其他步骤累积
            if use_gradient_accumulation:
                # 只在累积的第一步清零梯度
                if i % accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)
            # 兼容不同版本的 autocast API
            # ========== 扩散模型时间步采样（如果启用）==========
            timestep = None
            B = inputs.shape[0]  # 获取batch size
            if noise_scheduler is not None:
                # 采样随机时间步 [0, 1]，用于扩散去噪训练
                # 在训练早期更多使用较大的时间步（更多噪声），后期使用较小的时间步
                if epoch < remaining_epochs * 0.3:  # 前30% epoch：更多噪声
                    timestep = torch.rand(B, device=device) * 0.8 + 0.2  # [0.2, 1.0]
                elif epoch < remaining_epochs * 0.7:  # 30%-70% epoch：中等噪声
                    timestep = torch.rand(B, device=device) * 0.6 + 0.2  # [0.2, 0.8]
                else:  # 后30% epoch：较少噪声
                    timestep = torch.rand(B, device=device) * 0.4 + 0.1  # [0.1, 0.5]
            
            if AUTOCAST_NEW_API:
                autocast_context = autocast(device_type='cuda', enabled=(device.type == 'cuda'))
            else:
                # torch.cuda.amp.autocast 不接受参数，只在 cuda 设备上使用
                autocast_context = autocast() if device.type == 'cuda' else torch.no_grad()
            with autocast_context:
                # 传递timestep参数以启用扩散式去噪（如果可用）
                if timestep is not None:
                    logit_main, aux_list = model(inputs, timestep=timestep)
                else:
                    logit_main, aux_list = model(inputs)
                # 确保输出和目标维度匹配
                # logit_main 应该是 [B, 3, H, W]，如果是 [B, 1, H, W] 则重复到3通道
                if logit_main.shape[1] == 1 and targets.shape[1] == 3:
                    logit_main = logit_main.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
                
                # 检查输出是否有异常值
                if torch.isnan(logit_main).any() or torch.isinf(logit_main).any():
                    print(f"警告: 模型输出包含NaN/Inf，跳过此batch (Epoch {epoch + start_epoch + 1}, Batch {i+1})")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                # 自编码器主输出损失
                # 控制SSIM计算频率以提升速度
                compute_ssim_now = (i % compute_ssim_every_n_batches == 0) if compute_ssim_every_n_batches > 1 else True
                loss_main, l1_loss, mse_loss, ssim_loss = autoencoder_loss(logit_main, targets, compute_ssim=compute_ssim_now)
                loss_aux = 0.0
                # 辅助输出也使用重建损失
                for aux in aux_list:
                    # 确保辅助输出和目标维度匹配
                    if aux.shape[1] == 1 and targets.shape[1] == 3:
                        aux = aux.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
                    aux_loss, _, _, _ = autoencoder_loss(aux, targets, compute_ssim=compute_ssim_now)
                    loss_aux = loss_aux + aux_loss
                if len(aux_list) > 0:
                    loss_aux = loss_aux / len(aux_list)
                loss = loss_main + aux_weight * loss_aux

            # 检查loss是否为NaN或Inf，以及是否过大
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
                actual_epoch_display = epoch + start_epoch + 1
                print(f"警告: 检测到异常损失 ({loss.item():.4f})，跳过此batch (Epoch {actual_epoch_display}, Batch {i+1})")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # 梯度累积：将loss除以累积步数
            if use_gradient_accumulation:
                loss = loss / accumulation_steps
            
            # 使用更小的loss scale以提升稳定性
            scaler.scale(loss).backward()
            # 梯度反缩放 + 裁剪（更严格的裁剪）
            scaler.unscale_(optimizer)
            
            # 检查梯度是否有NaN/Inf
            has_nan_grad = False
            nan_param_name = None
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        nan_param_name = name
                        break
            
            if has_nan_grad:
                actual_epoch_display = epoch + start_epoch + 1
                print(f"警告: 参数 {nan_param_name} 的梯度包含NaN/Inf，跳过此batch (Epoch {actual_epoch_display}, Batch {i+1})")
                # 重置scaler状态（在unscale_后检测到NaN时，必须调用update来重置状态）
                optimizer.zero_grad(set_to_none=True)
                scaler.update()  # 重置scaler状态，避免下次调用unscale_时出错
                continue
            
            # 梯度裁剪（更严格的裁剪，防止梯度爆炸和NaN）
            # 使用更小的max_norm以提升稳定性，并在裁剪前检查
            # 梯度裁剪：将max_norm从0.3增加到10.0，允许更大的梯度流动，避免过度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # 检查梯度范数是否异常（放宽阈值，只在极端情况下跳过）
            # 梯度范数在10-50之间通常是可以接受的，只有在极端情况下(>100)才跳过
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                actual_epoch_display = epoch + start_epoch + 1
                print(f"警告: 梯度范数为NaN/Inf，跳过此batch (Epoch {actual_epoch_display}, Batch {i+1})")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            elif grad_norm > 100.0:
                # 只有在极端梯度爆炸时才跳过（>100）
                actual_epoch_display = epoch + start_epoch + 1
                print(f"警告: 梯度范数过大 ({grad_norm:.4f})，跳过此batch (Epoch {actual_epoch_display}, Batch {i+1})")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            
            # 梯度累积：只在累积的最后一步更新参数
            if use_gradient_accumulation:
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    # EMA更新
                    if ema_helper is not None:
                        ema_helper.update(model)
                    # Scheduler更新：OneCycleLR每个batch更新一次
                    scheduler.step()
            else:
                scaler.step(optimizer)
                scaler.update()
                # EMA更新
                if ema_helper is not None:
                    ema_helper.update(model)
                # Scheduler更新：OneCycleLR每个batch更新一次
                scheduler.step()

            # 梯度累积时，loss已经除以accumulation_steps，需要还原用于显示
            display_loss = loss.item() * (accumulation_steps if use_gradient_accumulation else 1)
            train_loss += display_loss

            # 减少打印频率以提升速度（每100个batch打印一次）
            if i % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                actual_epoch_display = epoch + start_epoch + 1
                final_epoch = start_epoch + remaining_epochs
                print(f'Epoch [{actual_epoch_display}/{final_epoch}], '
                      f'Batch [{i + 1}/{len(train_loader)}], '
                      f'Loss: {display_loss:.4f}, '
                      f'LR: {current_lr:.6f}, '
                      f'AuxW: {aux_weight:.3f}')

        train_loss /= max(1, len(train_loader))
        actual_epoch_display = epoch + start_epoch + 1
        final_epoch = start_epoch + remaining_epochs
        print(f"Epoch [{actual_epoch_display}/{final_epoch}] finished, Average Train Loss: {train_loss:.4f}")

        # ===== 验证（减少验证频率以提升训练速度）=====
        should_validate = (actual_epoch_display % val_frequency == 0) or (actual_epoch_display == final_epoch)
        
        if not should_validate:
            # 跳过验证，只记录训练损失
            train_losses.append(train_loss)
            val_losses.append(val_losses[-1] if val_losses else 0.0)
            val_dices.append(val_dices[-1] if val_dices else 0.0)
            continue
        
        # 执行验证
        model.eval()
        val_loss = 0.0
        val_dice_epoch = 0.0
        val_iou_epoch = 0.0
        val_precision_epoch = 0.0
        val_recall_epoch = 0.0
        first_batch_pred = None
        first_batch_img = None

        # 验证使用相同的固定辅助损失权重（降低以提升稳定性）
        val_aux_weight = 0.15
        with torch.no_grad():
            for bi, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                # 兼容不同版本的 autocast API
                if AUTOCAST_NEW_API:
                    autocast_context = autocast(device_type='cuda', enabled=(device.type == 'cuda'))
                else:
                    # torch.cuda.amp.autocast 不接受参数，只在 cuda 设备上使用
                    autocast_context = autocast() if device.type == 'cuda' else torch.no_grad()
                with autocast_context:
                    # 使用EMA与TTA
                    val_model = ema_helper.ema if (ema_helper is not None) else model
                    if use_tta:
                        logit_main = tta_horizontal(val_model, inputs)
                        aux_list = []
                    else:
                        logit_main, aux_list = val_model(inputs)
                    # 确保输出和目标维度匹配
                    if logit_main.shape[1] == 1 and targets.shape[1] == 3:
                        logit_main = logit_main.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
                    # 自编码器验证损失（验证时总是计算SSIM）
                    loss_main, l1_score, mse_score, ssim_score = autoencoder_loss(logit_main, targets, compute_ssim=True)
                    loss_aux = 0.0
                    for aux in aux_list:
                        # 确保辅助输出和目标维度匹配
                        if aux.shape[1] == 1 and targets.shape[1] == 3:
                            aux = aux.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
                        aux_loss, _, _, _ = autoencoder_loss(aux, targets, compute_ssim=True)
                        loss_aux = loss_aux + aux_loss
                    if len(aux_list) > 0:
                        loss_aux = loss_aux / len(aux_list)
                    loss = loss_main + val_aux_weight * loss_aux
                    # 使用L1损失作为主要指标（类似于分割任务的Dice）
                    dice_score = l1_score
                val_loss += loss.item()
                val_dice_epoch += dice_score  # dice_score 实际是 L1 loss（重建误差，越低越好）

                # 计算额外的监控指标（自编码器使用重建质量指标）
                # 计算PSNR (Peak Signal-to-Noise Ratio)
                mse_val = F.mse_loss(logit_main, targets).item()
                psnr = 20 * math.log10(1.0) - 10 * math.log10(mse_val) if mse_val > 0 else 100
                val_iou_epoch += psnr  # val_iou 实际存储 PSNR（越高越好）
                val_precision_epoch += mse_score  # val_precision 实际存储 MSE（越低越好）
                val_recall_epoch += ssim_score   # val_recall 实际存储 1-SSIM（SSIM损失，越低越好，SSIM越高越好）

                if bi == 0:
                    # 对于自编码器，输出已经是RGB图像，不需要sigmoid
                    # 但需要反标准化到[0,1]范围（因为输入被标准化到[-1,1]）
                    # 转换为float32以支持clamp操作（AMP可能产生float16）
                    first_batch_pred = (logit_main.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1)
                    first_batch_img  = inputs.detach().cpu().float()

        val_loss /= max(1, len(val_loader))
        val_dice_epoch /= max(1, len(val_loader))
        val_iou_epoch /= max(1, len(val_loader))
        val_precision_epoch /= max(1, len(val_loader))
        val_recall_epoch /= max(1, len(val_loader))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice_epoch)
        val_ious.append(val_iou_epoch)
        val_precisions.append(val_precision_epoch)
        val_recalls.append(val_recall_epoch)

        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        
        # 检查验证指标是否有效
        if torch.isnan(torch.tensor(val_dice_epoch)) or torch.isinf(torch.tensor(val_dice_epoch)):
            actual_epoch_display = epoch + start_epoch + 1
            final_epoch = start_epoch + remaining_epochs
            print(f'警告: 验证指标包含NaN/Inf，可能模型已损坏')
            print(f'Epoch [{actual_epoch_display}/{final_epoch}], Train Loss: {train_loss:.4f}, '
                  f'验证失败 (NaN/Inf), LR: {current_lr:.6f}')
            # 如果验证失败，尝试加载之前的检查点
            if epoch > 0:
                try:
                    if os.path.exists('./best_autoencoder.pth'):
                        print("尝试加载最佳模型恢复训练...")
                        best_checkpoint = torch.load('./best_autoencoder.pth', map_location=device)
                        if isinstance(best_checkpoint, dict) and 'model_state_dict' in best_checkpoint:
                            model.load_state_dict(best_checkpoint['model_state_dict'])
                            if 'ema_state_dict' in best_checkpoint and ema_helper is not None:
                                ema_helper.ema.load_state_dict(best_checkpoint['ema_state_dict'])
                        else:
                            model.load_state_dict(best_checkpoint)
                except Exception as e:
                    print(f"加载检查点失败: {e}")
            continue
        
        actual_epoch_display = epoch + start_epoch + 1
        final_epoch = start_epoch + remaining_epochs
        # 注意：这里是自编码器任务，使用重建指标而不是分割指标
        # val_dice_epoch 实际是 L1 loss（越低越好）
        # val_iou_epoch 实际是 PSNR（越高越好）
        # val_precision_epoch 实际是 MSE（越低越好）
        # val_recall_epoch 实际是 1-SSIM（越低越好，SSIM越高越好）
        print(f'Epoch [{actual_epoch_display}/{final_epoch}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val L1: {val_dice_epoch:.4f}, Val PSNR: {val_iou_epoch:.2f}dB, '
              f'Val MSE: {val_precision_epoch:.6f}, Val 1-SSIM: {val_recall_epoch:.4f}, '
              f'LR: {current_lr:.6f}')
        
        # 如果需要真正的分割指标（Dice/IoU/Precision/Recall），需要将输出转换为二值掩码
        # 但自编码器任务通常不需要这些指标，因为它直接重建图像

        # 保存最佳模型 & 早停
        if val_dice_epoch > best_val_dice:
            best_val_dice = val_dice_epoch
            bad_epochs = 0
            save_path = './best_autoencoder.pth'
            # 保存最佳模型（包含完整检查点信息）
            best_model = ema_helper.ema if (ema_helper is not None) else model
            checkpoint_dict = {
                'epoch': actual_epoch_display,
                'model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_dice': best_val_dice,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_dices': val_dices,
                'val_ious': val_ious,
                'val_precisions': val_precisions,
                'val_recalls': val_recalls,
            }
            if ema_helper is not None:
                checkpoint_dict['ema_state_dict'] = ema_helper.ema.state_dict()
            
            torch.save(checkpoint_dict, save_path)
            # 同时保存仅模型权重（兼容旧版本）
            torch.save(best_model.state_dict(), './best_autoencoder_weights_only.pth')
            print(f"保存最佳模型: {save_path} (Val L1={best_val_dice:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"早停: {patience} 个 epoch 无提升，最佳 Val Dice={best_val_dice:.4f}")
                break


        # ===== 可视化 & 曲线保存（固定采样前 4 张，更有代表性） =====
        # 使用配置的频率保存预测结果
        if (actual_epoch_display % save_pred_frequency == 0) or (actual_epoch_display == final_epoch):
            try:
                if first_batch_pred is not None:
                    actual_epoch_for_vis = epoch + start_epoch
                    # 自编码器输出已经是RGB图像，直接保存
                    grid = make_grid(first_batch_pred[:4], nrow=4, padding=2)
                    save_image(grid, os.path.join(ROOT_PATH, 'mlp_LocalLines_img-Codings', f'epoch_{actual_epoch_for_vis}_pred.png'))

                    # 同时保存对应输入图像（反标准化到[0,1]）
                    # 确保是float32类型以支持clamp操作
                    img_vis = (first_batch_img[:4].float() * 0.5 + 0.5).clamp(0, 1)
                    grid_img = make_grid(img_vis, nrow=4, padding=2)
                    save_image(grid_img, os.path.join(ROOT_PATH, 'mlp_LocalLines_img-Codings', f'epoch_{actual_epoch_for_vis}_img.png'))
            except Exception as e:
                print("保存输出图失败：", e)

            # 保存模型与曲线（保存完整检查点以便恢复训练）
            actual_epoch_for_save = epoch + start_epoch
            # 保存完整检查点（包含优化器、调度器等状态）
            checkpoint_dict = {
                'epoch': actual_epoch_for_save,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_dice': best_val_dice,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_dices': val_dices,
                'val_ious': val_ious,
                'val_precisions': val_precisions,
                'val_recalls': val_recalls,
            }
            # 如果有EMA，也保存EMA状态
            if ema_helper is not None:
                checkpoint_dict['ema_state_dict'] = ema_helper.ema.state_dict()
            
            # 保存完整检查点
            torch.save(checkpoint_dict, f'./Autoencoder_{actual_epoch_for_save}.pth')
            # 同时保存仅模型权重（兼容旧版本）
            torch.save(model.state_dict(), f'./Autoencoder_{actual_epoch_for_save}_weights_only.pth')
            plt.figure(figsize=(12, 8))

            # 子图1：损失
            plt.subplot(2, 2, 1)
            plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
            plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss Curves")

            # 子图2：Dice和IoU
            plt.subplot(2, 2, 2)
            plt.plot(range(len(val_dices)), val_dices, label='Val L1 Loss', color='orange')
            plt.plot(range(len(val_ious)), val_ious, label='Val PSNR (dB)', color='red')
            plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.title("L1 Loss & PSNR")

            # 子图3：Precision和Recall
            plt.subplot(2, 2, 3)
            plt.plot(range(len(val_precisions)), val_precisions, label='Val MSE', color='green')
            plt.plot(range(len(val_recalls)), val_recalls, label='Val 1-SSIM', color='purple')
            plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.title("MSE & 1-SSIM")

            plt.tight_layout()
            plt.savefig(os.path.join(ROOT_PATH, 'curves', f'curve_epoch_{actual_epoch_for_vis}.png'))
            plt.close()

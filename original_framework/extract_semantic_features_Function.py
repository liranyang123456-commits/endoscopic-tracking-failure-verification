"""
语义特征提取脚本
提取模型的语义编码特征，用于图像相似度计算
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont

# 导入模型定义
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Copy12 import SwinUNetLarge


def create_comparison_image(img1_path, img2_path, distance, similarity, output_path, 
                           img1_label="前一张图像", img2_label="当前图像"):
    """
    合并两张图像并添加相似度标注
    
    Args:
        img1_path: 第一张图像的路径
        img2_path: 第二张图像的路径
        distance: 欧氏距离
        similarity: 转换相似度
        output_path: 输出图像路径
        img1_label: 第一张图像的标签
        img2_label: 第二张图像的标签
    """
    try:
        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"警告: 无法读取图像 {img1_path} 或 {img2_path}")
            return False
        
        # 调整图像大小使其一致（取较大的尺寸）
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        # 调整图像大小
        img1_resized = cv2.resize(img1, (target_w, target_h))
        img2_resized = cv2.resize(img2, (target_w, target_h))
        
        # 水平拼接图像
        combined_img = np.hstack([img1_resized, img2_resized])
        
        # 转换为PIL图像以便添加中文文字
        combined_img_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(combined_img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # 尝试加载中文字体
        try:
            # Windows系统字体路径
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/simsun.ttc",   # 宋体
                "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
            ]
            font = None
            for fp in font_paths:
                if os.path.exists(fp):
                    font = ImageFont.truetype(fp, 30)
                    break
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 计算文字位置
        img_width = combined_img.shape[1]
        img_height = combined_img.shape[0]
        
        # 在第一张图像上方添加标签
        text1 = f"{img1_label}"
        bbox1 = draw.textbbox((0, 0), text1, font=font)
        text1_w = bbox1[2] - bbox1[0]
        text1_h = bbox1[3] - bbox1[1]
        draw.rectangle([(10, 10), (10 + text1_w + 10, 10 + text1_h + 10)], fill=(0, 0, 0, 200))
        draw.text((15, 15), text1, fill=(255, 255, 255), font=font)
        
        # 在第二张图像上方添加标签
        text2 = f"{img2_label}"
        bbox2 = draw.textbbox((0, 0), text2, font=font)
        text2_w = bbox2[2] - bbox2[0]
        text2_h = bbox2[3] - bbox2[1]
        draw.rectangle([(img_width // 2 + 10, 10), (img_width // 2 + 10 + text2_w + 10, 10 + text2_h + 10)], fill=(0, 0, 0, 200))
        draw.text((img_width // 2 + 15, 15), text2, fill=(255, 255, 255), font=font)
        
        # 在图像底部添加相似度信息
        info_text = [
            f"欧氏距离: {distance:.4f}",
            f"转换相似度: {similarity:.4f}"
        ]
        
        # 计算信息文本区域
        line_height = text1_h + 5
        info_y_start = img_height - len(info_text) * line_height - 20
        
        # 绘制半透明背景
        info_bg_height = len(info_text) * line_height + 20
        overlay = Image.new('RGBA', (img_width, info_bg_height), (0, 0, 0, 180))
        pil_img.paste(overlay, (0, info_y_start), overlay)
        
        # 添加文字
        for i, text in enumerate(info_text):
            y_pos = info_y_start + 10 + i * line_height
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            # 居中显示
            x_pos = (img_width - text_w) // 2
            draw.text((x_pos, y_pos), text, fill=(255, 255, 255), font=font)
        
        # 在中间添加分隔线
        line_x = img_width // 2
        draw.line([(line_x, 0), (line_x, img_height)], fill=(255, 255, 255), width=3)
        
        # 保存图像
        pil_img.save(output_path, quality=95)
        return True
        
    except Exception as e:
        print(f"创建比较图像失败: {e}")
        return False


class SegDataset(Dataset):
    """分割数据集类"""
    def __init__(
        self,
        root_dir_img: str,
        root_dir_mask: str,
        image_size: Tuple[int, int] = (128, 128),
    ):
        self.root_dir_img = root_dir_img
        self.root_dir_mask = root_dir_mask
        
        img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        all_img_files = sorted(os.listdir(self.root_dir_img))
        all_mask_files = sorted(os.listdir(self.root_dir_mask))
        
        self.image_files = [f for f in all_img_files if any(f.lower().endswith(ext) for ext in img_extensions)]
        self.mask_files = [f for f in all_mask_files if any(f.lower().endswith(ext) for ext in img_extensions)]
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {root_dir_img} 中未找到图像文件")
        if len(self.mask_files) == 0:
            raise ValueError(f"在 {root_dir_img} 中未找到标签文件")
        
        img_names = [os.path.splitext(f)[0] for f in self.image_files]
        mask_names = [os.path.splitext(f)[0] for f in self.mask_files]
        if set(img_names) != set(mask_names):
            print(f"警告: 图像和标签文件名不完全匹配")
            common_names = set(img_names) & set(mask_names)
            if len(common_names) == 0:
                raise ValueError(f"图像和标签文件没有匹配的文件名")
            self.image_files = [f for f in self.image_files if os.path.splitext(f)[0] in common_names]
            self.mask_files = [f for f in self.mask_files if os.path.splitext(f)[0] in common_names]
            print(f"  匹配的文件数量: {len(self.image_files)}")
        
        assert len(self.image_files) == len(self.mask_files), f"匹配后文件数量不一致"

        resize_h, resize_w = image_size
        self.transform = A.Compose([
            A.Resize(resize_h, resize_w, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir_img, self.image_files[idx])
        mask_path = os.path.join(self.root_dir_mask, self.mask_files[idx])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取标签: {mask_path}")

        out = self.transform(image=img, mask=mask)
        image_t = out["image"]
        mask_t = out["mask"].unsqueeze(0).float()
        mask_t = (mask_t > 0.5).float()
        return image_t, mask_t, self.image_files[idx]


class SemanticFeatureExtractor:
    """语义特征提取器"""
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册hook到用于匹配的层"""
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.features[name] = output.detach().cpu()
                elif isinstance(output, (list, tuple)):
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        self.features[name] = output[0].detach().cpu()
            return hook
        
        # 注册hook到用于匹配的层
        hook_targets = [
            ('swin_bottleneck', self.model.swin_bottleneck),
            ('swin4', self.model.swin4),
        ]
        
        for name, module in hook_targets:
            hook = module.register_forward_hook(get_activation(name))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_features(self):
        """清空特征缓存"""
        self.features = {}
    
    def extract_features(self, inputs):
        """
        提取语义特征
        
        Returns:
            features_dict: 包含以下特征
                - 'latent': 全局特征向量 [B, 512]
                - 'swin_bottleneck': 语义特征 [B, C, H/16, W/16]
                - 'swin4': 平衡特征 [B, C, H/8, W/8]
        """
        self.clear_features()
        
        try:
            # 前向传播（会触发hook）
            output, aux_outputs = self.model(inputs)
            
            # 提取latent特征（全局特征向量）
            latent = self.model.encode(inputs)  # [B, 512]
        except Exception as e:
            # 如果编译后的模型在运行时失败（例如缺少 Triton），抛出更清晰的错误
            error_msg = str(e)
            if "triton" in error_msg.lower() or "dynamo" in error_msg.lower():
                raise RuntimeError(
                    f"模型运行时失败（可能是 torch.compile 相关问题）: {e}\n"
                    f"建议：设置环境变量 VAL_DISABLE_TORCH_COMPILE=1 来禁用 torch.compile"
                ) from e
            else:
                raise
        
        # 获取hook捕获的特征
        features = {
            'latent': latent.cpu(),  # [B, 512] 全局特征向量
        }
        
        # 添加hook捕获的特征
        if 'swin_bottleneck' in self.features:
            features['swin_bottleneck'] = self.features['swin_bottleneck']  # [B, C, H/16, W/16]
        
        if 'swin4' in self.features:
            features['swin4'] = self.features['swin4']  # [B, C, H/8, W/8]
        
        return features


class ModelManager:
    """模型管理器（单例模式）"""
    _instance = None
    _model = None
    _feature_extractor = None
    _device = None
    _transform = None
    _image_size = (128, 128)
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def initialize(self, model_path, image_size=(128, 128), device=None):
        """
        初始化模型
        
        Args:
            model_path: 模型权重文件路径
            image_size: 图像尺寸 (height, width)
            device: 计算设备，None则自动选择
        """
        if self._model is not None:
            # 模型已加载，检查是否使用相同路径
            return
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 兼容上层传入 "cuda"/"cpu" 字符串
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except Exception:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device
        self._image_size = image_size
        
        print(f"初始化模型: {model_path}")
        print(f"使用设备: {device}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载checkpoint
        # 设置 weights_only=False 以兼容 PyTorch 2.6+ 的默认行为
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 处理checkpoint格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 检测base_c
        base_c = None
        if 'swin2.net.0.norm1.weight' in state_dict:
            swin2_channels = state_dict['swin2.net.0.norm1.weight'].shape[0]
            base_c = swin2_channels // 2
            print(f"✓ 检测到 base_c = {base_c}")
        elif 'head_main.weight' in state_dict:
            head_in_channels = state_dict['head_main.weight'].shape[1]
            base_c = head_in_channels
            print(f"✓ 检测到 base_c = {base_c}")
        else:
            base_c = 32
            print(f"⚠️  无法检测base_c，使用默认值: {base_c}")
        
        if base_c not in [32, 64, 128]:
            base_c = 32
        
        # 创建模型
        print(f"创建模型 (base_c={base_c})...")
        self._model = SwinUNetLarge(
            in_channels=3,
            base_c=base_c,
            dropout_p=0.15,
            window_size=8,
            depths=(2, 3, 4),
            bottleneck_depth=4,
            num_heads_stages=(4, 8, 8),
            bottleneck_heads=16,
            use_pos_emb=True,
            use_se_sparse=True,
            use_input_pos_emb=True
        ).to(device)
        
        # 加载权重
        try:
            model_state_dict = self._model.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            
            for key, value in state_dict.items():
                if key in model_state_dict:
                    if model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                    else:
                        skipped_keys.append(f"{key}: shape mismatch")
                else:
                    skipped_keys.append(f"{key}: not in model")
            
            self._model.load_state_dict(filtered_state_dict, strict=False)
            print(f"✓ 成功加载 {len(filtered_state_dict)}/{len(state_dict)} 个权重层")
            if skipped_keys:
                print(f"⚠️  跳过了 {len(skipped_keys)} 个不匹配的层")
        except Exception as e:
            raise RuntimeError(f"加载模型权重失败: {e}")
        
        self._model.eval()
        
        # 性能优化：如果支持，使用torch.compile加速（PyTorch 2.0+）
        # 可以通过环境变量 VAL_DISABLE_TORCH_COMPILE=1 来禁用编译
        use_compile = os.environ.get("VAL_DISABLE_TORCH_COMPILE", "0").strip().lower() != "1"
        self._model_compiled = False
        
        if use_compile and hasattr(torch, 'compile') and device.type == 'cuda':
            try:
                print("  尝试使用 torch.compile 加速模型...")
                # 保存原始模型引用，以便在运行时失败时回退
                original_model = self._model
                # 测试编译：先进行一次前向传播测试
                try:
                    with torch.no_grad():
                        test_input = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
                        _ = original_model(test_input)  # 确保原始模型可以运行
                    
                    # 尝试编译
                    compiled_model = torch.compile(original_model, mode='reduce-overhead')
                    
                    # 测试编译后的模型是否能正常运行
                    with torch.no_grad():
                        _ = compiled_model(test_input)
                    
                    self._model = compiled_model
                    self._model_compiled = True
                    print("  ✓ torch.compile 已启用并测试通过")
                except Exception as test_e:
                    # 编译或测试失败，回退到原始模型
                    error_msg = str(test_e)
                    if "triton" in error_msg.lower():
                        print(f"  ⚠️  torch.compile 失败（缺少 Triton），已自动禁用，使用普通模式")
                        print(f"  💡 提示：如需禁用此警告，可设置环境变量 VAL_DISABLE_TORCH_COMPILE=1")
                    else:
                        print(f"  ⚠️  torch.compile 测试失败: {test_e}，继续使用普通模式")
                    self._model = original_model
                    self._model_compiled = False
            except Exception as e:
                print(f"  ⚠️  torch.compile 启用失败: {e}，继续使用普通模式")
                self._model_compiled = False
        else:
            if not use_compile:
                print("  ℹ️  torch.compile 已通过环境变量禁用")
            elif device.type != 'cuda':
                print("  ℹ️  torch.compile 仅在 CUDA 设备上可用")
            else:
                print("  ℹ️  torch.compile 不可用，使用普通模式")
        
        # 创建特征提取器
        self._feature_extractor = SemanticFeatureExtractor(self._model)
        self._feature_extractor.register_hooks()
        
        # 创建图像变换
        resize_h, resize_w = image_size
        self._transform = A.Compose([
            A.Resize(resize_h, resize_w, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
        
        print("✓ 模型初始化完成")
    
    def get_model(self):
        """获取模型"""
        if self._model is None:
            raise RuntimeError("模型未初始化，请先调用 initialize() 方法")
        return self._model
    
    def get_feature_extractor(self):
        """获取特征提取器"""
        if self._feature_extractor is None:
            raise RuntimeError("模型未初始化，请先调用 initialize() 方法")
        return self._feature_extractor
    
    def get_device(self):
        """获取设备"""
        return self._device
    
    def get_transform(self):
        """获取图像变换"""
        if self._transform is None:
            raise RuntimeError("模型未初始化，请先调用 initialize() 方法")
        return self._transform
    
    def preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image: 图像路径（str）或numpy数组（np.ndarray，形状为HWC，BGR或RGB）
            
        Returns:
            torch.Tensor: 预处理后的图像张量 [1, 3, H, W]
        """
        # 如果是路径，读取图像
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图像: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image.copy()
            # 如果是BGR，转换为RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                # 假设是BGR格式（OpenCV默认），转换为RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}，支持str（路径）或np.ndarray")
        
        # 应用变换
        transformed = self._transform(image=img)
        image_t = transformed["image"].unsqueeze(0)  # [1, 3, H, W]
        
        return image_t


def main():
    # ====================== 配置参数 ======================
    model_path = r'D:\reloc3r\Autoencoder_129.pth'
    val_img_dir = r'L:\val_Contour_IMGs'
    val_mask_dir = None  # 如果mask文件夹不同，请修改这里
    
    # 输出配置
    output_dir = r'L:\matching_features'  # 特征保存目录
    batch_size = 64  # 批量大小
    num_workers = 8  # 数据加载线程数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试模式配置
    test_mode = False  # 是否启用测试模式（只处理前N个样本），False表示处理全部图像
    test_num_samples = 10  # 测试模式下处理的样本数量（仅在test_mode=True时生效）
    
    image_size = (128, 128)
    
    print(f"使用设备: {device}")
    print(f"图像路径: {val_img_dir}")
    print(f"模型路径: {model_path}")
    print(f"输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 自动查找mask文件夹
    if val_mask_dir is None:
        base_dir = os.path.dirname(val_img_dir) if os.path.dirname(val_img_dir) else r'L:\\'
        possible_mask_dirs = [
            os.path.join(base_dir, 'val_Contour_IMGs_Output'),
            os.path.join(base_dir, 'val_Contour_IMGs_Masks'),
            os.path.join(base_dir, 'val_Contour_IMGs_Labels'),
            os.path.join(base_dir, 'val_Contour_IMGs_GT'),
            os.path.join(val_img_dir, 'masks'),
            os.path.join(val_img_dir, 'labels'),
            val_img_dir,
        ]
        
        found = False
        for mask_dir in possible_mask_dirs:
            if os.path.exists(mask_dir) and len([f for f in os.listdir(mask_dir) 
                                                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]) > 0:
                val_mask_dir = mask_dir
                found = True
                print(f"✓ 自动找到mask文件夹: {val_mask_dir}")
                break
        
        if not found:
            print("警告: 未找到mask文件夹，将使用图像文件夹作为mask文件夹")
            val_mask_dir = val_img_dir
    
    # 创建数据集
    try:
        val_dataset = SegDataset(val_img_dir, val_mask_dir, image_size)
        print(f"数据集大小: {len(val_dataset)}")
    except Exception as e:
        print(f"创建数据集失败: {e}")
        return
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # 加载模型
    print(f"\n加载模型权重: {model_path}")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    # 设置 weights_only=False 以兼容 PyTorch 2.6+ 的默认行为
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 处理checkpoint格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 检测base_c
    base_c = None
    if 'swin2.net.0.norm1.weight' in state_dict:
        swin2_channels = state_dict['swin2.net.0.norm1.weight'].shape[0]
        base_c = swin2_channels // 2
        print(f"✓ 检测到 base_c = {base_c}")
    elif 'head_main.weight' in state_dict:
        head_in_channels = state_dict['head_main.weight'].shape[1]
        base_c = head_in_channels
        print(f"✓ 检测到 base_c = {base_c}")
    else:
        base_c = 32
        print(f"⚠️  无法检测base_c，使用默认值: {base_c}")
    
    if base_c not in [32, 64, 128]:
        base_c = 32
    
    # 创建模型
    print(f"\n创建模型 (base_c={base_c})...")
    model = SwinUNetLarge(
        in_channels=3,
        base_c=base_c,
        dropout_p=0.15,
        window_size=8,
        depths=(2, 3, 4),
        bottleneck_depth=4,
        num_heads_stages=(4, 8, 8),
        bottleneck_heads=16,
        use_pos_emb=True,
        use_se_sparse=True,
        use_input_pos_emb=True
    ).to(device)
    
    # 加载权重
    try:
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    skipped_keys.append(f"{key}: shape mismatch")
            else:
                skipped_keys.append(f"{key}: not in model")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"✓ 成功加载 {len(filtered_state_dict)}/{len(state_dict)} 个权重层")
        if skipped_keys:
            print(f"⚠️  跳过了 {len(skipped_keys)} 个不匹配的层")
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        return
    
    model.eval()
    
    # 性能优化：如果支持，使用torch.compile加速（PyTorch 2.0+）
    # 可以通过环境变量 VAL_DISABLE_TORCH_COMPILE=1 来禁用编译
    use_compile = os.environ.get("VAL_DISABLE_TORCH_COMPILE", "0").strip().lower() != "1"
    
    if use_compile and hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            print("  尝试使用 torch.compile 加速模型...")
            # 测试编译：先进行一次前向传播测试
            try:
                with torch.no_grad():
                    test_input = torch.randn(1, 3, image_size[0], image_size[1]).to(device)
                    _ = model(test_input)  # 确保原始模型可以运行
                
                # 尝试编译
                compiled_model = torch.compile(model, mode='reduce-overhead')
                
                # 测试编译后的模型是否能正常运行
                with torch.no_grad():
                    _ = compiled_model(test_input)
                
                model = compiled_model
                print("  ✓ torch.compile 已启用并测试通过")
            except Exception as test_e:
                # 编译或测试失败，回退到原始模型
                error_msg = str(test_e)
                if "triton" in error_msg.lower():
                    print(f"  ⚠️  torch.compile 失败（缺少 Triton），已自动禁用，使用普通模式")
                    print(f"  💡 提示：如需禁用此警告，可设置环境变量 VAL_DISABLE_TORCH_COMPILE=1")
                else:
                    print(f"  ⚠️  torch.compile 测试失败: {test_e}，继续使用普通模式")
        except Exception as e:
            print(f"  ⚠️  torch.compile 启用失败: {e}，继续使用普通模式")
    else:
        if not use_compile:
            print("  ℹ️  torch.compile 已通过环境变量禁用")
        elif device.type != 'cuda':
            print("  ℹ️  torch.compile 仅在 CUDA 设备上可用")
        else:
            print("  ℹ️  torch.compile 不可用，使用普通模式")
    
    # 创建特征提取器
    feature_extractor = SemanticFeatureExtractor(model)
    feature_extractor.register_hooks()
    
    print(f"\n开始提取语义特征并比较相邻图像...")
    
    # 存储所有特征（用于后续保存）
    all_latent_features = []  # 全局特征
    all_bottleneck_features = []  # 语义特征
    all_swin4_features = []  # 平衡特征
    all_filenames = []
    
    # 存储相似度结果（用于统计）
    similarity_results = []  # 存储每对图像的相似度
    
    # 创建输出目录用于保存合并图像
    comparison_output_dir = os.path.join(os.path.dirname(val_dataset.root_dir_img), "comparison_images")
    os.makedirs(comparison_output_dir, exist_ok=True)
    print(f"合并图像将保存到: {comparison_output_dir}")
    
    # 如果启用测试模式，限制迭代次数
    max_iterations = None
    if test_mode:
        max_iterations = (test_num_samples + batch_size - 1) // batch_size
        print(f"测试模式: 只处理前 {test_num_samples} 个样本 (最多 {max_iterations} 个batch)")
    
    print(f"\n提取特征并比较相邻图像:")
    print(f"{'='*120}")
    print(f"{'序号':<6} {'前一张图像':<45} {'当前图像':<45} {'组合距离':<12} {'各尺度距离':<30}")
    print("-" * 120)
    
    with torch.no_grad():
        sample_count = 0
        prev_features = None  # 上一张图像的特征
        prev_filename = None  # 上一张图像的文件名
        
        for batch_idx, (inputs, targets, filenames) in enumerate(tqdm(val_loader, desc="提取特征")):
            # 测试模式：如果已处理足够的样本，提前退出
            if test_mode and sample_count >= test_num_samples:
                break
            
            # 测试模式：限制batch数量
            if test_mode and max_iterations is not None and batch_idx >= max_iterations:
                break
            
            inputs = inputs.to(device, non_blocking=True)
            
            # 提取语义特征
            features = feature_extractor.extract_features(inputs)
            
            # 处理batch中的每张图像
            batch_size_current = inputs.shape[0]
            for i in range(batch_size_current):
                if test_mode and sample_count >= test_num_samples:
                    break
                
                current_filename = filenames[i]
                
                # 提取当前图像的特征
                current_latent = features['latent'][i:i+1]  # [1, 512]
                
                if 'swin_bottleneck' in features:
                    bottleneck_feat = features['swin_bottleneck'][i:i+1]  # [1, C, H, W]
                    bottleneck_global = F.adaptive_avg_pool2d(bottleneck_feat, (1, 1)).squeeze(-1).squeeze(-1)  # [1, C]
                else:
                    bottleneck_global = None
                
                if 'swin4' in features:
                    swin4_feat = features['swin4'][i:i+1]  # [1, C, H, W]
                    swin4_global = F.adaptive_avg_pool2d(swin4_feat, (1, 1)).squeeze(-1).squeeze(-1)  # [1, C]
                else:
                    swin4_global = None
                
                # 计算欧氏距离（L2距离）
                # 欧氏距离 = sqrt(sum((A - B)^2))
                # 范围: [0, +∞)，值越小越相似
                
                # 如果有上一张图像，计算距离
                if prev_features is not None:
                    prev_latent = prev_features['latent']
                    
                    # 计算latent的欧氏距离
                    distance = torch.norm(current_latent - prev_latent, p=2).item()
                    
                    # 如果有多尺度特征，也计算多尺度距离
                    if bottleneck_global is not None and prev_features.get('bottleneck') is not None:
                        prev_bottleneck = prev_features['bottleneck']
                        dist_bottleneck = torch.norm(bottleneck_global - prev_bottleneck, p=2).item()
                    else:
                        dist_bottleneck = None
                    
                    if swin4_global is not None and prev_features.get('swin4') is not None:
                        prev_swin4 = prev_features['swin4']
                        dist_swin4 = torch.norm(swin4_global - prev_swin4, p=2).item()
                    else:
                        dist_swin4 = None
                    
                    # 说明：欧氏距离本身就是衡量相似度的指标（距离越小越相似）
                    # 如果需要"相似度"形式（值越大越相似），可以使用以下转换：
                    # 转换公式: similarity = 1 / (1 + distance)
                    # - 距离 = 0 时，相似度 = 1.0（完全相同）
                    # - 距离 → +∞ 时，相似度 → 0（完全不同）
                    # - 范围: [0, 1]，值越大越相似
                    # 
                    # 注意：如果不需要转换，可以直接使用距离值（距离越小越相似）
                    similarity = 1.0 / (1.0 + distance)
                    
                    if dist_bottleneck is not None:
                        sim_bottleneck = 1.0 / (1.0 + dist_bottleneck)
                    else:
                        sim_bottleneck = None
                    
                    if dist_swin4 is not None:
                        sim_swin4 = 1.0 / (1.0 + dist_swin4)
                    else:
                        sim_swin4 = None
                    
                    # 打印比较结果（序号从1开始，表示第几对比较）
                    comparison_idx = sample_count  # 第sample_count对比较（图像sample_count-1和sample_count）
                    if sim_bottleneck is not None and sim_swin4 is not None:
                        # 多尺度相似度（基于距离转换）
                        combined_sim = 0.4 * similarity + 0.35 * sim_bottleneck + 0.25 * sim_swin4
                        # 计算组合距离（加权平均）
                        combined_distance = 0.4 * distance + 0.35 * dist_bottleneck + 0.25 * dist_swin4
                        detail_str = f"D:{distance:.3f} B:{dist_bottleneck:.3f} S:{dist_swin4:.3f}"
                        # 更清晰的输出格式
                        print(f"{comparison_idx:<6} {prev_filename:<45} {current_filename:<45} {combined_distance:>10.4f} {detail_str:<30}")
                        print(f"      └─> 比较: '{prev_filename}' 与 '{current_filename}'")
                        print(f"         欧氏距离: Latent={distance:.4f}, Bottleneck={dist_bottleneck:.4f}, Swin4={dist_swin4:.4f}")
                        print(f"         组合距离: {combined_distance:.4f} (越小越相似)")
                        print(f"         转换相似度: {combined_sim:.4f} (范围: 0到1, 越大越相似, 公式: 1/(1+距离))")
                        similarity_results.append({
                            'idx': comparison_idx,
                            'image1': prev_filename,
                            'image2': current_filename,
                            'combined_sim': combined_sim,
                            'combined_distance': combined_distance,
                            'latent_distance': distance,
                            'bottleneck_distance': dist_bottleneck,
                            'swin4_distance': dist_swin4,
                            'latent_sim': similarity,
                            'bottleneck_sim': sim_bottleneck,
                            'swin4_sim': sim_swin4
                        })
                    else:
                        # 仅latent距离
                        print(f"{comparison_idx:<6} {prev_filename:<45} {current_filename:<45} {distance:>10.4f} {'(仅Latent)':<30}")
                        print(f"      └─> 比较: '{prev_filename}' 与 '{current_filename}'")
                        print(f"         欧氏距离: {distance:.4f} (范围: 0到+∞, 越小越相似)")
                        print(f"         转换相似度: {similarity:.4f} (范围: 0到1, 越大越相似, 公式: 1/(1+距离))")
                        similarity_results.append({
                            'idx': comparison_idx,
                            'image1': prev_filename,
                            'image2': current_filename,
                            'combined_sim': similarity,
                            'combined_distance': distance,
                            'latent_distance': distance,
                            'bottleneck_distance': None,
                            'swin4_distance': None,
                            'latent_sim': similarity,
                            'bottleneck_sim': None,
                            'swin4_sim': None
                        })
                
                # 如果有上一张图像，创建并保存合并图像
                if prev_features is not None:
                    # 构建图像路径
                    img1_path = os.path.join(val_dataset.root_dir_img, prev_filename)
                    img2_path = os.path.join(val_dataset.root_dir_img, current_filename)
                    
                    # 确定使用的距离和相似度
                    if sim_bottleneck is not None and sim_swin4 is not None:
                        use_distance = combined_distance
                        use_similarity = combined_sim
                    else:
                        use_distance = distance
                        use_similarity = similarity
                    
                    # 生成输出文件名
                    output_filename = f"comparison_{comparison_idx:04d}_{os.path.splitext(prev_filename)[0]}_vs_{os.path.splitext(current_filename)[0]}.jpg"
                    output_path = os.path.join(comparison_output_dir, output_filename)
                    
                    # 创建合并图像
                    create_comparison_image(
                        img1_path=img1_path,
                        img2_path=img2_path,
                        distance=use_distance,
                        similarity=use_similarity,
                        output_path=output_path,
                        img1_label=f"前一张: {prev_filename}",
                        img2_label=f"当前: {current_filename}"
                    )
                
                # 保存当前特征作为下一张图像的"上一张"（保存原始特征用于欧氏距离计算）
                prev_features = {
                    'latent': current_latent,
                }
                if bottleneck_global is not None:
                    prev_features['bottleneck'] = bottleneck_global
                if swin4_global is not None:
                    prev_features['swin4'] = swin4_global
                prev_filename = current_filename
                
                # 保存特征用于后续保存文件
                all_latent_features.append(current_latent.numpy())
                if bottleneck_global is not None:
                    all_bottleneck_features.append(bottleneck_global.numpy())
                if swin4_global is not None:
                    all_swin4_features.append(swin4_global.numpy())
                all_filenames.append(current_filename)
                
                sample_count += 1
                
                # 测试模式：如果已处理足够的样本，提前退出
                if test_mode and sample_count >= test_num_samples:
                    break
            
            if test_mode and sample_count >= test_num_samples:
                break
    
    # 移除hook
    feature_extractor.remove_hooks()
    
    # 合并所有特征
    print(f"\n合并特征...")
    if len(all_latent_features) > 0:
        latent_features_array = np.concatenate(all_latent_features, axis=0)  # [N, 512]
        print(f"  ✓ 全局特征 (latent): {latent_features_array.shape}")
    
    if len(all_bottleneck_features) > 0:
        bottleneck_features_array = np.concatenate(all_bottleneck_features, axis=0)  # [N, C]
        print(f"  ✓ 语义特征 (swin_bottleneck): {bottleneck_features_array.shape}")
    
    if len(all_swin4_features) > 0:
        swin4_features_array = np.concatenate(all_swin4_features, axis=0)  # [N, C]
        print(f"  ✓ 平衡特征 (swin4): {swin4_features_array.shape}")
    
    # 保存特征
    print(f"\n保存特征...")
    
    # 保存全局匹配特征
    if len(all_latent_features) > 0:
        global_feature_path = os.path.join(output_dir, 'global_features.pt')
        torch.save({
            'features': torch.from_numpy(latent_features_array),
            'filenames': all_filenames,
            'feature_type': 'latent',
            'shape': latent_features_array.shape,
            'description': '全局特征向量，用于全局匹配'
        }, global_feature_path)
        print(f"  ✓ 全局匹配特征已保存: {global_feature_path}")
        print(f"    形状: {latent_features_array.shape}")
    
    # 保存多尺度匹配特征
    if len(all_latent_features) > 0 and len(all_bottleneck_features) > 0 and len(all_swin4_features) > 0:
        multi_scale_feature_path = os.path.join(output_dir, 'multi_scale_features.pt')
        torch.save({
            'latent': torch.from_numpy(latent_features_array),
            'swin_bottleneck': torch.from_numpy(bottleneck_features_array),
            'swin4': torch.from_numpy(swin4_features_array),
            'filenames': all_filenames,
            'feature_types': {
                'latent': '全局特征向量 [N, 512]',
                'swin_bottleneck': '语义特征 [N, C] (经过全局平均池化)',
                'swin4': '平衡特征 [N, C] (经过全局平均池化)'
            },
            'shapes': {
                'latent': latent_features_array.shape,
                'swin_bottleneck': bottleneck_features_array.shape,
                'swin4': swin4_features_array.shape
            },
            'description': '多尺度匹配特征，包含全局、语义和平衡特征'
        }, multi_scale_feature_path)
        print(f"  ✓ 多尺度匹配特征已保存: {multi_scale_feature_path}")
        print(f"    包含: latent ({latent_features_array.shape}), swin_bottleneck ({bottleneck_features_array.shape}), swin4 ({swin4_features_array.shape})")
    
    # 打印相似度统计信息
    if similarity_results:
        print(f"\n{'='*120}")
        print("相似度统计信息汇总")
        print(f"{'='*120}")
        similarities = [r['combined_sim'] for r in similarity_results]
        print(f"\n总体统计:")
        print(f"  比较对数: {len(similarity_results)} 对相邻图像")
        print(f"  距离计算方法: 欧氏距离 (Euclidean Distance / L2 Distance)")
        print(f"  距离公式: d = sqrt(sum((A - B)^2))")
        print(f"  距离范围: [0, +∞)，值越小表示越相似")
        print(f"     - 0.0: 完全相同")
        print(f"     - 值越大: 差异越大")
        
        # 提取距离信息
        distances = [r['latent_distance'] for r in similarity_results]
        print(f"\n  欧氏距离统计（主要指标，距离越小越相似）:")
        print(f"    平均距离: {np.mean(distances):.4f}")
        print(f"    最小距离: {np.min(distances):.4f} (最相似)")
        min_idx = np.argmin(distances)
        print(f"      └─> 图像对: '{similarity_results[min_idx]['image1']}' 与 '{similarity_results[min_idx]['image2']}'")
        print(f"    最大距离: {np.max(distances):.4f} (最不相似)")
        max_idx = np.argmax(distances)
        print(f"      └─> 图像对: '{similarity_results[max_idx]['image1']}' 与 '{similarity_results[max_idx]['image2']}'")
        print(f"    标准差: {np.std(distances):.4f}")
        
        # 转换相似度统计（基于距离转换）
        print(f"\n  转换相似度统计（辅助指标，由距离转换得到）:")
        print(f"    转换公式: 相似度 = 1 / (1 + 距离)")
        print(f"    目的: 将距离转换为相似度形式（值越大越相似）")
        print(f"    相似度范围: [0, 1]")
        print(f"       - 距离 = 0 时，相似度 = 1.0（完全相同）")
        print(f"       - 距离 → +∞ 时，相似度 → 0（完全不同）")
        print(f"    平均相似度: {np.mean(similarities):.4f}")
        print(f"    最大相似度: {np.max(similarities):.4f} (对应最小距离)")
        print(f"    最小相似度: {np.min(similarities):.4f} (对应最大距离)")
        print(f"\n  注意: 欧氏距离是主要指标，转换相似度仅用于直观理解")
        
        if len(similarity_results) > 0 and similarity_results[0].get('bottleneck_distance') is not None:
            latent_dists = [r['latent_distance'] for r in similarity_results]
            bottleneck_dists = [r['bottleneck_distance'] for r in similarity_results]
            swin4_dists = [r['swin4_distance'] for r in similarity_results]
            print(f"\n各尺度平均欧氏距离:")
            print(f"  Latent (全局特征): {np.mean(latent_dists):.4f}")
            print(f"  Bottleneck (语义特征): {np.mean(bottleneck_dists):.4f}")
            print(f"  Swin4 (平衡特征): {np.mean(swin4_dists):.4f}")
            
            latent_sims = [r['latent_sim'] for r in similarity_results]
            bottleneck_sims = [r['bottleneck_sim'] for r in similarity_results]
            swin4_sims = [r['swin4_sim'] for r in similarity_results]
            print(f"\n各尺度平均转换相似度:")
            print(f"  Latent (全局特征): {np.mean(latent_sims):.4f}")
            print(f"  Bottleneck (语义特征): {np.mean(bottleneck_sims):.4f}")
            print(f"  Swin4 (平衡特征): {np.mean(swin4_sims):.4f}")
        
        # 打印所有比较结果的详细列表
        print(f"\n所有比较结果详情:")
        print(f"{'='*120}")
        print(f"{'序号':<6} {'前一张图像':<45} {'当前图像':<45} {'欧氏距离':<12} {'转换相似度':<12} {'说明':<20}")
        print("-" * 120)
        for r in similarity_results:
            dist = r.get('latent_distance', 0)
            sim = r['combined_sim']
            # 计算组合距离（如果有多个尺度）
            if r.get('bottleneck_distance') is not None:
                combined_dist = 0.4 * r['latent_distance'] + 0.35 * r['bottleneck_distance'] + 0.25 * r['swin4_distance']
            else:
                combined_dist = dist
            print(f"{r['idx']:<6} {r['image1']:<45} {r['image2']:<45} {combined_dist:>10.4f} {sim:>12.4f} {'距离越小越相似':<20}")
    
    print(f"\n{'='*60}")
    print("特征提取完成！")
    print(f"{'='*60}")
    
    # 输出合并图像信息
    if similarity_results:
        print(f"\n合并图像保存:")
        print(f"  保存目录: {comparison_output_dir}")
        print(f"  共生成 {len(similarity_results)} 张合并图像")
        print(f"  图像命名格式: comparison_XXXX_图像1_vs_图像2.jpg")
    
    print(f"\n输出文件:")
    print(f"  - global_features.pt (全局匹配特征)")
    print(f"  - multi_scale_features.pt (多尺度匹配特征)")
    print(f"\n共处理 {len(all_filenames)} 个样本")
    print(f"共比较 {len(similarity_results)} 对相邻图像")
    
    # 使用示例
    print(f"\n使用示例:")
    print(f"  加载特征:")
    print(f"    data = torch.load('{os.path.join(output_dir, 'multi_scale_features.pt')}')")
    print(f"    latent = data['latent']  # [N, 512]")
    print(f"    bottleneck = data['swin_bottleneck']  # [N, C]")
    print(f"    swin4 = data['swin4']  # [N, C]")
    print(f"\n  计算相似度:")
    print(f"    # 归一化特征")
    print(f"    latent_norm = F.normalize(latent, p=2, dim=1)")
    print(f"    bottleneck_norm = F.normalize(bottleneck, p=2, dim=1)")
    print(f"    swin4_norm = F.normalize(swin4, p=2, dim=1)")
    print(f"\n    # 计算相似度")
    print(f"    sim_latent = torch.mm(latent_norm[query_idx:query_idx+1], latent_norm.t())[0]")
    print(f"    sim_bottleneck = torch.mm(bottleneck_norm[query_idx:query_idx+1], bottleneck_norm.t())[0]")
    print(f"    sim_swin4 = torch.mm(swin4_norm[query_idx:query_idx+1], swin4_norm.t())[0]")
    print(f"\n    # 组合相似度")
    print(f"    combined_sim = 0.4 * sim_latent + 0.35 * sim_bottleneck + 0.25 * sim_swin4")


def calculate_contour_similarity(
    contour1, 
    contour2, 
    model_path=r'D:\reloc3r\Autoencoder_129.pth',
    image_size=(128, 128),
    device=None
):
    """
    计算两个轮廓的相似度（转换相似度）
    
    输入：两张轮廓图像（前一帧轮廓和后一帧轮廓）
    输出：转换后的相似度值（float，范围[0,1]，值越大越相似）
    
    Args:
        contour1: 前一帧轮廓，可以是：
            - 图像路径（str）
            - numpy数组（np.ndarray，形状为HWC，BGR或RGB格式）
        contour2: 后一帧轮廓，可以是：
            - 图像路径（str）
            - numpy数组（np.ndarray，形状为HWC，BGR或RGB格式）
        model_path: 模型权重文件路径，默认 'E:\\reloc3r\\Autoencoder_129.pth'
        image_size: 图像尺寸 (height, width)，默认 (128, 128)
        device: 计算设备，None则自动选择
    
    Returns:
        float: 转换相似度值
            - 范围: [0, 1]
            - 值越大表示越相似
            - 1.0 表示完全相同
            - 0.0 表示完全不同
            公式: similarity = 1 / (1 + distance)，其中distance为欧氏距离
    
    Examples:
        >>> # 使用图像路径
        >>> similarity = calculate_contour_similarity(
        ...     contour1='L:\\val_Contour_IMGs\\img1.png',
        ...     contour2='L:\\val_Contour_IMGs\\img2.png'
        ... )
        >>> print(f"相似度: {similarity:.4f}")
        
        >>> # 使用numpy数组
        >>> import cv2
        >>> img1 = cv2.imread('path1.png')
        >>> img2 = cv2.imread('path2.png')
        >>> similarity = calculate_contour_similarity(contour1=img1, contour2=img2)
        >>> print(f"相似度: {similarity:.4f}")
    """
    # 初始化模型管理器（单例模式）
    model_manager = ModelManager()
    
    # 如果模型未初始化，则初始化
    try:
        model_manager.get_model()
    except RuntimeError:
        model_manager.initialize(model_path=model_path, image_size=image_size, device=device)
    
    # 获取模型、特征提取器和设备
    feature_extractor = model_manager.get_feature_extractor()
    device = model_manager.get_device()
    
    # 预处理图像
    img1_tensor = model_manager.preprocess_image(contour1).to(device)
    img2_tensor = model_manager.preprocess_image(contour2).to(device)
    
    # 合并为batch [2, 3, H, W]
    inputs = torch.cat([img1_tensor, img2_tensor], dim=0)
    
    # 提取特征
    with torch.no_grad():
        features = feature_extractor.extract_features(inputs)
    
    # 提取两个图像的特征
    # 第一张图像的特征
    latent1 = features['latent'][0:1]  # [1, 512]
    
    # 第二张图像的特征
    latent2 = features['latent'][1:2]  # [1, 512]
    
    # 处理多尺度特征（如果可用）
    bottleneck1 = None
    bottleneck2 = None
    swin4_1 = None
    swin4_2 = None
    
    if 'swin_bottleneck' in features:
        bottleneck_feat1 = features['swin_bottleneck'][0:1]  # [1, C, H, W]
        bottleneck_feat2 = features['swin_bottleneck'][1:2]  # [1, C, H, W]
        bottleneck1 = F.adaptive_avg_pool2d(bottleneck_feat1, (1, 1)).squeeze(-1).squeeze(-1)  # [1, C]
        bottleneck2 = F.adaptive_avg_pool2d(bottleneck_feat2, (1, 1)).squeeze(-1).squeeze(-1)  # [1, C]
    
    if 'swin4' in features:
        swin4_feat1 = features['swin4'][0:1]  # [1, C, H, W]
        swin4_feat2 = features['swin4'][1:2]  # [1, C, H, W]
        swin4_1 = F.adaptive_avg_pool2d(swin4_feat1, (1, 1)).squeeze(-1).squeeze(-1)  # [1, C]
        swin4_2 = F.adaptive_avg_pool2d(swin4_feat2, (1, 1)).squeeze(-1).squeeze(-1)  # [1, C]
    
    # 计算欧氏距离
    latent_distance = torch.norm(latent1 - latent2, p=2).item()
    
    # 计算转换相似度（主要输出）
    # 转换公式: similarity = 1 / (1 + distance)
    # - 距离 = 0 时，相似度 = 1.0（完全相同）
    # - 距离 → +∞ 时，相似度 → 0（完全不同）
    # - 范围: [0, 1]，值越大越相似
    latent_similarity = 1.0 / (1.0 + latent_distance)
    
    # 初始化结果字典
    result = {
        'similarity': latent_similarity,  # 主要输出：转换相似度
        'distance': latent_distance,      # 欧氏距离
        'latent_similarity': latent_similarity,
        'latent_distance': latent_distance,
        'bottleneck_similarity': None,
        'bottleneck_distance': None,
        'swin4_similarity': None,
        'swin4_distance': None,
        'combined_similarity': None,
        'combined_distance': None
    }
    
    # 如果有多尺度特征，计算多尺度距离和相似度
    if bottleneck1 is not None and bottleneck2 is not None:
        bottleneck_distance = torch.norm(bottleneck1 - bottleneck2, p=2).item()
        bottleneck_similarity = 1.0 / (1.0 + bottleneck_distance)
        result['bottleneck_distance'] = bottleneck_distance
        result['bottleneck_similarity'] = bottleneck_similarity
    
    if swin4_1 is not None and swin4_2 is not None:
        swin4_distance = torch.norm(swin4_1 - swin4_2, p=2).item()
        swin4_similarity = 1.0 / (1.0 + swin4_distance)
        result['swin4_distance'] = swin4_distance
        result['swin4_similarity'] = swin4_similarity
    
    # 如果有多尺度特征，计算组合相似度
    if result['bottleneck_similarity'] is not None and result['swin4_similarity'] is not None:
        # 加权组合：latent 0.4, bottleneck 0.35, swin4 0.25
        combined_similarity = (
            0.4 * result['latent_similarity'] + 
            0.35 * result['bottleneck_similarity'] + 
            0.25 * result['swin4_similarity']
        )
        # 使用组合相似度作为主要输出
        result['similarity'] = combined_similarity
    
    # 直接返回转换相似度值
    return result['similarity']


def calculate_contour_similarity_with_grad(
    contour1, 
    contour2, 
    model_path=r'D:\reloc3r\Autoencoder_129.pth',
    image_size=(128, 128),
    device=None
):
    """
    计算两个轮廓的相似度（支持梯度计算版本）
    
    与 calculate_contour_similarity 的区别：
    - 不使用 torch.no_grad()，保持梯度连接
    - 返回tensor而不是float，可以参与反向传播
    - 模型参数设置为 requires_grad=False，不更新模型权重
    
    输入：两张轮廓图像（前一帧轮廓和后一帧轮廓）
    输出：转换后的相似度tensor（范围[0,1]，值越大越相似），支持梯度计算
    
    Args:
        contour1: 前一帧轮廓，可以是：
            - 图像路径（str）
            - numpy数组（np.ndarray，形状为HWC，BGR或RGB格式）
            - torch.Tensor（形状为[1, 3, H, W]或[H, W, 3]）
        contour2: 后一帧轮廓，可以是：
            - 图像路径（str）
            - numpy数组（np.ndarray，形状为HWC，BGR或RGB格式）
            - torch.Tensor（形状为[1, 3, H, W]或[H, W, 3]）
        model_path: 模型权重文件路径，默认 'E:\\reloc3r\\Autoencoder_129.pth'
        image_size: 图像尺寸 (height, width)，默认 (128, 128)
        device: 计算设备，None则自动选择
    
    Returns:
        torch.Tensor: 转换相似度tensor（支持梯度）
            - 形状: [1] 或 scalar tensor
            - 范围: [0, 1]
            - 值越大表示越相似
            - 1.0 表示完全相同
            - 0.0 表示完全不同
            公式: similarity = 1 / (1 + distance)，其中distance为欧氏距离
    """
    # 初始化模型管理器（单例模式）
    model_manager = ModelManager()
    
    # 如果模型未初始化，则初始化
    try:
        model = model_manager.get_model()
    except RuntimeError:
        model_manager.initialize(model_path=model_path, image_size=image_size, device=device)
        model = model_manager.get_model()
    
    # 确保模型参数不更新（只用于特征提取，不更新权重）
    for param in model.parameters():
        param.requires_grad = False
    
    # 获取特征提取器和设备
    feature_extractor = model_manager.get_feature_extractor()
    device = model_manager.get_device()
    
    # 预处理图像
    # 如果输入是tensor，需要特殊处理
    if isinstance(contour1, torch.Tensor):
        # 如果是tensor，直接使用（假设已经预处理）
        if contour1.dim() == 4:  # [B, C, H, W]
            img1_tensor = contour1.to(device)
        elif contour1.dim() == 3:  # [H, W, C] 或 [C, H, W]
            if contour1.shape[0] == 3 or contour1.shape[-1] == 3:
                # 转换为 [1, C, H, W]
                if contour1.shape[-1] == 3:  # [H, W, C]
                    img1_tensor = contour1.permute(2, 0, 1).unsqueeze(0).to(device)
                else:  # [C, H, W]
                    img1_tensor = contour1.unsqueeze(0).to(device)
            else:
                # 单通道，转换为RGB
                img1_tensor = contour1.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        else:
            raise ValueError(f"不支持的tensor形状: {contour1.shape}")
    else:
        img1_tensor = model_manager.preprocess_image(contour1).to(device)
    
    if isinstance(contour2, torch.Tensor):
        # 如果是tensor，直接使用（假设已经预处理）
        if contour2.dim() == 4:  # [B, C, H, W]
            img2_tensor = contour2.to(device)
        elif contour2.dim() == 3:  # [H, W, C] 或 [C, H, W]
            if contour2.shape[0] == 3 or contour2.shape[-1] == 3:
                # 转换为 [1, C, H, W]
                if contour2.shape[-1] == 3:  # [H, W, C]
                    img2_tensor = contour2.permute(2, 0, 1).unsqueeze(0).to(device)
                else:  # [C, H, W]
                    img2_tensor = contour2.unsqueeze(0).to(device)
            else:
                # 单通道，转换为RGB
                img2_tensor = contour2.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        else:
            raise ValueError(f"不支持的tensor形状: {contour2.shape}")
    else:
        img2_tensor = model_manager.preprocess_image(contour2).to(device)
    
    # 合并为batch [2, 3, H, W]
    inputs = torch.cat([img1_tensor, img2_tensor], dim=0)
    
    # 直接提取特征（不使用extract_features方法，因为它会将特征移到CPU）
    # 前向传播（会触发hook，但我们需要直接访问模型输出）
    output, aux_outputs = model(inputs)
    
    # 提取latent特征（全局特征向量）- 保持在GPU上，保持梯度
    latent = model.encode(inputs)  # [B, 512]，保持在GPU上
    
    # 提取两个图像的特征
    # 第一张图像的特征
    latent1 = latent[0:1]  # [1, 512]
    
    # 第二张图像的特征
    latent2 = latent[1:2]  # [1, 512]
    
    # 处理多尺度特征（如果可用）- 从hook捕获的特征中获取
    bottleneck1 = None
    bottleneck2 = None
    swin4_1 = None
    swin4_2 = None
    
    # 检查hook是否捕获了特征
    if hasattr(feature_extractor, 'features') and 'swin_bottleneck' in feature_extractor.features:
        bottleneck_feat = feature_extractor.features['swin_bottleneck']  # [B, C, H, W]
        if bottleneck_feat.device != device:
            bottleneck_feat = bottleneck_feat.to(device)
        bottleneck_feat1 = bottleneck_feat[0:1]  # [1, C, H, W]
        bottleneck_feat2 = bottleneck_feat[1:2]  # [1, C, H, W]
        bottleneck1 = F.adaptive_avg_pool2d(bottleneck_feat1, (1, 1)).squeeze(-1).squeeze(-1)  # [1, C]
        bottleneck2 = F.adaptive_avg_pool2d(bottleneck_feat2, (1, 1)).squeeze(-1).squeeze(-1)  # [1, C]
    
    if hasattr(feature_extractor, 'features') and 'swin4' in feature_extractor.features:
        swin4_feat = feature_extractor.features['swin4']  # [B, C, H, W]
        if swin4_feat.device != device:
            swin4_feat = swin4_feat.to(device)
        swin4_feat1 = swin4_feat[0:1]  # [1, C, H, W]
        swin4_feat2 = swin4_feat[1:2]  # [1, C, H, W]
        swin4_1 = F.adaptive_avg_pool2d(swin4_feat1, (1, 1)).squeeze(-1).squeeze(-1)  # [1, C]
        swin4_2 = F.adaptive_avg_pool2d(swin4_feat2, (1, 1)).squeeze(-1).squeeze(-1)  # [1, C]
    
    # 计算欧氏距离（保持tensor形式，不使用.item()）
    latent_distance = torch.norm(latent1 - latent2, p=2)  # tensor，不是float
    
    # 计算转换相似度（主要输出，保持tensor形式）
    # 转换公式: similarity = 1 / (1 + distance)
    # - 距离 = 0 时，相似度 = 1.0（完全相同）
    # - 距离 → +∞ 时，相似度 → 0（完全不同）
    # - 范围: [0, 1]，值越大越相似
    latent_similarity = 1.0 / (1.0 + latent_distance)
    
    # 如果有多尺度特征，计算组合相似度
    if bottleneck1 is not None and bottleneck2 is not None and swin4_1 is not None and swin4_2 is not None:
        bottleneck_distance = torch.norm(bottleneck1 - bottleneck2, p=2)
        bottleneck_similarity = 1.0 / (1.0 + bottleneck_distance)
        
        swin4_distance = torch.norm(swin4_1 - swin4_2, p=2)
        swin4_similarity = 1.0 / (1.0 + swin4_distance)
        
        # 加权组合：latent 0.4, bottleneck 0.35, swin4 0.25
        combined_similarity = (
            0.4 * latent_similarity + 
            0.35 * bottleneck_similarity + 
            0.25 * swin4_similarity
        )
        return combined_similarity
    else:
        # 只使用latent特征
        return latent_similarity


if __name__ == '__main__':
    main()


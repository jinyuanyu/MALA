"""
MALA项目数据加载模块
==================

本模块提供遥感图像时序修复任务的数据集类，支持多种掩码类型和数据增强方式。

主要功能：
- 加载Sentinel-2遥感图像序列
- 生成多种类型的掩码（云掩码、条带掩码、混合掩码等）
- 支持训练和推理模式的数据加载
- 集成海洋掩码和LaMa初始修复结果

作者: MALA Team
日期: 2024
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple, List, Optional, Dict


class Datasets(Dataset):
    """
    训练数据集类
    
    用于加载遥感图像序列，支持多种掩码生成策略。
    适用于模型训练阶段的数据加载。
    
    属性:
        data_dir: 图像数据目录路径
        max_seq_len: 序列最大长度（帧数）
        ocean_mask_path: 海洋掩码图像路径
        transform: 图像变换组合
    """
    
    def __init__(
        self, 
        data_dir: str, 
        max_seq_len: int = 8,
        ocean_mask_path: Optional[str] = None
    ):
        """
        初始化训练数据集
        
        参数:
            data_dir: 包含图像序列的目录路径
            max_seq_len: 时间序列的最大长度，默认为8帧
            ocean_mask_path: 海洋掩码图像文件路径，用于排除海洋区域
        """
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        
        # 图像变换：转换为张量并归一化到[-1, 1]范围
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 加载数据
        self.img_list, self.fname_list = self._load_data()
        self.mask_list = self._load_mask()
        self.lama_init = self._load_lama_init()
        
        # 检查图像尺寸
        if self.img_list:
            self.img_size = self.img_list[0].size[::-1]  # (height, width)
        else:
            raise ValueError(f"数据目录中未找到图像: {data_dir}")
        
        # 加载海洋掩码（可选）
        if ocean_mask_path is not None:
            ocean_mask = Image.open(ocean_mask_path).convert('L')
            self.ocean_mask = transforms.ToTensor()(ocean_mask).float()
        else:
            raise ValueError("必须提供海洋掩码路径ocean_mask_path")

    def _load_data(self) -> Tuple[List[Image.Image], List[int]]:
        """
        加载数据目录中的所有图像文件
        
        返回:
            img_list: 图像对象列表
            fname_list: 对应的文件名列表（提取为整数）
        """
        img_list = []
        fname_list = []
        
        for frame_file in sorted(os.listdir(self.data_dir)):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frame_path = os.path.join(self.data_dir, frame_file)
                image = Image.open(frame_path)
                img_list.append(image)
                
                # 从文件名提取时间标识
                fname = int(frame_file.split('.')[0].split('_')[-1])
                fname_list.append(fname)
        
        return img_list, fname_list

    def _load_mask(self) -> Optional[torch.Tensor]:
        """
        加载预定义的掩码图像序列
        
        掩码图像应存放在固定路径 E:/lama/mask_img 目录下
        
        返回:
            掩码张量，形状为 (num_masks, 1, height, width)，值域[0, 1]
        """
        mask_dir = 'E:/lama/mask_img'
        if not os.path.exists(mask_dir):
            return None
            
        mask_list = []
        for frame_file in sorted(os.listdir(mask_dir)):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frame_path = os.path.join(mask_dir, frame_file)
                mask = Image.open(frame_path).convert('L')
                mask_list.append(mask)
        
        if mask_list:
            mask_array = np.array([np.array(mask) for mask in mask_list])
            mask_tensor = torch.tensor(mask_array / 255, dtype=torch.float32).unsqueeze(1)
            return mask_tensor
        
        return None

    def _load_lama_init(self) -> List[Image.Image]:
        """
        加载LaMa模型的初始修复结果
        
        用于提供修复的初始猜测，改善修复效果
        
        返回:
            LaMa修复结果图像列表
        """
        lama_dir = 'E:/lama/inpainted_img/lama_init'
        if not os.path.exists(lama_dir):
            return []
            
        lama_init = []
        for frame_file in sorted(os.listdir(lama_dir)):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frame_path = os.path.join(lama_dir, frame_file)
                image = Image.open(frame_path)
                lama_init.append(image)
        
        return lama_init

    def __len__(self) -> int:
        """返回数据集可划分的序列数量"""
        return len(self.img_list) // self.max_seq_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取指定索引的训练样本
        
        参数:
            idx: 样本索引
            
        返回:
            包含以下键的字典:
            - video: 原始视频帧序列
            - masked: 掩码处理后的视频帧序列
            - mask: 掩码张量
            - times: 时间标识
            - lama_init: LaMa初始修复结果
            - ocean_mask: 海洋掩码
        """
        start_idx = idx * self.max_seq_len
        end_idx = min(start_idx + self.max_seq_len, len(self.img_list))
        
        frames = self.img_list[start_idx:end_idx]
        fnames = self.fname_list[start_idx:end_idx]
        lama_init = self.lama_init[:]
        
        # 根据数据路径选择掩码生成策略
        if self.data_dir == 'E:/lama/masked_img/test_img/':
            mask = self._generate_random_mask()
        elif self.data_dir == 'E:/lama/jet_S2_Daily_Mosaic/' and self.mask_list is not None:
            mask = self.mask_list[start_idx:end_idx]
        else:
            raise ValueError('请为此数据路径设置正确的掩码策略')
        
        # 填充序列以达到最大长度
        if len(frames) < self.max_seq_len:
            frames += [frames[-1]] * (self.max_seq_len - len(frames))
        
        # 应用图像变换
        frames = [self.transform(frame) for frame in frames]
        fnames = torch.tensor(fnames, dtype=torch.int64)
        
        # 处理LaMa初始结果
        if lama_init:
            lama_init = [self.transform(frame) for frame in lama_init]
            lama_init = torch.stack(lama_init, dim=0)
        else:
            lama_init = torch.zeros((self.max_seq_len, 3, *self.img_size[::-1]))
        
        # 组合视频张量并应用掩码
        video = torch.stack(frames, dim=0)
        masked_video = video * (1 - mask)
        
        return {
            'video': video,
            'masked': masked_video,
            'mask': mask,
            'times': fnames,
            'lama_init': lama_init,
            'ocean_mask': self.ocean_mask
        }

    def _generate_random_mask(self) -> torch.Tensor:
        """
        生成随机掩码
        
        使用三种策略随机生成掩码：
        - 策略0: 单个大型连续云块
        - 策略1: 多个小型分散云块
        - 策略2: 每帧随机生成云块
        
        返回:
            形状为 (max_seq_len, 1, height, width) 的掩码张量
        """
        mask_type = np.random.randint(0, 3)
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        
        if mask_type == 0:
            # 单个大型连续云块
            num_clouds = np.random.randint(1, 4)
            for _ in range(num_clouds):
                size_range = (
                    max(10, int(min(self.img_size) * 0.1)),
                    int(min(self.img_size) * 0.4)
                )
                cloud_patch = self._generate_cloud_patch(size_range)
                if cloud_patch is not None:
                    h, w = cloud_patch.shape
                    h_start = np.random.randint(0, self.img_size[0] - h)
                    w_start = np.random.randint(0, self.img_size[1] - w)
                    t_start = np.random.randint(0, self.max_seq_len - 1)
                    t_end = min(t_start + np.random.randint(1, 4), self.max_seq_len)
                    mask[t_start:t_end, 0, h_start:h_start+h, w_start:w_start+w] = \
                        torch.from_numpy(cloud_patch).float()
        
        elif mask_type == 1:
            # 多个小型分散云块
            num_clouds = np.random.randint(5, 16)
            for _ in range(num_clouds):
                size_range = (
                    max(15, int(min(self.img_size) * 0.05)),
                    int(min(self.img_size) * 0.2)
                )
                cloud_patch = self._generate_cloud_patch(size_range)
                if cloud_patch is not None:
                    h, w = cloud_patch.shape
                    h_start = np.random.randint(0, self.img_size[0] - h)
                    w_start = np.random.randint(0, self.img_size[1] - w)
                    t = np.random.randint(0, self.max_seq_len)
                    mask[t, 0, h_start:h_start+h, w_start:w_start+w] = \
                        torch.from_numpy(cloud_patch).float()
        
        else:
            # 每帧随机生成云块
            for t in range(self.max_seq_len):
                if np.random.rand() < 0.5:
                    num_clouds = np.random.randint(1, 4)
                    for _ in range(num_clouds):
                        size_range = (
                            max(15, int(min(self.img_size) * 0.05)),
                            int(min(self.img_size) * 0.3)
                        )
                        cloud_patch = self._generate_cloud_patch(size_range)
                        if cloud_patch is not None:
                            h, w = cloud_patch.shape
                            h_start = np.random.randint(0, self.img_size[0] - h)
                            w_start = np.random.randint(0, self.img_size[1] - w)
                            mask[t, 0, h_start:h_start+h, w_start:w_start+w] = \
                                torch.from_numpy(cloud_patch).float()
        
        return mask

    def _generate_cloud_patch(self, size_range: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        生成单个云块图案
        
        使用多个重叠圆形和高斯模糊创建自然的云块形状
        
        参数:
            size_range: 云块尺寸范围 (min_size, max_size)
            
        返回:
            云块掩码数组，值为0或1
        """
        size = np.random.randint(size_range[0], size_range[1] + 1)
        canvas_size = size * 3
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        
        # 生成多个重叠圆形模拟云团
        num_circles = np.random.randint(3, 7)
        for _ in range(num_circles):
            cx = canvas_size // 2 + np.random.randint(-size//2, size//2)
            cy = canvas_size // 2 + np.random.randint(-size//2, size//2)
            radius = np.random.randint(size//4, size//2)
            cv2.circle(canvas, (cx, cy), radius, 1, -1)
        
        # 高斯模糊使边缘平滑
        blur_size = min(11, size // 5)
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(canvas, (blur_size, blur_size), 0)
        
        # 二值化处理
        _, cloud_patch = cv2.threshold(blurred, 0.3, 1, cv2.THRESH_BINARY)
        cloud_patch = cloud_patch.astype(np.uint8)
        
        # 形态学操作：膨胀和腐蚀
        kernel_size = max(3, size // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cloud_patch = cv2.dilate(cloud_patch, kernel, iterations=1)
        cloud_patch = cv2.erode(cloud_patch, kernel, iterations=1)
        
        # 裁剪到有效区域
        rows, cols = np.where(cloud_patch)
        if len(rows) == 0:
            return None
            
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        cropped = cloud_patch[min_row:max_row+1, min_col:max_col+1]
        
        # 随机尺度变换
        scale_factor = np.random.uniform(0.8, 1.2)
        new_h = max(1, int(cropped.shape[0] * scale_factor))
        new_w = max(1, int(cropped.shape[1] * scale_factor))
        
        if new_h > 0 and new_w > 0:
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            _, resized = cv2.threshold(resized, 0.5, 1, cv2.THRESH_BINARY)
            return resized.astype(np.uint8)
        
        return cropped


class Datasets_inference(Dataset):
    """
    推理数据集类
    
    用于模型推理/测试阶段，支持多种预定义掩码类型。
    相比训练数据集，提供更灵活的掩码选择。
    
    属性:
        data_dir: 图像数据目录路径
        max_seq_len: 序列最大长度
        mask_type: 掩码类型 ('random', 'cloud', 'strip', 'mixed')
        mask_ratio: 掩码缺失比例 (0.0-1.0)
    """
    
    def __init__(
        self, 
        data_dir: str, 
        max_seq_len: int = 8,
        ocean_mask_path: Optional[str] = None,
        mask_type: str = "random",
        mask_ratio: float = 0.5
    ):
        """
        初始化推理数据集
        
        参数:
            data_dir: 包含图像序列的目录路径
            max_seq_len: 时间序列的最大长度
            ocean_mask_path: 海洋掩码图像路径
            mask_type: 掩码类型，可选 'random', 'cloud', 'strip', 'mixed', 'predefined'
            mask_ratio: 掩码缺失比例，范围0.0-1.0
        """
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio
        
        # 图像变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 加载数据
        self.img_list, self.fname_list = self._load_data()
        self.mask_list = self._load_mask()
        self.lama_init = self._load_lama_init()
        
        if self.img_list:
            self.img_size = self.img_list[0].size[::-1]
        else:
            raise ValueError(f"数据目录中未找到图像: {data_dir}")
        
        # 加载海洋掩码
        if ocean_mask_path is not None:
            ocean_mask = Image.open(ocean_mask_path).convert('L')
            if hasattr(self, 'img_size'):
                ocean_mask = ocean_mask.resize((self.img_size[0], self.img_size[1]))
            self.ocean_mask = transforms.ToTensor()(ocean_mask).float()
        else:
            self.ocean_mask = None

    def _load_data(self) -> Tuple[List[Image.Image], List[int]]:
        """加载数据目录中的所有图像文件"""
        img_list = []
        fname_list = []
        
        for frame_file in sorted(os.listdir(self.data_dir)):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frame_path = os.path.join(self.data_dir, frame_file)
                image = Image.open(frame_path)
                img_list.append(image)
                fname = int(frame_file.split('.')[0].split('_')[-1])
                fname_list.append(fname)
        
        return img_list, fname_list

    def _load_mask(self) -> Optional[torch.Tensor]:
        """加载预定义的掩码图像序列"""
        mask_dir = 'E:/lama/mask_img'
        if not os.path.exists(mask_dir):
            return None
            
        mask_list = []
        for frame_file in sorted(os.listdir(mask_dir)):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frame_path = os.path.join(mask_dir, frame_file)
                mask = Image.open(frame_path).convert('L')
                mask_list.append(mask)
        
        if mask_list:
            mask_array = np.array([np.array(mask) for mask in mask_list])
            mask_tensor = torch.tensor(mask_array / 255, dtype=torch.float32).unsqueeze(1)
            return mask_tensor
        
        return None

    def _load_lama_init(self) -> List[Image.Image]:
        """加载LaMa模型的初始修复结果"""
        lama_dir = 'E:/lama/inpainted_img/lama_init'
        if not os.path.exists(lama_dir):
            return []
            
        lama_init = []
        for frame_file in sorted(os.listdir(lama_dir)):
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                frame_path = os.path.join(lama_dir, frame_file)
                image = Image.open(frame_path)
                lama_init.append(image)
        
        return lama_init

    def __len__(self) -> int:
        return len(self.img_list) // self.max_seq_len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取指定索引的推理样本"""
        start_idx = idx * self.max_seq_len
        end_idx = min(start_idx + self.max_seq_len, len(self.img_list))
        
        frames = self.img_list[start_idx:end_idx]
        fnames = self.fname_list[start_idx:end_idx]
        lama_init = self.lama_init[:] if self.lama_init else []
        
        # 根据mask_type生成掩码
        if self.mask_type == "predefined" and self.mask_list is not None:
            mask = self.mask_list[start_idx:end_idx]
        else:
            mask = self._generate_mask_by_type(self.mask_type, self.mask_ratio)
        
        # 填充序列
        if len(frames) < self.max_seq_len:
            frames += [frames[-1]] * (self.max_seq_len - len(frames))
        
        # 应用变换
        frames = [self.transform(frame) for frame in frames]
        fnames = torch.tensor(fnames, dtype=torch.int64)
        
        # 处理LaMa初始结果
        if lama_init:
            lama_init = [self.transform(frame) for frame in lama_init]
            lama_init = torch.stack(lama_init, dim=0)
        else:
            lama_init = torch.zeros((self.max_seq_len, 3, *self.img_size[::-1])) \
                       if self.img_list else torch.zeros((self.max_seq_len, 3, 224, 224))
        
        # 组合视频张量并应用掩码
        video = torch.stack(frames, dim=0)
        masked_video = video * (1 - mask)
        
        ocean_mask_tensor = self.ocean_mask if self.ocean_mask is not None \
                           else torch.zeros((1, *self.img_size[::-1]))
        
        return {
            'video': video,
            'masked': masked_video,
            'mask': mask,
            'times': fnames,
            'lama_init': lama_init,
            'ocean_mask': ocean_mask_tensor
        }

    def _generate_mask_by_type(self, mask_type: str, mask_ratio: float) -> torch.Tensor:
        """
        根据掩码类型生成相应的掩码
        
        参数:
            mask_type: 掩码类型
            mask_ratio: 掩码缺失比例
            
        返回:
            掩码张量
        """
        if mask_type == "random":
            return self._generate_thin_cloud_mask(mask_ratio)
        elif mask_type == "thin_cloud":
            return self._generate_thin_cloud_mask(mask_ratio)
        elif mask_type == "cloud":
            return self._generate_cloud_mask(mask_ratio)
        elif mask_type == "strip":
            return self._generate_strip_mask(mask_ratio)
        elif mask_type == "mixed":
            return self._generate_mixed_mask(mask_ratio)
        else:
            raise ValueError(f"未知的掩码类型: {mask_type}")

    def _generate_cloud_mask(self, mask_ratio: float) -> torch.Tensor:
        """生成云状掩码"""
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        total_pixels = self.img_size[0] * self.img_size[1]
        target_masked_pixels = int(total_pixels * mask_ratio)
        
        for t in range(self.max_seq_len):
            current_masked = 0
            attempts = 0
            max_attempts = 100
            
            while current_masked < target_masked_pixels and attempts < max_attempts:
                cloud_size = np.random.randint(
                    min(100, self.img_size[0] * mask_ratio), 
                    min(200, self.img_size[1] * mask_ratio)
                )
                
                cloud_patch = self._generate_cloud_patch((cloud_size, cloud_size))
                if cloud_patch is None:
                    attempts += 1
                    continue
                    
                h, w = cloud_patch.shape
                if h >= self.img_size[0] or w >= self.img_size[1]:
                    attempts += 1
                    continue
                
                h_start = np.random.randint(0, self.img_size[0] - h)
                w_start = np.random.randint(0, self.img_size[1] - w)
                
                new_cloud = torch.from_numpy(cloud_patch).float()
                existing_region = mask[t, 0, h_start:h_start+h, w_start:w_start+w]
                new_pixels = torch.sum(new_cloud * (1 - existing_region)).item()
                
                mask[t, 0, h_start:h_start+h, w_start:w_start+w] = torch.max(
                    existing_region, new_cloud
                )
                
                current_masked += new_pixels
                attempts += 1
        
        return mask

    def _generate_strip_mask(self, mask_ratio: float) -> torch.Tensor:
        """生成条带掩码"""
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        
        strip_width = max(1, int(self.img_size[1] * 0.05))
        num_strips = int(self.img_size[1] * mask_ratio / strip_width)
        
        for t in range(self.max_seq_len):
            for _ in range(num_strips):
                strip_pos = np.random.randint(0, self.img_size[1] - strip_width)
                orientation = np.random.choice(['horizontal', 'vertical'])
                
                if orientation == 'horizontal':
                    mask[t, 0, :, strip_pos:strip_pos+strip_width] = 1
                else:
                    mask[t, 0, strip_pos:strip_pos+strip_width, :] = 1
        
        return mask

    def _generate_mixed_mask(self, mask_ratio: float) -> torch.Tensor:
        """生成混合掩码（云+薄云+条带）"""
        cloud_ratio = mask_ratio * 0.5
        thin_cloud_ratio = mask_ratio * 0.3
        strip_ratio = mask_ratio * 0.2
        
        cloud_mask = self._generate_cloud_mask(cloud_ratio)
        thin_cloud_mask = self._generate_thin_cloud_mask(thin_cloud_ratio)
        strip_mask = self._generate_strip_mask(strip_ratio)
        
        combined_mask = torch.clamp(cloud_mask + thin_cloud_mask + strip_mask, 0, 1)
        return combined_mask

    def _generate_cloud_patch(self, size_range: Tuple[int, int]) -> Optional[np.ndarray]:
        """生成单个云块图案"""
        if len(size_range) != 2:
            size = size_range[0] if hasattr(size_range, '__len__') else size_range
        else:
            size = np.random.randint(size_range[0], size_range[1] + 1)
        
        canvas_size = max(size * 2, 50)
        canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        
        num_circles = np.random.randint(3, 8)
        for _ in range(num_circles):
            cx = canvas_size // 2 + np.random.randint(-size//3, size//3)
            cy = canvas_size // 2 + np.random.randint(-size//3, size//3)
            radius = np.random.randint(size//6, size//2)
            cv2.circle(canvas, (cx, cy), radius, 1, -1)
        
        blur_size = max(5, size // 8)
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(canvas, (blur_size, blur_size), 0)
        
        _, cloud_patch = cv2.threshold(blurred, 0.2, 1, cv2.THRESH_BINARY)
        
        kernel_size = max(3, size // 15)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cloud_patch = cv2.morphologyEx(cloud_patch, cv2.MORPH_CLOSE, kernel)
        
        rows, cols = np.where(cloud_patch > 0)
        if len(rows) == 0:
            return None
            
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        cropped = cloud_patch[min_row:max_row+1, min_col:max_col+1]
        
        return cropped.astype(np.uint8)

    def _generate_thin_cloud_mask(self, mask_ratio: float) -> torch.Tensor:
        """生成薄云掩码"""
        mask = torch.zeros((self.max_seq_len, 1, *self.img_size))
        
        for t in range(self.max_seq_len):
            cloud_mask = self._generate_thin_cloud_pattern(mask_ratio)
            mask[t, 0] = torch.from_numpy(cloud_mask).float()
        
        return mask

    def _generate_thin_cloud_pattern(self, mask_ratio: float) -> np.ndarray:
        """生成薄云图案（使用多尺度噪声）"""
        h, w = self.img_size
        
        large_scale = self._generate_noise_layer(h, w, scale=0.01, octaves=3)
        medium_scale = self._generate_noise_layer(h, w, scale=0.05, octaves=2)
        small_scale = self._generate_noise_layer(h, w, scale=0.1, octaves=1)
        
        cloud_pattern = (0.6 * large_scale + 0.3 * medium_scale + 0.1 * small_scale)
        cloud_pattern = (cloud_pattern - cloud_pattern.min()) / (cloud_pattern.max() - cloud_pattern.min() + 1e-8)
        
        threshold = np.percentile(cloud_pattern, (1 - mask_ratio) * 100)
        thin_cloud_mask = (cloud_pattern > threshold).astype(np.float32)
        
        if np.random.random() > 0.5:
            kernel_size = max(3, min(h, w) // 100)
            if kernel_size % 2 == 0:
                kernel_size += 1
            thin_cloud_mask = cv2.GaussianBlur(thin_cloud_mask, (kernel_size, kernel_size), 0)
            thin_cloud_mask = (thin_cloud_mask > 0.5).astype(np.float32)
        
        return thin_cloud_mask

    def _generate_noise_layer(self, h: int, w: int, scale: float = 0.05, octaves: int = 1) -> np.ndarray:
        """生成噪声层（模拟Perlin噪声）"""
        noise = np.random.randn(max(1, int(h * scale)), max(1, int(w * scale)))
        noise_resized = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        
        for _ in range(octaves):
            kernel_size = max(3, min(h, w) // 50)
            if kernel_size % 2 == 0:
                kernel_size += 1
            noise_resized = cv2.GaussianBlur(noise_resized, (kernel_size, kernel_size), 0)
        
        return noise_resized

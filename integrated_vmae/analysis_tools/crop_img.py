import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class ImageAnalyzer:
    def __init__(self, experiment_root):
        """
        初始化分析器
        
        Args:
            experiment_root: 实验根目录路径
        """
        self.experiment_root = Path(experiment_root)
        self.algorithms = []
        self.experiment_dirs = []
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 算法名称到中文的映射
        self.algorithm_name_map = {
            'DINEOF': '数据插值（DINEOF）',
            'Proposed': '掩码自编码（MaskAE）',
            'Lama': 'LaMA',
            'Spline': '样条插值（Spline）',
            'Nearest_Neighbor': '最近邻插值（NN）',
            'EMAE': '方法一（EMAE）',
            'MALA': '方法二（MALA）'
        }
        
    def discover_experiments_and_algorithms(self):
        """
        自动发现实验目录和算法
        """
        self.experiment_dirs = []
        self.algorithms = set()
        
        # 遍历实验根目录
        for exp_dir in self.experiment_root.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith('exp'):
                self.experiment_dirs.append(exp_dir)
                
                # 遍历每个实验目录下的场景
                for scene_dir in exp_dir.iterdir():
                    if scene_dir.is_dir():
                        images_dir = scene_dir / 'images'
                        if images_dir.exists():
                            # 发现算法目录
                            for item in images_dir.iterdir():
                                if item.is_dir() and item.name in self.algorithm_name_map:
                                    self.algorithms.add(item.name)
        
        self.algorithms = sorted(list(self.algorithms))
        print(f"发现实验目录: {[d.name for d in self.experiment_dirs]}")
        print(f"发现算法: {[self.algorithm_name_map.get(alg, alg) for alg in self.algorithms]}")
    
    def load_image_sequence(self, image_dir, algorithm=None, as_color=True):
        """
        根据算法加载图像序列
        
        Args:
            image_dir: 图像目录路径
            algorithm: 算法名称，None表示原始图像或mask
            as_color: 是否以彩色模式读取图像
            
        Returns:
            tuple: (图像数组列表, 文件名列表)
        """
        images = []
        filenames = []
        
        # 选择读取模式
        read_flag = cv2.IMREAD_COLOR if as_color else cv2.IMREAD_GRAYSCALE
        
        if algorithm is None:
            # 加载原始图像或mask
            frame_files = sorted([f for f in os.listdir(image_dir) 
                                 if f.endswith('.png') and ('original_frame' in f or 'mask_frame' in f)])
        else:
            # 根据算法名称确定文件命名规则
            alg_dir = Path(image_dir) / algorithm
            if not alg_dir.exists():
                return None, None
                
            naming_rules = {
                'Lama': 'lama_inpainted_frame',
                'EMAE': 'emae_inpainted_frame', 
                'MALA': 'mala_inpainted_frame',
                'DINEOF': 'frame',
                'Nearest_Neighbor': 'frame',
                'Proposed': 'frame',
                'Spline': 'frame'
            }
            
            prefix = naming_rules.get(algorithm, 'frame')
            frame_files = sorted([f for f in os.listdir(alg_dir) 
                                 if f.startswith(prefix) and f.endswith('.png')])
            image_dir = alg_dir
        
        for frame_file in frame_files:
            img_path = os.path.join(image_dir, frame_file)
            img = cv2.imread(img_path, read_flag)
            if img is not None:
                images.append(img)
                filenames.append(frame_file)
        
        return np.array(images), filenames if images else (None, None)
    
    def load_specific_sequence(self, images_dir, sequence_type, as_color=True):
        """
        加载特定类型的图像序列
        
        Args:
            images_dir: 图像目录路径
            sequence_type: 序列类型 ('original', 'mask', 'masked')
            as_color: 是否以彩色模式读取图像
            
        Returns:
            tuple: (图像序列, 文件名列表)
        """
        frame_files = []
        prefix_map = {
            'original': 'original_frame',
            'mask': 'mask_frame', 
            # 'masked': 'masked_frame'
            'masked': 'white_fill_masked_frame'
        }
        
        # 选择读取模式
        read_flag = cv2.IMREAD_COLOR if as_color else cv2.IMREAD_GRAYSCALE
        
        prefix = prefix_map[sequence_type]
        all_files = os.listdir(images_dir)
        
        # 找到所有匹配的帧文件并按帧号排序
        frame_numbers = []
        for f in all_files:
            if f.startswith(prefix) and f.endswith('.png'):
                # 提取帧号
                frame_num_str = f[len(prefix):-4]  # 去掉前缀和.png
                if frame_num_str.isdigit():
                    frame_numbers.append(int(frame_num_str))
        
        frame_numbers.sort()
        
        images = []
        filenames = []
        for frame_num in frame_numbers:
            filename = f"{prefix}{frame_num:02d}.png"
            img_path = os.path.join(images_dir, filename)
            if os.path.exists(img_path):
                img = cv2.imread(img_path, read_flag)
                if img is not None:
                    images.append(img)
                    filenames.append(filename)
        
        return np.array(images), filenames if images else (None, None)
    
    def create_marked_image(self, image, rect_points, color=(0, 255, 0), thickness=3):
        """
        创建带有裁剪区域标记的图像
        
        Args:
            image: 输入图像
            rect_points: 矩形区域的四个点坐标
            color: 标记颜色 (B, G, R)
            thickness: 线条粗细
            
        Returns:
            带标记的图像
        """
        # 确保图像是彩色的
        if len(image.shape) == 2:
            marked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            marked_image = image.copy()
        
        # 计算矩形边界
        x_coords = [p[0] for p in rect_points]
        y_coords = [p[1] for p in rect_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 确保坐标为整数
        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)
        
        # 绘制矩形框（不添加文字）
        cv2.rectangle(marked_image, (x_min, y_min), (x_max, y_max), color, thickness)
        
        return marked_image
    
    def crop_region_from_images(self, rect_points, output_dir=None, save_marked_images=True):
        """
        从所有图像中裁剪指定矩形区域，输出三组图像：原始图像组、masked图像组和算法结果图像组
        
        Args:
            rect_points: 矩形区域的四个点坐标，格式为 [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            output_dir: 输出目录，如果为None则不保存图像
            save_marked_images: 是否保存带有裁剪区域标记的图像
            
        Returns:
            dict: 包含所有裁剪后图像的字典，按算法和场景分类
        """
        # 计算矩形边界
        x_coords = [p[0] for p in rect_points]
        y_coords = [p[1] for p in rect_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 确保坐标为整数
        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)
        
        print(f"裁剪区域: x={x_min}:{x_max}, y={y_min}:{y_max}")
        
        # 创建输出目录
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储所有裁剪后的图像
        cropped_images = {}
        
        # 遍历所有实验目录和场景
        for exp_dir in self.experiment_dirs:
            exp_name = exp_dir.name
            cropped_images[exp_name] = {}
            
            for scene_dir in exp_dir.iterdir():
                if scene_dir.is_dir():
                    scene_name = scene_dir.name
                    cropped_images[exp_name][scene_name] = {}
                    images_dir = scene_dir / 'images'
                    
                    # 第1组：处理原始图像（彩色）
                    original_images, original_filenames = self.load_specific_sequence(images_dir, 'original', as_color=True)
                    if original_images is not None:
                        cropped_original = []
                        for i, img in enumerate(original_images):
                            # 裁剪区域
                            cropped = img[y_min:y_max, x_min:x_max]
                            cropped_original.append(cropped)
                            
                            # 保存裁剪后的图像
                            if output_dir is not None:
                                exp_output_dir = output_dir / exp_name / scene_name / 'original'
                                exp_output_dir.mkdir(parents=True, exist_ok=True)
                                output_path = exp_output_dir / original_filenames[i]
                                cv2.imwrite(str(output_path), cropped)
                                
                                # 保存带标记的原始图像
                                if save_marked_images:
                                    marked_dir = output_dir / exp_name / scene_name / 'original_marked'
                                    marked_dir.mkdir(parents=True, exist_ok=True)
                                    marked_image = self.create_marked_image(img, rect_points)
                                    marked_path = marked_dir / f"marked_{original_filenames[i]}"
                                    cv2.imwrite(str(marked_path), marked_image)
                        
                        cropped_images[exp_name][scene_name]['original'] = cropped_original
                    
                    # 第2组：处理masked图像（彩色）
                    masked_images, masked_filenames = self.load_specific_sequence(images_dir, 'masked', as_color=True)
                    if masked_images is not None:
                        cropped_masked = []
                        for i, img in enumerate(masked_images):
                            # 裁剪区域
                            cropped = img[y_min:y_max, x_min:x_max]
                            cropped_masked.append(cropped)
                            
                            # 保存裁剪后的图像
                            if output_dir is not None:
                                exp_output_dir = output_dir / exp_name / scene_name / 'masked'
                                exp_output_dir.mkdir(parents=True, exist_ok=True)
                                output_path = exp_output_dir / masked_filenames[i]
                                cv2.imwrite(str(output_path), cropped)
                                
                                # 保存带标记的图像
                                if save_marked_images:
                                    marked_dir = output_dir / exp_name / scene_name / 'masked_marked'
                                    marked_dir.mkdir(parents=True, exist_ok=True)
                                    marked_image = self.create_marked_image(img, rect_points)
                                    marked_path = marked_dir / f"marked_{masked_filenames[i]}"
                                    cv2.imwrite(str(marked_path), marked_image)
                        
                        cropped_images[exp_name][scene_name]['masked'] = cropped_masked
                    
                    # 第3组：处理各算法输出的结果图像（彩色）
                    for algorithm in self.algorithms:
                        alg_images, alg_filenames = self.load_image_sequence(images_dir, algorithm, as_color=True)
                        if alg_images is not None:
                            cropped_alg = []
                            for i, img in enumerate(alg_images):
                                # 如果是灰度图，转换为彩色
                                if len(img.shape) == 2:
                                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                                
                                # 裁剪区域
                                cropped = img[y_min:y_max, x_min:x_max]
                                cropped_alg.append(cropped)
                                
                                # 保存裁剪后的图像
                                if output_dir is not None:
                                    exp_output_dir = output_dir / exp_name / scene_name / algorithm
                                    exp_output_dir.mkdir(parents=True, exist_ok=True)
                                    output_path = exp_output_dir / alg_filenames[i]
                                    cv2.imwrite(str(output_path), cropped)
                                    
                                    # 保存带标记的图像
                                    if save_marked_images:
                                        marked_dir = output_dir / exp_name / scene_name / f"{algorithm}_marked"
                                        marked_dir.mkdir(parents=True, exist_ok=True)
                                        marked_image = self.create_marked_image(img, rect_points)
                                        marked_path = marked_dir / f"marked_{alg_filenames[i]}"
                                        cv2.imwrite(str(marked_path), marked_image)
                            
                            cropped_images[exp_name][scene_name][algorithm] = cropped_alg
        
        return cropped_images

    def visualize_cropped_region(self, rect_points, sample_image_path=None):
        """
        可视化裁剪区域
        
        Args:
            rect_points: 矩形区域的四个点坐标
            sample_image_path: 示例图像路径，用于显示裁剪区域
        """
        # 计算矩形边界
        x_coords = [p[0] for p in rect_points]
        y_coords = [p[1] for p in rect_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 如果没有提供示例图像，尝试从第一个实验的第一个场景获取一张原始图像
        if sample_image_path is None and self.experiment_dirs:
            first_exp = self.experiment_dirs[0]
            first_scene = next(first_exp.iterdir())
            images_dir = first_scene / 'images'
            original_images, _ = self.load_specific_sequence(images_dir, 'original', as_color=True)
            if original_images is not None:
                sample_image = original_images[0]
            else:
                print("无法获取示例图像")
                return
        else:
            sample_image = cv2.imread(sample_image_path, cv2.IMREAD_COLOR)
        
        # 创建带裁剪区域标记的图像
        marked_image = self.create_marked_image(sample_image, rect_points)
        
        # 显示图像
        plt.figure(figsize=(15, 6))
        
        # 显示带标记的原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
        plt.title("裁剪区域标记")
        plt.axis('off')
        
        # 显示原始图像
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis('off')
        
        # 显示裁剪后的区域
        cropped_region = sample_image[y_min:y_max, x_min:x_max]
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
        plt.title("裁剪区域")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def create_comparison_visualization(self, rect_points, exp_name, scene_name, frame_index=0):
        """
        创建算法对比可视化，显示原始图像、masked图像和各算法的裁剪结果
        
        Args:
            rect_points: 矩形区域坐标
            exp_name: 实验名称
            scene_name: 场景名称
            frame_index: 帧索引
        """
        exp_dir = None
        for exp in self.experiment_dirs:
            if exp.name == exp_name:
                exp_dir = exp
                break
        
        if exp_dir is None:
            print(f"未找到实验: {exp_name}")
            return
        
        scene_dir = exp_dir / scene_name
        if not scene_dir.exists():
            print(f"未找到场景: {scene_name}")
            return
        
        images_dir = scene_dir / 'images'
        
        # 计算矩形边界
        x_coords = [p[0] for p in rect_points]
        y_coords = [p[1] for p in rect_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 加载图像
        original_images, _ = self.load_specific_sequence(images_dir, 'original', as_color=True)
        masked_images, _ = self.load_specific_sequence(images_dir, 'masked', as_color=True)
        
        if original_images is None:
            print("无法加载原始图像")
            return
        
        # 计算子图布局
        num_algorithms = len(self.algorithms)
        total_images = num_algorithms + 1  # +1 for original
        if masked_images is not None:
            total_images += 1  # +1 for masked
        
        cols = min(4, total_images)
        rows = (total_images + cols - 1) // cols
        
        plt.figure(figsize=(cols * 4, rows * 3))
        
        plot_idx = 1
        
        # 显示原始图像裁剪区域
        plt.subplot(rows, cols, plot_idx)
        original_cropped = original_images[frame_index][y_min:y_max, x_min:x_max]
        plt.imshow(cv2.cvtColor(original_cropped, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis('off')
        plot_idx += 1
        
        # 显示masked图像裁剪区域（如果存在）
        if masked_images is not None:
            plt.subplot(rows, cols, plot_idx)
            masked_cropped = masked_images[frame_index][y_min:y_max, x_min:x_max]
            plt.imshow(cv2.cvtColor(masked_cropped, cv2.COLOR_BGR2RGB))
            plt.title("Masked图像")
            plt.axis('off')
            plot_idx += 1
        
        # 显示各算法的裁剪结果
        for algorithm in self.algorithms:
            alg_images, _ = self.load_image_sequence(images_dir, algorithm, as_color=True)
            if alg_images is not None and len(alg_images) > frame_index:
                plt.subplot(rows, cols, plot_idx)
                alg_img = alg_images[frame_index]
                if len(alg_img.shape) == 2:
                    alg_img = cv2.cvtColor(alg_img, cv2.COLOR_GRAY2BGR)
                alg_cropped = alg_img[y_min:y_max, x_min:x_max]
                plt.imshow(cv2.cvtColor(alg_cropped, cv2.COLOR_BGR2RGB))
                plt.title(self.algorithm_name_map.get(algorithm, algorithm))
                plt.axis('off')
            plot_idx += 1
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 初始化分析器
    analyzer = ImageAnalyzer("E:/lama/experiment_results")
    analyzer.discover_experiments_and_algorithms()
    
    # 定义矩形区域的四个点坐标
    #? 建筑区域
    # rect_points = [(650, 650), (850, 650), (850, 850), (650, 850)]# 左上、右上、右下、左下
    #? 海洋区域
    # rect_points = [(450, 1450), (650, 1450), (650, 1650), (450, 1650)]# 左上、右上、右下、左下
    #? 陆地+海洋区域
    # rect_points = [(300, 1250), (500, 1250), (500, 1450), (300, 1450)]# 左上、右上、右下、左下
    #? 植被区域
    # rect_points = [(0, 1250), (200, 1250), (200, 1450), (0, 1450)]# 左上、右上、右下、左下
    #? heatmap分析区域
    rect_points = [(100, 1550), (300, 1550), (300, 1750), (100, 1750)]# 左上、右上、右下、左下

    # 可视化裁剪区域
    analyzer.visualize_cropped_region(rect_points)
    
    # 裁剪区域并保存图像（包括带标记的图像）
    cropped_images = analyzer.crop_region_from_images(
        rect_points, 
        # output_dir="E:/lama/building_cropped_images",
        # output_dir="E:/lama/ocean_cropped_images",
        # output_dir="E:/lama/landOcean_cropped_images",
        # output_dir="E:/lama/tree_cropped_images",
        output_dir="E:/lama/error_heatmap_cropped_images",
        save_marked_images=True  # 保存带裁剪区域标记的图像
    )
    
    # 创建对比可视化
    # analyzer.create_comparison_visualization(rect_points, "exp1", "scene1", frame_index=0)
    
    print("图像裁剪完成！输出三组图像：")
    print("1. 原始图像组：裁剪后的原始图像 + 带标记的原始图像")
    print("2. Masked图像组：裁剪后的masked图像 + 带标记的masked图像")
    print("3. 算法结果图像组：各算法的裁剪结果 + 各算法的带标记图像")
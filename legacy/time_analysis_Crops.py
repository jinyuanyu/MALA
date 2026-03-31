import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import argparse
from utils.paths import resolve_data_path

class CroppedRegionTimeSeriesAnalyzer:
    def __init__(self, cropped_root):
        """
        初始化分析器，用于分析裁剪区域的时序数据
        
        Args:
            cropped_root: 裁剪后的图像根目录路径
        """
        self.cropped_root = Path(cropped_root)
        self.algorithms = []
        self.experiment_dirs = []
        
        # 设置中文字体和更大的字体大小
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.rcParams['font.size'] = 18  # 增大基础字体大小
        plt.rcParams['axes.titlesize'] = 18  # 标题字体大小
        plt.rcParams['axes.labelsize'] = 16  # 坐标轴标签字体大小
        plt.rcParams['legend.fontsize'] = 18  # 图例字体大小
        plt.rcParams['xtick.labelsize'] = 18  # x轴刻度字体大小
        plt.rcParams['ytick.labelsize'] = 18  # y轴刻度字体大小

        # 时相索引映射到年积日
        self.day_mapping = {
            0: 19, 1: 29, 2: 54, 3: 59, 4: 64, 5: 274, 6: 329, 7: 364
        }
        
        # 年积日对应的中文名称
        self.day_names = {
            19: "1月19日", 29: "1月29日", 54: "2月23日", 59: "2月28日",
            64: "3月5日", 274: "10月1日", 329: "11月25日", 364: "12月30日"
        }
        
        # 新的配色方案
        self.ALGORITHM_COLORS = {
            'EMAE': "#df3131",  # 红色
            'MALA': "#A00505",  # 酒红色
            'Proposed': '#1f77b4', # 蓝色
            'Spline': '#ff7f0e',   # 橙色
            'DINEOF': '#2ca02c',   # 绿色
            'Lama': '#9467bd',     # 紫色
            'Nearest_Neighbor': '#8c564b',  # 棕色
            'NN': '#8c564b',  # 最近邻的别名
            'LaMA': '#9467bd',  # Lama的别名
            'MaskAE': '#1f77b4', # Proposed的别名
        }
        
        # 缺失比例对应的背景色（从黄到红）
        self.missing_colors = {
            'low': '#FFFACD',      # 0-25%: 浅黄色
            'medium': '#FFD700',   # 25-50%: 金黄色
            'high': '#FFA500',     # 50-75%: 橙色
            'very_high': '#FF4500' # 75-100%: 红橙色
        }
        
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
        
        # 定义三组算法的组合
        self.algorithm_groups = {
            'group1': ['DINEOF', 'Proposed', 'Lama', 'Spline', 'Nearest_Neighbor', 'EMAE', 'MALA'],
            'group2': ['DINEOF', 'Spline', 'EMAE', 'Proposed'],
            'group3': ['Lama', 'MALA', 'Nearest_Neighbor', 'EMAE']
        }
        
        # 线型和标记样式
        self.line_styles = {
            'DINEOF': '-', 'Proposed': '--', 'Lama': '-.', 'Spline': ':',
            'Nearest_Neighbor': '-', 'EMAE': '--', 'MALA': '-.'
        }
        
        self.markers = {
            'DINEOF': '^', 'Proposed': 'v', 'Lama': '<', 'Spline': '>',
            'Nearest_Neighbor': 'D', 'EMAE': 'p', 'MALA': '*'
        }
        
        # 组名映射
        self.group_names = {
            'group1': '所有算法对比',
            'group2': 'DINEOF_Spline_EMAE_MaskAE',
            'group3': 'LaMA_MALA_NN_EMAE'
        }
        
    def discover_experiments_and_algorithms(self):
        """
        自动发现实验目录和算法（针对裁剪后的目录结构）
        """
        self.experiment_dirs = []
        self.algorithms = set()
        
        # 遍历裁剪根目录
        for exp_dir in self.cropped_root.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith('exp'):
                self.experiment_dirs.append(exp_dir)
                
                # 遍历每个实验目录下的场景
                for scene_dir in exp_dir.iterdir():
                    if scene_dir.is_dir():
                        # 发现算法目录
                        for item in scene_dir.iterdir():
                            if item.is_dir() and item.name in self.algorithm_name_map:
                                self.algorithms.add(item.name)
        
        self.algorithms = sorted(list(self.algorithms))
        print(f"发现实验目录: {[d.name for d in self.experiment_dirs]}")
        print(f"发现算法: {[self.algorithm_name_map.get(alg, alg) for alg in self.algorithms]}")
    
    def load_cropped_sequence(self, image_dir):
        """
        加载裁剪后的图像序列
        
        Args:
            image_dir: 图像目录路径
            
        Returns:
            tuple: (图像数组列表, 文件名列表)
        """
        images = []
        filenames = []
        
        if not image_dir.exists():
            return None, None
            
        # 获取所有PNG文件并按文件名排序
        frame_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        if not frame_files:
            return None, None
        
        # 先读取第一张图像获取尺寸
        first_img_path = os.path.join(image_dir, frame_files[0])
        first_img = cv2.imread(first_img_path, cv2.IMREAD_GRAYSCALE)
        if first_img is None:
            return None, None
        
        target_shape = first_img.shape
        images.append(first_img)
        filenames.append(frame_files[0])
        
        # 加载剩余图像并确保尺寸一致
        for frame_file in frame_files[1:]:
            img_path = os.path.join(image_dir, frame_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 如果图像尺寸不一致，调整到目标尺寸
                if img.shape != target_shape:
                    img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
                images.append(img)
                filenames.append(frame_file)
        
        if not images:
            return None, None
            
        try:
            return np.array(images), filenames
        except ValueError as e:
            print(f"图像尺寸不一致，无法创建数组: {e}")
            # 如果仍然失败，返回列表形式
            return images, filenames
    
    def extract_region_timeseries(self, image_sequence):
        """
        提取整个区域的时序数据（平均值）
        
        Args:
            image_sequence: 图像序列 (T, H, W) 或图像列表
            
        Returns:
            array: 区域平均值的时序数据
        """
        if isinstance(image_sequence, list):
            # 如果是列表，逐个计算平均值
            return np.array([np.mean(img) for img in image_sequence])
        elif len(image_sequence.shape) == 3:
            # 如果是numpy数组，计算每帧的平均值
            return np.mean(image_sequence, axis=(1, 2))
        else:
            return np.array([np.mean(image_sequence)])
    
    def get_missing_frames_info(self, exp_name, scene_name):
        """
        从masked图像中获取缺失帧的信息和缺失比例
        
        Args:
            exp_name: 实验名称
            scene_name: 场景名称
            
        Returns:
            tuple: (缺失帧的索引列表, 每帧的缺失比例列表)
        """
        # 首先尝试masked目录，如果不存在则尝试mask目录
        masked_dir = self.cropped_root / exp_name / scene_name / 'masked'
        if not masked_dir.exists():
            raise ValueError(f"masked目录不存在: {masked_dir}")
        
        if not masked_dir.exists():
            print(f"masked/mask目录不存在: {masked_dir}")
            return [], []
        
        # 加载masked序列
        masked_seq, _ = self.load_cropped_sequence(masked_dir)
        
        if masked_seq is None:
            print(f"无法加载masked序列: {masked_dir}")
            return [], []
        
        missing_frames = []
        missing_ratios = []
        
        # 检查每帧masked的缺失比例
        for frame_idx, masked_frame in enumerate(masked_seq):
            # 计算白色像素（缺失区域）的比例
            if isinstance(masked_frame, np.ndarray):
                missing_pixels = np.sum(masked_frame ==0)
                total_pixels = masked_frame.size
            else:
                # 如果是列表中的图像
                missing_pixels = np.sum(np.array(masked_frame) ==0)
                total_pixels = np.array(masked_frame).size
            
            missing_ratio = missing_pixels / total_pixels if total_pixels > 0 else 0
            
            missing_ratios.append(missing_ratio)
            
            if missing_ratio > 0:  # 有缺失区域
                missing_frames.append(frame_idx)
        
        return missing_frames, missing_ratios
    
    def get_background_color(self, missing_ratio):
        """
        根据缺失比例获取背景色
        
        Args:
            missing_ratio: 缺失比例 (0-1)
            
        Returns:
            str: 背景颜色
        """
        if missing_ratio <= 0.25:
            return self.missing_colors['low']
        elif missing_ratio <= 0.5:
            return self.missing_colors['medium']
        elif missing_ratio <= 0.75:
            return self.missing_colors['high']
        else:
            return self.missing_colors['very_high']
    
    def calculate_quartiles(self, timeseries_data):
        """
        计算时序数据的四分位数和标准差
        
        Args:
            timeseries_data: 时序数据字典
            
        Returns:
            dict: 包含四分位数信息的字典
        """
        quartiles = {}
        
        for alg_name, data in timeseries_data.items():
            values = data['timeseries']
            mean_val = np.mean(values)
            std_val = np.std(values)
            q1 = np.percentile(values, 25)
            q2 = np.percentile(values, 50)  # 中位数
            q3 = np.percentile(values, 75)
            
            quartiles[alg_name] = {
                'q1': q1,
                'q2': q2,
                'q3': q3,
                'mean': mean_val,
                'std': std_val,
                'timeseries': values
            }
        
        return quartiles
    
    def analyze_cropped_region(self, exp_name, scene_name, region_type='original'):
        """
        分析裁剪区域的时序数据
        
        Args:
            exp_name: 实验名称
            scene_name: 场景名称
            region_type: 区域类型 ('original', 'masked', 或算法名称)
            
        Returns:
            dict: 分析结果
        """
        region_dir = self.cropped_root / exp_name / scene_name / region_type
        
        if not region_dir.exists():
            print(f"区域目录不存在: {region_dir}")
            return None
        
        # 加载裁剪后的图像序列
        image_seq, filenames = self.load_cropped_sequence(region_dir)
        
        if image_seq is None:
            print(f"无法加载图像序列: {region_dir}")
            return None
        
        print(f"加载完成 - {region_type}: {image_seq.shape}")
        
        # 提取区域时序数据（平均值）
        timeseries = self.extract_region_timeseries(image_seq)
        
        return {
            'region_type': region_type,
            'timeseries': timeseries,
            'frames': len(timeseries),
            'filenames': filenames
        }
    
    def analyze_all_regions(self, exp_name, scene_name):
        """
        分析所有区域的时序数据
        
        Args:
            exp_name: 实验名称
            scene_name: 场景名称
            
        Returns:
            dict: 所有区域的分析结果
        """
        results = {}
        
        # 分析原始区域
        original_result = self.analyze_cropped_region(exp_name, scene_name, 'original')
        if original_result:
            results['original'] = original_result
        
        # 分析各算法区域
        for algorithm in self.algorithms:
            alg_result = self.analyze_cropped_region(exp_name, scene_name, algorithm)
            if alg_result:
                results[algorithm] = alg_result
        
        return results
    
    def plot_single_group_comparison(self, analysis_results, group_name, group_algorithms, 
                               exp_name, scene_name, missing_frames, missing_ratios, 
                               save_path=None):
        """
        绘制单组算法的时序对比图并保存为单独图片
        """
        if not analysis_results or 'original' not in analysis_results:
            return
        
        # 设置图形样式
        plt.style.use('default')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        frames = range(analysis_results['original']['frames'])
        
        # 计算四分位数
        quartiles = self.calculate_quartiles(analysis_results)
        
        # 获取Y轴范围
        all_values = [analysis_results['original']['timeseries']]
        for alg in group_algorithms:
            if alg in analysis_results:
                all_values.append(analysis_results[alg]['timeseries'])
        
        flat_values = np.concatenate(all_values)
        y_min, y_max = np.min(flat_values), np.max(flat_values)
        y_range = y_max - y_min
        y_min -= y_range * 0.1
        y_max += y_range * 0.1
        
        # === 修改部分：使用等间距的横坐标 ===
        # 创建等间距的x坐标（0, 1, 2, 3, ...）
        x_positions = list(range(len(frames)))
        
        # 绘制背景色（根据缺失比例）- 使用等间距坐标
        # 扩展x坐标用于背景色绘制
        extended_x = np.concatenate(([x_positions[0] - 0.5], 
                                [x + 0.5 for x in x_positions]))
        
        # 逐帧涂色
        for i, frame_idx in enumerate(frames):
            if i >= len(missing_ratios):
                continue
            bg_color = self.get_background_color(missing_ratios[i])
            
            # 区间左边界
            if i == 0:
                x_left = -0.5
            else:
                x_left = extended_x[i]
            
            # 区间右边界
            if i == len(frames) - 1:
                x_right = x_positions[-1] + 0.5
            else:
                x_right = extended_x[i + 1]
            
            ax.axvspan(x_left, x_right,
                    facecolor=bg_color, alpha=0.4, zorder=0)
        
        # 绘制原始区域曲线（粗黑线）- 使用等间距坐标
        ax.plot(x_positions, analysis_results['original']['timeseries'], 
                'k-', linewidth=4, label='真值', marker='o', markersize=8, zorder=10)
        
        # 为真值添加标准差误差线
        if 'original' in quartiles:
            orig_q = quartiles['original']
            orig_values = orig_q['timeseries']
            orig_std = orig_q['std']
            
            # 创建与数据点数量匹配的误差线数组
            n_points = len(orig_values)
            yerr_lower = np.full(n_points, orig_std * 0.5)
            yerr_upper = np.full(n_points, orig_std * 0.5)
            
            ax.errorbar(x_positions, orig_values, 
                    yerr=[yerr_lower, yerr_upper], 
                    fmt='none', ecolor='black', capsize=4, capthick=2, 
                    alpha=0.6, zorder=9)
        
        # 绘制该组算法结果 - 使用等间距坐标
        for alg_name in group_algorithms:
            if alg_name in analysis_results:
                color = self.ALGORITHM_COLORS.get(alg_name, '#000000')
                linestyle = self.line_styles.get(alg_name, '-')
                marker = self.markers.get(alg_name, 'o')
                
                # 使用中文算法名称
                chinese_name = self.algorithm_name_map.get(alg_name, alg_name)
                
                ax.plot(x_positions, analysis_results[alg_name]['timeseries'], 
                        color=color, linewidth=3, linestyle=linestyle, 
                        label=chinese_name, marker=marker, markersize=6, alpha=0.9, zorder=5)
                
                # 为算法添加标准差误差线
                if alg_name in quartiles:
                    alg_q = quartiles[alg_name]
                    alg_values = alg_q['timeseries']
                    alg_std = alg_q['std']
                    
                    n_points = len(alg_values)
                    yerr_lower = np.full(n_points, alg_std * 0.3)
                    yerr_upper = np.full(n_points, alg_std * 0.3)
                    
                    ax.errorbar(x_positions, alg_values, 
                            yerr=[yerr_lower, yerr_upper], 
                            fmt='none', ecolor=color, capsize=3, capthick=1.5, 
                            alpha=0.5, zorder=4)
        
        # === 修改部分：设置等间距的x轴标签 ===
        # 设置x轴刻度为等间距（0, 1, 2, ...）
        ax.set_xticks(x_positions)
        
        # 创建双标签：上方显示时相编号，下方显示年积日
        day_labels = []
        for i in range(len(frames)):
            if i in self.day_mapping:
                day_of_year = self.day_mapping[i]
                day_labels.append(f"{day_of_year}")
            else:
                day_labels.append(f"{i}")
        
        ax.set_xticklabels(day_labels, fontsize=16,fontweight ='bold')
        
        # 添加第二行x轴标签显示年积日
        # 创建一个次要的x轴来显示年积日
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())  # 与主x轴范围一致
        
        # 设置次要x轴的刻度位置（与主x轴相同）
        ax2.set_xticks(x_positions)
        
        # 创建年积日标签（只显示有对应关系的）
        doy_labels = []
        for i in range(len(frames)):
            if i in self.day_mapping:
                day_of_year = self.day_mapping[i]
                doy_labels.append(f"{self.day_names.get(day_of_year, str(day_of_year))}")
            else:
                doy_labels.append("")

        ax2.set_xticklabels(doy_labels, fontsize=16,fontweight ='bold', rotation=45, ha='left')
        ax2.tick_params(axis='x', which='major', pad=25)  # 增加标签与轴的距离
        
        # 设置坐标轴标签
        # ax.set_xlabel('时相编号', fontsize=16, fontweight='bold', labelpad=15)
        # ax2.set_xlabel('年积日 (日期)', fontsize=14, fontweight='bold', labelpad=25)
        ax.set_ylabel('区域平均像素值', fontsize=18, fontweight='bold')
        
        # ax.set_title(f'{group_name} - {exp_name}/{scene_name}', fontsize=18, fontweight='bold', pad=30)
        
        # 设置图例 - 主图例（算法）
        handles, labels = ax.get_legend_handles_labels()
        main_legend = ax.legend(handles, labels, loc='upper left', frameon=True, 
                            fancybox=True, shadow=True, fontsize=16)
        
        # 设置网格
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1)
        ax.set_xlim(-0.5, len(frames) - 0.5)  # 设置x轴范围以包含所有点
        ax.set_ylim(y_min, y_max)
        
        # 美化图形
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # 添加缺失比例图例
        legend_elements = [
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.missing_colors['low'], 
                            alpha=0.4, label='缺失比例 0-25%'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.missing_colors['medium'], 
                            alpha=0.4, label='缺失比例 25-50%'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.missing_colors['high'], 
                            alpha=0.4, label='缺失比例 50-75%'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.missing_colors['very_high'], 
                            alpha=0.4, label='缺失比例 75-100%'),
        ]
        
        # 添加第二个图例（缺失比例）
        legend2 = ax.legend(handles=legend_elements, loc='lower right', 
                        frameon=True, fancybox=True, shadow=True, fontsize=16)
        ax.add_artist(main_legend)
        ax.add_artist(legend2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"图像保存至: {save_path}")
        
        plt.show()
        plt.close(fig)
    
    def plot_three_groups_separately(self, analysis_results, exp_name, scene_name, base_save_path=None):
        """
        分别绘制三组算法的时序对比图并保存为三张单独的图片
        """
        if not analysis_results:
            return
        
        # 获取缺失帧信息和缺失比例
        missing_frames, missing_ratios = self.get_missing_frames_info(exp_name, scene_name)
        
        # 绘制第一组：所有算法
        if base_save_path:
            save_path1 = f"{base_save_path}_group1_all_algorithms.png"
        else:
            save_path1 = None
            
        self.plot_single_group_comparison(
            analysis_results, '第一组：所有算法对比', 
            self.algorithm_groups['group1'],
            exp_name, scene_name, missing_frames, missing_ratios, save_path1
        )
        
        # 绘制第二组：DINEOF、Spline、EMAE、MaskAE
        if base_save_path:
            save_path2 = f"{base_save_path}_group2_DINEOF_Spline_EMAE_MaskAE.png"
        else:
            save_path2 = None
            
        self.plot_single_group_comparison(
            analysis_results, '第二组：DINEOF、Spline、EMAE、MaskAE', 
            self.algorithm_groups['group2'],
            exp_name, scene_name, missing_frames, missing_ratios, save_path2
        )
        
        # 绘制第三组：LaMA、MALA、NN、EMAE
        if base_save_path:
            save_path3 = f"{base_save_path}_group3_LaMA_MALA_NN_EMAE.png"
        else:
            save_path3 = None
            
        self.plot_single_group_comparison(
            analysis_results, '第三组：LaMA、MALA、NN、EMAE', 
            self.algorithm_groups['group3'],
            exp_name, scene_name, missing_frames, missing_ratios, save_path3
        )
    
    def calculate_region_metrics(self, analysis_results):
        """
        计算区域评估指标
        """
        if not analysis_results or 'original' not in analysis_results:
            return None
        
        ground_truth = analysis_results['original']['timeseries']
        
        metrics = {}
        for alg_name, result in analysis_results.items():
            if alg_name != 'original':  # 跳过原始区域
                pred_values = result['timeseries']
                
                # 计算MSE和PSNR
                mse = np.mean((ground_truth - pred_values) ** 2)
                psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
                
                # 计算MAE
                mae = np.mean(np.abs(ground_truth - pred_values))
                
                # 计算相关系数
                correlation = np.corrcoef(ground_truth, pred_values)[0, 1]
                
                metrics[alg_name] = {
                    'MSE': mse,
                    'PSNR': psnr,
                    'MAE': mae,
                    'Correlation': correlation
                }
        
        return metrics
    
    def print_region_metrics_table(self, metrics):
        """
        以表格形式打印区域评估指标
        """
        if not metrics:
            return
            
        print("\n" + "="*100)
        print(f"{'算法':<20} {'PSNR':<12} {'MSE':<12} {'MAE':<12} {'相关系数':<12}")
        print("="*100)
        
        for alg_name, metric_dict in metrics.items():
            # 使用中文算法名称
            chinese_name = self.algorithm_name_map.get(alg_name, alg_name)
            print(f"{chinese_name:<20} "
                  f"{metric_dict['PSNR']:<12.2f} "
                  f"{metric_dict['MSE']:<12.2f} "
                  f"{metric_dict['MAE']:<12.2f} "
                  f"{metric_dict['Correlation']:<12.4f}")
        print("="*100)

# 使用示例
def main(cropped_root="E:/lama/landOcean_cropped_images", exp_name="exp2_missing_ratios", scene_name="10percent", base_save_path="landOcean_region_timeseries_comparison"):
    analyzer = CroppedRegionTimeSeriesAnalyzer(resolve_data_path(cropped_root))
    analyzer.discover_experiments_and_algorithms()
    results = analyzer.analyze_all_regions(
        exp_name=exp_name,
        scene_name=scene_name
    )
    
    if results:
        analyzer.plot_three_groups_separately(
            results, 
            exp_name,
            scene_name,
            base_save_path=base_save_path
        )
        metrics = analyzer.calculate_region_metrics(results)
        if metrics:
            analyzer.print_region_metrics_table(metrics)

def parse_args_and_run():
    parser = argparse.ArgumentParser(description="裁剪区域时序分析")
    parser.add_argument("--cropped-root", default="E:/lama/landOcean_cropped_images")
    parser.add_argument("--exp-name", default="exp2_missing_ratios")
    parser.add_argument("--scene-name", default="10percent")
    parser.add_argument("--base-save-path", default="landOcean_region_timeseries_comparison")
    args = parser.parse_args()
    main(args.cropped_root, args.exp_name, args.scene_name, args.base_save_path)


if __name__ == "__main__":
    parse_args_and_run()

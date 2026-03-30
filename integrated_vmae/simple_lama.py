import os
from simple_lama_inpainting import SimpleLama
from PIL import Image
import glob

def process_specific_experiment(experiment_path):
    """
    处理特定的实验目录
    
    Args:
        experiment_path: 特定实验的路径
    """
    # 初始化SimpleLama
    simple_lama = SimpleLama()
    
    print(f"处理特定实验目录: {experiment_path}")
    
    # 查找所有图像目录
    image_dirs = []
    for root, dirs, files in os.walk(experiment_path):
        if "images" in dirs:
            image_dirs.append(os.path.join(root, "images"))
    
    # 处理每个图像目录
    for img_dir in image_dirs:
        print(f"  处理图像目录: {img_dir}")
        
        # 创建Lama子目录（如果不存在）
        lama_dir = os.path.join(img_dir, "Lama")
        os.makedirs(lama_dir, exist_ok=True)
        
        # 查找所有masked图像文件
        masked_files = glob.glob(os.path.join(img_dir, "masked_frame*.png"))
        
        # 处理每个masked图像
        for masked_path in masked_files:
            # 提取帧号
            frame_num = os.path.basename(masked_path).split("_frame")[1].split(".png")[0]
            
            # 构建对应的mask文件路径
            mask_path = os.path.join(img_dir, f"mask_frame{frame_num}.png")
            
            # 检查mask文件是否存在
            if not os.path.exists(mask_path):
                print(f"    警告: 找不到对应的mask文件 {mask_path}")
                continue
            
            # 构建输出文件路径
            output_path = os.path.join(lama_dir, f"lama_inpainted_frame{frame_num}.png")
            
            # 如果输出文件已存在，跳过
            if os.path.exists(output_path):
                print(f"    跳过已处理的帧 {frame_num}")
                continue
            
            print(f"    处理帧 {frame_num}: {masked_path} + {mask_path} -> {output_path}")
            
            try:
                # 加载图像和掩码
                image = Image.open(masked_path)
                mask = Image.open(mask_path).convert('L')
                
                # 应用SimpleLama修复
                result = simple_lama(image, mask)
                
                # 保存修复结果
                result.save(output_path)
                print(f"    成功保存修复结果: {output_path}")
                
            except Exception as e:
                print(f"    处理帧 {frame_num} 时出错: {e}")
                continue

def find_and_process_all_experiments(root_dir="experiment_results"):
    """
    自动查找并处理所有实验目录
    
    Args:
        root_dir: 实验结果的根目录
    """
    print(f"开始自动遍历处理目录: {root_dir}")
    
    # 检查根目录是否存在
    if not os.path.exists(root_dir):
        print(f"错误: 根目录 {root_dir} 不存在")
        return
    
    # 查找所有实验目录
    experiment_dirs = []
    
    # 第一级目录：实验类型（如exp1_missing_types, exp2_missing_ratios）
    for exp_type in os.listdir(root_dir):
        exp_type_path = os.path.join(root_dir, exp_type)
        if os.path.isdir(exp_type_path):
            # 第二级目录：具体实验条件（如thin_cloud, 10percent等）
            for exp_condition in os.listdir(exp_type_path):
                exp_condition_path = os.path.join(exp_type_path, exp_condition)
                if os.path.isdir(exp_condition_path):
                    experiment_dirs.append(exp_condition_path)
    
    print(f"找到 {len(experiment_dirs)} 个实验目录")
    
    # 处理每个实验目录
    for i, exp_dir in enumerate(experiment_dirs, 1):
        print(f"\n[{i}/{len(experiment_dirs)}] 处理实验: {exp_dir}")
        process_specific_experiment(exp_dir)
    
    print("\n所有实验处理完成！")

def process_by_experiment_type(root_dir="experiment_results", target_types=None):
    """
    按实验类型处理特定的实验
    
    Args:
        root_dir: 根目录
        target_types: 要处理的实验类型列表，如 ['missing_types', 'missing_ratios']
    """
    if target_types is None:
        target_types = ['missing_types', 'missing_ratios']
    
    print(f"处理特定实验类型: {target_types}")
    
    for exp_type in target_types:
        exp_type_path = os.path.join(root_dir, f"exp1_{exp_type}" if exp_type == "missing_types" else f"exp2_{exp_type}")
        
        if not os.path.exists(exp_type_path):
            print(f"警告: 实验类型目录 {exp_type_path} 不存在")
            continue
        
        print(f"\n处理实验类型: {exp_type}")
        
        # 处理该类型下的所有实验条件
        for exp_condition in os.listdir(exp_type_path):
            exp_condition_path = os.path.join(exp_type_path, exp_condition)
            if os.path.isdir(exp_condition_path):
                print(f"  处理实验条件: {exp_condition}")
                process_specific_experiment(exp_condition_path)

if __name__ == "__main__":
    # 方法1: 自动遍历处理所有实验
    find_and_process_all_experiments("experiment_results")
    
    # 方法2: 只处理特定类型的实验
    # process_by_experiment_type("experiment_results", ['missing_types'])
    # process_by_experiment_type("experiment_results", ['missing_ratios'])
    
    # 方法3: 只处理单个特定实验
    # process_specific_experiment("experiment_results/exp1_missing_types/thin_cloud")
    # process_specific_experiment("experiment_results/exp2_missing_ratios/10percent")
import os
import shutil
from pathlib import Path

def clean_yolo_subset(input_dir, output_dir, target_class=1):
    """
    清洗单个数据子集（train或val）
    """
    # 创建输出目录
    images_out = Path(output_dir) / 'images'
    labels_out = Path(output_dir) / 'labels'
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    # 获取输入路径
    labels_dir = Path(input_dir) / 'labels'
    images_dir = Path(input_dir) / 'images'
    
    # 统计计数器
    total_files = 0
    kept_files = 0
    total_labels = 0
    kept_labels = 0
    
    # 遍历所有标签文件
    for label_file in labels_dir.glob('*.txt'):
        total_files += 1
        keep_file = False
        
        # 读取标签文件
        filtered_lines = []
        with open(label_file, 'r') as f:
            lines = f.readlines()
            total_labels += len(lines)
            
            # 只保留目标类别的标注行，并将类别1改为0
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # 确保标注格式正确
                    class_id = int(parts[0])
                    if class_id == target_class:
                        # 将类别1改为0
                        modified_line = f"0 {' '.join(parts[1:])}\n"
                        keep_file = True
                        filtered_lines.append(modified_line)
                        kept_labels += 1
        
        # 如果文件包含目标类别，则保存文件
        if keep_file:
            kept_files += 1
            
            # 保存筛选后的标签文件
            new_label_path = labels_out / label_file.name
            with open(new_label_path, 'w') as f:
                f.writelines(filtered_lines)
            
            # 复制对应的图片文件
            img_extensions = ['.jpg', '.jpeg', '.png']
            img_stem = label_file.stem
            for ext in img_extensions:
                img_path = images_dir / f"{img_stem}{ext}"
                if img_path.exists():
                    shutil.copy2(img_path, images_out / img_path.name)
                    break
    
    return {
        'total_files': total_files,
        'kept_files': kept_files,
        'total_labels': total_labels,
        'kept_labels': kept_labels
    }

def clean_yolo_dataset(data_dir, output_dir, target_class=1):
    """
    清洗完整的YOLO数据集，包括train和val子集
    """
    subsets = ['train', 'val']
    all_stats = {}
    
    for subset in subsets:
        print(f"\n处理{subset}数据集...")
        input_subset = Path(data_dir) / subset
        output_subset = Path(output_dir) / subset
        
        if input_subset.exists():
            stats = clean_yolo_subset(input_subset, output_subset, target_class)
            all_stats[subset] = stats
            
            print(f"{subset}数据集统计:")
            print(f"- 总文件数: {stats['total_files']}")
            print(f"- 保留文件数: {stats['kept_files']}")
            print(f"- 删除文件数: {stats['total_files'] - stats['kept_files']}")
            print(f"- 原始标签总数: {stats['total_labels']}")
            print(f"- 保留标签数(类别{target_class}): {stats['kept_labels']}")
            print(f"- 删除标签数: {stats['total_labels'] - stats['kept_labels']}")
    
    # 打印总体统计
    if all_stats:
        print("\n整体统计:")
        total_files = sum(s['total_files'] for s in all_stats.values())
        total_kept = sum(s['kept_files'] for s in all_stats.values())
        total_labels = sum(s['total_labels'] for s in all_stats.values())
        total_kept_labels = sum(s['kept_labels'] for s in all_stats.values())
        
        print(f"总文件数: {total_files}")
        print(f"保留文件数: {total_kept}")
        print(f"删除文件数: {total_files - total_kept}")
        print(f"原始标签总数: {total_labels}")
        print(f"保留标签数: {total_kept_labels}")
        print(f"删除标签数: {total_labels - total_kept_labels}")
        
    print(f"\n清洗后的数据保存在: {output_dir}")

if __name__ == "__main__":
    # 设置路径
    data_dir = "datasets/yolo_data"  # 原始数据目录
    output_dir = "datasets/yolo_data_cleaned"  # 清洗后的数据保存目录
    target_class = 1  # 要保留的目标类别
    
    # 执行清洗
    clean_yolo_dataset(data_dir, output_dir, target_class) 
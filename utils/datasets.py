import os
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np

class LoadDataset(Dataset):
    def __init__(self, data_dir, img_size=640, is_train=True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_train = is_train
        
        # 获取所有图片文件路径
        self.img_files = []
        self.label_files = []
        
        # 假设图片在 images 子目录，标签在 labels 子目录
        img_dir = os.path.join(data_dir, 'images')
        label_dir = os.path.join(data_dir, 'labels')
        
        for img_file in os.listdir(img_dir):
            if img_file.endswith(('.jpg', '.png')):
                img_path = os.path.join(img_dir, img_file)
                label_path = os.path.join(label_dir, img_file.rsplit('.', 1)[0] + '.txt')
                
                if os.path.exists(label_path):
                    self.img_files.append(img_path)
                    self.label_files.append(label_path)
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 读取图片
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1) / 255.0  # HWC to CHW, 归一化
        
        # 读取标签
        label_path = self.label_files[idx]
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)  # YOLO格式: class x y w h
        else:
            labels = np.zeros((0, 5))
            
        return torch.FloatTensor(img), torch.FloatTensor(labels) 

# 使用您的实际数据集
train_data_dir = 'D:/Code/python/YOLOv11/datasets/yolo_data_cleaned/train'
val_data_dir = 'D:/Code/python/YOLOv11/datasets/yolo_data_cleaned/val'
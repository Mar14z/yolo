import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ultralytics.models import YOLO
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter


from utils.datasets import LoadDataset  # 数据加载器
from utils.loss import YOLOLoss  # 损失函数
from utils.metrics import calculate_map  # mAP计算函数

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.writer = SummaryWriter(cfg['log_dir'])
        
        # 初始化模型
        self.model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
        self.model.train(
            data='ultralytics/cfg/datasets/demo1.yaml',  # 使用 coco8 数据集配置
            epochs=100,
            imgsz=640,
            batch=16,
    
        )
        
        # 数据加载
        self.train_loader = self.get_dataloader(is_train=True)
        self.val_loader = self.get_dataloader(is_train=False)
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.1
        )
        self.criterion = YOLOLoss()
        
        # 训练状态记录
        self.best_map = 0
        self.train_losses = []
        self.val_maps = []
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.cfg['log_file']),
                logging.StreamHandler()
            ]
        )
        
    def get_dataloader(self, is_train=True):
        dataset = LoadDataset(
            data_dir=self.cfg['train_data_dir'] if is_train else self.cfg['val_data_dir'],
            img_size=self.cfg['img_size'],
            is_train=is_train
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=is_train,
            num_workers=self.cfg['num_workers'],
            pin_memory=True
        )
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.cfg["epochs"]}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            predictions = self.model(images)
            loss = self.criterion(predictions, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新进度条
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 记录到TensorBoard
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), step)
            
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch):
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                predictions = self.model(images)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算mAP
        map_score = calculate_map(all_predictions, all_targets)
        self.writer.add_scalar('val/mAP', map_score, epoch)
        
        return map_score
    
    def save_checkpoint(self, epoch, map_score):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'map': map_score,
        }
        
        # 保存最新检查点
        torch.save(checkpoint, f'{self.cfg["checkpoint_dir"]}/last.pt')
        
        # 保存最佳模型
        if map_score > self.best_map:
            self.best_map = map_score
            torch.save(checkpoint, f'{self.cfg["checkpoint_dir"]}/best.pt')
            logging.info(f'New best mAP: {map_score:.4f}')
    
    def plot_training_progress(self):
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        
        # mAP曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.val_maps, label='Validation mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('Validation mAP Over Time')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.cfg["log_dir"]}/training_progress.png')
        plt.close()
    
    def train(self):
        logging.info(f"Starting training on device: {self.device}")
        
        for epoch in range(self.cfg['epochs']):
            # 训练一个epoch
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # 验证
            map_score = self.validate(epoch)
            self.val_maps.append(map_score)
            
            # 更新学习率
            self.scheduler.step(train_loss)
            
            # 保存检查点
            self.save_checkpoint(epoch, map_score)
            
            # 更新进度图
            self.plot_training_progress()
            
            # 记录日志
            logging.info(
                f"Epoch {epoch+1}/{self.cfg['epochs']}, "
                f"Loss: {train_loss:.4f}, mAP: {map_score:.4f}"
            )
        
        logging.info("Training completed!")
        self.writer.close()

if __name__ == "__main__":
    config = {
        'train_data_dir': 'datasets/yolo_data_cleaned/train',
        'val_data_dir': 'datasets/yolo_data_cleaned/val',
        'log_dir': 'runs/train',
        'log_file': 'training.log',
        'checkpoint_dir': 'checkpoints',
        'num_classes': 1,  # 因为只保留了类别1
        'img_size': 640,
        'batch_size': 16,
        'num_workers': 4,
        'lr': 0.001,
        'epochs': 100,
    }
    
    # 创建必要的目录
    for dir_path in [config['log_dir'], config['checkpoint_dir']]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 开始训练
    trainer = Trainer(config)
    trainer.train()
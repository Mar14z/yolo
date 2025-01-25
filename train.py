# import torch
# from ultralytics.models import YOLO

# # 初始化模型
# model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')

# # 训练模型
# if __name__ == "__main__":
#     model.train(
#         data='ultralytics/cfg/datasets/demo1.yaml',  # 数据集配置文件
#         epochs=100,                                  # 训练轮次
#         imgsz=640,                                  # 图像大小
#         batch=16,                                   # 批次大小
#         device=0 if torch.cuda.is_available() else 'cpu',  # 使用的设备
#         plots=True,                                # 保存训练图表
#         save=True,                                 # 保存模型
#         save_period=10,                            # 每10轮保存一次
#         verbose=True                               # 显示详细信息
#     )

# # 运行 TensorBoard:
# # tensorboard --logdir runs/train/exp

import torch
from ultralytics.models import YOLO

# 初始化模型
model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')

# 训练模型
if __name__ == "__main__":
    model.train(
        data='ultralytics/cfg/datasets/demo1.yaml',  # 数据集配置文件
        epochs=300,                                  # 训练轮次
        imgsz=640,                                  # 图像大小
        batch=16,                                   # 批次大小
        device=0 if torch.cuda.is_available() else 'cpu',  # 使用的设备

        
        # 优化器参数
        lr0=0.01,                                   # 初始学习率
        lrf=0.01,                                   # 最终学习率
        momentum=0.937,                             # SGD动量
        weight_decay=0.0005,                        # 权重衰减
        warmup_epochs=3,                            # 预热轮次
        warmup_momentum=0.8,                        # 预热动量
        warmup_bias_lr=0.1,                         # 预热偏置学习率
        
        # 数据增强
        hsv_h=0.015,                               # HSV-H增强
        hsv_s=0.7,                                 # HSV-S增强
        hsv_v=0.4,                                 # HSV-V增强
        degrees=0.0,                               # 旋转角度
        translate=0.1,                             # 平移
        scale=0.5,                                 # 缩放
        shear=0.0,                                 # 剪切
        perspective=0.0,                           # 透视
        flipud=0.0,                               # 上下翻转
        fliplr=0.5,                               # 左右翻转
        mosaic=1.0,                               # 马赛克增强
        mixup=0.0,                                # mixup增强
        copy_paste=0.0,                           # 复制粘贴增强
        
        # 训练策略
        label_smoothing=0.0,                       # 标签平滑
        multi_scale=True,                          # 多尺度训练
        single_cls=False,                          # 单类别模式
        rect=False,                                # 矩形训练
        cos_lr=True,                              # 余弦学习率调度
        
        # 保存和日志
        plots=True,                                # 保存训练图表
        save=True,                                 # 保存模型
        save_period=10,                            # 每50轮保存一次
        patience=100,                             # 早停轮次
        verbose=True,                             # 显示详细信息
        
        # 高级参数
        overlap_mask=True,                        # 重叠mask
        mask_ratio=4,                             # mask比例
        dropout=0.0,                              # dropout比例
        val=True,                                # 是否验证
        workers=8                                 # 数据加载线程数
    )

# 运行 TensorBoard:
# tensorboard --logdir /home/wfs/zmx/YOLOv11/runs/detect


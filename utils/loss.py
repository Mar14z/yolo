import torch
import torch.nn as nn
import numpy as np

class YOLOLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.obj_scale = 1
        self.noobj_scale = 0.5
        
    def forward(self, predictions, targets):
        """
        predictions: 模型输出列表 [batch_size, anchors, H, W]
        targets: 真实标签列表 [batch_size, max_objects, 5] (class, x, y, w, h)
        """
        # 确保predictions和targets都是列表形式
        if not isinstance(predictions, list):
            predictions = [predictions]
        if not isinstance(targets, list):
            targets = [targets]
            
        # 将每个预测结果和目标移到正确的设备
        predictions = [p.to(self.device) if isinstance(p, torch.Tensor) else p for p in predictions]
        targets = [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in targets]
        
        total_loss = 0
        batch_size = predictions[0].shape[0]
        
        # 对每个尺度的预测分别计算损失
        for pred, tgt in zip(predictions, targets):
            # pred shape: [batch_size, anchors, H, W]
            B, A, H, W = pred.shape
            
            # 创建目标掩码
            obj_mask = torch.zeros((B, A, H, W), dtype=torch.bool, device=self.device)
            
            # 创建用于存储目标的张量
            target_boxes = torch.zeros((0, 4), device=self.device)  # 存储所有目标框
            target_classes = torch.zeros(0, dtype=torch.long, device=self.device)  # 存储所有目标类别
            
            # 处理targets
            if isinstance(tgt, torch.Tensor) and len(tgt.shape) == 2:  # 如果是单个batch的targets
                valid_targets = tgt
            else:
                valid_targets = torch.cat([t for t in tgt if isinstance(t, torch.Tensor) and t.numel() > 0], dim=0)
            
            if valid_targets.numel() > 0:
                # 将目标坐标转换为网格索引
                grid_xy = (valid_targets[:, 1:3] * torch.tensor([W, H], device=self.device)).long()
                
                # 确保索引在有效范围内
                grid_xy[:, 0].clamp_(0, W-1)
                grid_xy[:, 1].clamp_(0, H-1)
                
                # 为每个目标在所有anchor上设置mask
                for i in range(len(valid_targets)):
                    x, y = grid_xy[i]
                    obj_mask[:, :, y, x] = True
                    
                # 存储目标框和类别
                target_boxes = valid_targets[:, 1:5]  # x, y, w, h
                target_classes = valid_targets[:, 0].long()
            
            noobj_mask = ~obj_mask
            
            # 计算定位损失（只针对有目标的网格）
            if torch.any(obj_mask):
                # 提取预测的box参数 (x, y, w, h)
                pred_box = pred[..., :4]  # [B, A, H, W, 4]
                # 提取预测的类别
                pred_cls = pred[..., 4:]  # [B, A, H, W, num_classes]
                
                # 计算定位损失
                loc_loss = self.mse(
                    pred_box[obj_mask],
                    target_boxes.repeat(obj_mask.sum() // len(target_boxes), 1)
                )
                
                # 计算分类损失
                cls_loss = self.bce(
                    pred_cls[obj_mask],
                    torch.zeros(pred_cls[obj_mask].shape[0], pred_cls.shape[-1], device=self.device).scatter_(
                        1, target_classes.repeat(obj_mask.sum() // len(target_classes)).unsqueeze(1), 1
                    )
                )
                
                # 计算目标置信度损失
                obj_loss = self.bce(
                    pred[..., 4][obj_mask],  # objectness score
                    torch.ones_like(pred[..., 4][obj_mask])
                )
            else:
                loc_loss = torch.tensor(0.0, device=self.device)
                cls_loss = torch.tensor(0.0, device=self.device)
                obj_loss = torch.tensor(0.0, device=self.device)
            
            # 计算非目标置信度损失
            if torch.any(noobj_mask):
                noobj_loss = self.bce(
                    pred[..., 4][noobj_mask],  # objectness score
                    torch.zeros_like(pred[..., 4][noobj_mask])
                )
            else:
                noobj_loss = torch.tensor(0.0, device=self.device)
            
            # 该尺度的总损失
            scale_loss = (
                loc_loss + 
                cls_loss + 
                self.obj_scale * obj_loss + 
                self.noobj_scale * noobj_loss
            )
            
            total_loss += scale_loss
            
        return total_loss / len(predictions)  # 返回平均损失
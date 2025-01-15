import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.obj_scale = 1
        self.noobj_scale = 0.5
        
    def forward(self, predictions, targets):
        """
        predictions: 模型输出 [batch_size, num_anchors, grid_size, grid_size, num_classes + 5]
        targets: 真实标签 [batch_size, max_objects, 5] (class, x, y, w, h)
        """
        obj_mask = targets[..., 0] > 0  # 目标存在的mask
        noobj_mask = ~obj_mask
        
        # 计算定位损失（只针对有目标的网格）
        loc_loss = self.mse(
            predictions[obj_mask][..., 1:5],
            targets[obj_mask][..., 1:5]
        )
        
        # 计算分类损失
        cls_loss = self.bce(
            predictions[obj_mask][..., 5:],
            targets[obj_mask][..., 0].long()
        )
        
        # 计算目标置信度损失
        obj_loss = self.bce(
            predictions[..., 0][obj_mask],
            torch.ones_like(predictions[..., 0][obj_mask])
        )
        
        # 计算非目标置信度损失
        noobj_loss = self.bce(
            predictions[..., 0][noobj_mask],
            torch.zeros_like(predictions[..., 0][noobj_mask])
        )
        
        # 总损失
        total_loss = (
            loc_loss + 
            cls_loss + 
            self.obj_scale * obj_loss + 
            self.noobj_scale * noobj_loss
        )
        
        return total_loss 
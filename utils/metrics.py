import numpy as np
from collections import defaultdict

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 计算交集区域
    xi1 = max(x1 - w1/2, x2 - w2/2)
    yi1 = max(y1 - h1/2, y2 - h2/2)
    xi2 = min(x1 + w1/2, x2 + w2/2)
    yi2 = min(y1 + h1/2, y2 + h2/2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # 计算并集区域
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def calculate_map(predictions, targets, iou_threshold=0.5):
    """计算mAP"""
    # 按类别整理预测结果和真实标签
    pred_boxes = defaultdict(list)
    true_boxes = defaultdict(list)
    
    # 整理数据
    for pred, target in zip(predictions, targets):
        pred_cls = pred[0]
        pred_boxes[pred_cls].append(pred[1:])  # [x,y,w,h,conf]
        
        target_cls = target[0]
        true_boxes[target_cls].append(target[1:])  # [x,y,w,h]
    
    # 计算每个类别的AP
    aps = []
    for cls in true_boxes.keys():
        if cls not in pred_boxes:
            continue
            
        # 计算该类别的所有预测框与真实框的IoU
        class_preds = np.array(pred_boxes[cls])
        class_trues = np.array(true_boxes[cls])
        
        # 按置信度排序
        sorted_idx = np.argsort(-class_preds[:, -1])
        class_preds = class_preds[sorted_idx]
        
        # 计算PR曲线
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        for pred_idx, pred_box in enumerate(class_preds):
            max_iou = 0
            max_idx = -1
            
            # 找到最匹配的真实框
            for true_idx, true_box in enumerate(class_trues):
                iou = calculate_iou(pred_box[:4], true_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = true_idx
            
            if max_iou >= iou_threshold:
                tp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # 计算累积值
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # 计算精确率和召回率
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / len(class_trues)
        
        # 计算AP
        ap = np.trapz(precision, recall)
        aps.append(ap)
    
    return np.mean(aps)  # 返回mAP 
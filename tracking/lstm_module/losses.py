import torch
import numpy as np # For pi

def giou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Calculates the Generalized IoU (GIoU) loss between predicted and target bounding boxes.

    Boxes are expected in (cx, cy, w, h) format.

    Args:
        pred_boxes (torch.Tensor): Predicted boxes, shape (N, 4).
        target_boxes (torch.Tensor): Target boxes, shape (N, 4).
        eps (float): Small epsilon value to prevent division by zero.

    Returns:
        torch.Tensor: Mean GIoU loss over the batch (scalar).
    """
    # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
    px1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    py1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    px2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    py2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    tx1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
    ty1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
    tx2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
    ty2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

    # Intersection area
    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_w * inter_h

    # Union area
    area_pred = (px2 - px1) * (py2 - py1)
    area_target = (tx2 - tx1) * (ty2 - ty1)
    union = area_pred + area_target - intersection + eps

    # IoU
    iou = intersection / union

    # Smallest enclosing box area
    c_x1 = torch.min(px1, tx1)
    c_y1 = torch.min(py1, ty1)
    c_x2 = torch.max(px2, tx2)
    c_y2 = torch.max(py2, ty2)
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1) + eps

    # GIoU calculation
    giou = iou - (c_area - union) / c_area

    # Loss is 1 - GIoU
    loss = 1.0 - giou

    return loss.mean() # Return mean loss over the batch


def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Calculates the Complete IoU (CIoU) loss between predicted and target bounding boxes.

    Boxes are expected in (cx, cy, w, h) format. CIoU considers overlap,
    center distance, and aspect ratio consistency.

    Args:
        pred_boxes (torch.Tensor): Predicted boxes, shape (N, 4).
        target_boxes (torch.Tensor): Target boxes, shape (N, 4).
        eps (float): Small epsilon value to prevent division by zero.

    Returns:
        torch.Tensor: Mean CIoU loss over the batch (scalar).
    """
    # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
    px, py, pw, ph = pred_boxes.unbind(-1)
    tx, ty, tw, th = target_boxes.unbind(-1)
    px1, py1 = px - pw / 2, py - ph / 2
    px2, py2 = px + pw / 2, py + ph / 2
    tx1, ty1 = tx - tw / 2, ty - th / 2
    tx2, ty2 = tx + tw / 2, ty + th / 2

    # Intersection area
    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_w * inter_h

    # Union area
    area_pred = (px2 - px1) * (py2 - py1)
    area_target = (tx2 - tx1) * (ty2 - ty1)
    union = area_pred + area_target - intersection + eps

    # IoU
    iou = intersection / union

    # Smallest enclosing box
    c_x1 = torch.min(px1, tx1)
    c_y1 = torch.min(py1, ty1)
    c_x2 = torch.max(px2, tx2)
    c_y2 = torch.max(py2, ty2)

    # Diagonal length of the enclosing box squared
    c_diag_sq = torch.pow(c_x2 - c_x1, 2) + torch.pow(c_y2 - c_y1, 2) + eps

    # Distance between center points squared
    center_dist_sq = torch.pow(px - tx, 2) + torch.pow(py - ty, 2)

    # Aspect ratio consistency term (v)
    arctan_pred = torch.atan(pw / (ph + eps))
    arctan_target = torch.atan(tw / (th + eps))
    v = (4 / (np.pi ** 2)) * torch.pow(arctan_pred - arctan_target, 2)

    # Trade-off parameter (alpha) - avoids optimization focus issues when IoU is high
    with torch.no_grad(): # Alpha calculation should not contribute to gradients
        alpha = v / (1 - iou + v + eps)

    # CIoU calculation
    # Penalty = (Center Distance Penalty) + (Aspect Ratio Penalty)
    penalty = (center_dist_sq / c_diag_sq) + (alpha * v)
    ciou = iou - penalty

    # Loss is 1 - CIoU
    loss = 1.0 - ciou

    return loss.mean() # Return mean loss over the batch
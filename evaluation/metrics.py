import torch
import numpy as np

def calculate_segmentation_metrics(pred_mask, true_mask, threshold=0.5):
    """
    Calculates Dice Score and IoU for a single batch or image.
    Both masks should be tensors of the same shape.
    """
    # 1. Binarize the prediction (convert probabilities to 0 or 1)
    pred_binary = (pred_mask > threshold).float()
    true_binary = true_mask.float()
    
    # Flatten the tensors to 1D arrays for easy math
    pred_flat = pred_binary.view(-1)
    true_flat = true_binary.view(-1)
    
    # Calculate Intersection and Union
    intersection = (pred_flat * true_flat).sum()
    
    # 2. Calculate Dice Score
    # Formula: (2 * Intersection) / (Total Pixels in Pred + Total Pixels in True)
    # We add 1e-6 (a tiny number) to prevent dividing by zero if both masks are empty!
    dice_score = (2. * intersection + 1e-6) / (pred_flat.sum() + true_flat.sum() + 1e-6)
    
    # 3. Calculate IoU (Jaccard Index)
    # Formula: Intersection / (Pred + True - Intersection)
    union = pred_flat.sum() + true_flat.sum() - intersection
    iou_score = (intersection + 1e-6) / (union + 1e-6)
    
    return dice_score.item(), iou_score.item()
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class PneumoniaLoss(nn.Module):
    def __init__(self):
        super(PneumoniaLoss, self).__init__()
        # Dice Loss handles class imbalance (focuses on the pneumonia shape)
        self.dice_loss = smp.losses.DiceLoss(mode='binary')
        
        # BCE handles pixel-by-pixel binary classification probability
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        # We add them together. You can also weight them (e.g., 0.7 * Dice + 0.3 * BCE)
        dice = self.dice_loss(y_pred, y_true)
        bce = self.bce_loss(y_pred, y_true)
        
        return dice + bce
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .dicom_loader import load_dicom_image
from .mask_generator import generate_mask

class PneumoniaDataset(Dataset):
    def __init__(self, df, image_dir, img_size=256, is_train=True):
        self.df = df
        self.image_dir = image_dir
        self.img_size = img_size
        
        # Get unique patients since some have multiple bounding boxes
        self.patient_ids = df['patientId'].unique()
        
        # 🚨 Medical-Safe Augmentations Only 🚨
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5), # Left/Right flip is anatomically okay for general lung tissue learning
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5), # Slight movements
                A.RandomBrightnessContrast(p=0.2), # Simulates different X-ray machine exposures
                ToTensorV2(), # Converts to PyTorch Tensor format (C, H, W)
            ])
        else:
            self.transform = A.Compose([
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # 1. Load Image
        img_path = os.path.join(self.image_dir, f"{patient_id}.dcm")
        image = load_dicom_image(img_path, self.img_size)
        
        # 2. Extract all bounding boxes for this patient
        patient_data = self.df[self.df['patientId'] == patient_id]
        bboxes = patient_data[['x', 'y', 'width', 'height']].values.tolist()
        
        # 3. Generate Mask
        mask = generate_mask(bboxes, self.img_size)
        
        # 4. Apply Augmentations (applies to BOTH image and mask simultaneously)
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed['image'] / 255.0 # Scale tensor to 0-1
        mask_tensor = transformed['mask'].unsqueeze(0) # Add channel dim: (1, H, W)
        
        return image_tensor.float(), mask_tensor.float()
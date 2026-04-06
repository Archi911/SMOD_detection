import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from data_pipeline.dataset import PneumoniaDataset
from models.attention_unet import build_model
from evaluation.metrics import calculate_segmentation_metrics

# 1. Load Your 4-Epoch Trained Model
print("⏳ Loading your trained model...")
model = build_model(encoder_name=config.ENCODER, weights=None)
model.load_state_dict(torch.load("models/best_pneumonia_model.pth", map_location="cpu"))
model.eval()
print("✅ Model Loaded!")

# 2. Fake the Split to get a "Test Set"
df = pd.read_csv("data/stage_2_train_labels.csv") # Make sure your CSV is in local data folder
df[['x', 'y', 'width', 'height']] = df[['x', 'y', 'width', 'height']].fillna(0)

# Extract a small 10% test set for local evaluation
_, test_df = train_test_split(df, test_size=0.1, random_state=42)
test_dataset = PneumoniaDataset(test_df, "data/stage_2_train_images", is_train=False)

# Keep batch size small for local PC (e.g., 4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 3. Run Evaluation
print(f"🚀 Running Evaluation on {len(test_df)} patients...")
total_dice = 0.0
total_iou = 0.0
batches = 0

with torch.no_grad():
    for images, masks in test_loader:
        predictions = model(images)
        predictions = torch.sigmoid(predictions)
        
        dice, iou = calculate_segmentation_metrics(predictions, masks)
        total_dice += dice
        total_iou += iou
        batches += 1
        print(f"Batch {batches} -> Dice: {dice:.4f} | IoU: {iou:.4f}")

print("\n🏆 --- FINAL RESUME METRICS ---")
print(f"Average Dice Score: {(total_dice/batches):.4f}")
print(f"Average IoU: {(total_iou/batches):.4f}")
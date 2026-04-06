import os
import torch

# Automatically detect if we are on Kaggle or Local
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    DATA_DIR = "/kaggle/input/rsna-pneumonia-detection-challenge"
    OUTPUT_DIR = "/kaggle/working/"
else:
    # Local paths (just for testing with a few sample images later)
    DATA_DIR = "./data"
    OUTPUT_DIR = "./output"

TRAIN_CSV = os.path.join(DATA_DIR, "stage_2_train_labels.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "stage_2_train_images")

# Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Config
ENCODER = "efficientnet-b0"
WEIGHTS = "imagenet"
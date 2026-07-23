import os
import torch
import pandas as pd
import torch.nn as nn 
from torch.utils.data import DataLoader
import config

from data_pipeline.dataset import PneumoniaDataset
from models.unetplusplus import UNetPlusPlus
from training.loss import PneumoniaSMODLoss
from training.trainer import train_one_epoch, validate_one_epoch

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("Loading datasets...")
    train_df = pd.read_csv(config.TRAIN_CSV)
    val_df = pd.read_csv(config.VAL_CSV) 
    
    train_df[['x', 'y', 'width', 'height']] = train_df[['x', 'y', 'width', 'height']].fillna(0)
    val_df[['x', 'y', 'width', 'height']] = val_df[['x', 'y', 'width', 'height']].fillna(0)

    train_dataset = PneumoniaDataset(train_df, config.IMAGE_DIR, config.IMG_SIZE, is_train=True)
    val_dataset = PneumoniaDataset(val_df, config.IMAGE_DIR, config.IMG_SIZE, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Initializing Model on {config.DEVICE}...")
    model = UNetPlusPlus() 
    
    # Wrapping in DataParallel for Kaggle's dual T4 GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for parallel training!")
        model = nn.DataParallel(model)
        
    model = model.to(config.DEVICE)
    
    criterion = PneumoniaSMODLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
    
    scaler = torch.amp.GradScaler('cuda')
    best_val_loss = float('inf')
    
    print(f" Starting UNet++ Training...")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.EPOCHS}]")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE, scaler, config.ACCUMULATION_STEPS)
        val_loss = validate_one_epoch(model, val_loader, criterion, config.DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Clean state_dict extraction (handles DataParallel unwrapping)
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        # 1. Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_save_path = os.path.join(config.OUTPUT_DIR, "best_unetplusplus_model.pth")
            torch.save(state_dict, best_save_path)
            print(f" New best model saved to {best_save_path}")
            
        # 2. Save Last Model
        last_save_path = os.path.join(config.OUTPUT_DIR, "last_unetplusplus_model.pth")
        torch.save(state_dict, last_save_path)

if __name__ == "__main__":
    main()
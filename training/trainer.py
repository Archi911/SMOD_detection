import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    
    # tqdm gives a nice progress bar in the Kaggle console
    pbar = tqdm(dataloader, desc="Training")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # ⚡ Mixed Precision Forward Pass ⚡
        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss = criterion(predictions, masks)
            
        # Mixed Precision Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, masks)
            running_loss += loss.item()
            
    return running_loss / len(dataloader)
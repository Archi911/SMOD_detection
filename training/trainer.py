# import torch
# from tqdm import tqdm

# def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
#     model.train()
#     running_loss = 0.0
    
#     # tqdm gives a nice progress bar in the Kaggle console
#     pbar = tqdm(dataloader, desc="Training")
    
#     for images, masks in pbar:
#         images = images.to(device)
#         masks = masks.to(device)
        
#         optimizer.zero_grad()
        
#         # ⚡ Mixed Precision Forward Pass ⚡
#         with torch.cuda.amp.autocast():
#             predictions = model(images)
#             loss = criterion(predictions, masks)
            
#         # Mixed Precision Backward Pass
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
        
#         running_loss += loss.item()
#         pbar.set_postfix(loss=loss.item())
        
#     return running_loss / len(dataloader)

# def validate_one_epoch(model, dataloader, criterion, device):
#     model.eval()
#     running_loss = 0.0
    
#     with torch.no_grad():
#         pbar = tqdm(dataloader, desc="Validation")
#         for images, masks in pbar:
#             images = images.to(device)
#             masks = masks.to(device)
            
#             predictions = model(images)
#             loss = criterion(predictions, masks)
#             running_loss += loss.item()
            
#     return running_loss / len(dataloader)


import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for i, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        with torch.amp.autocast('cuda'):
            predictions = model(images)
            loss = criterion(predictions, masks) / accumulation_steps
            
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accumulation_steps
        pbar.set_postfix(loss=(loss.item() * accumulation_steps))
        
    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            predictions = model(images)
            loss = criterion(predictions, masks)
            running_loss += loss.item()
            
    return running_loss / len(dataloader)
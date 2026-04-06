import numpy as np
import cv2

def generate_mask(bboxes, img_size=256, original_size=1024):
    """
    Converts a list of bounding boxes into a binary segmentation mask.
    """
    mask = np.zeros((img_size, img_size), dtype=np.float32)
    scale = img_size / original_size
    
    # If no bboxes (healthy patient), return empty mask
    if not isinstance(bboxes, list) or len(bboxes) == 0:
        return mask
        
    for bbox in bboxes:
        if np.isnan(bbox[0]): # Check for NaN values
            continue
            
        x, y, w, h = bbox
        x1 = int(x * scale)
        y1 = int(y * scale)
        x2 = int((x + w) * scale)
        y2 = int((y + h) * scale)
        
        # Fill the mask with 1s inside the bounding box
        mask[y1:y2, x1:x2] = 1.0
        
    return mask
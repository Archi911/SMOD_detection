import pydicom
import cv2
import numpy as np

def load_dicom_image(path, img_size=256):
    """Reads a DICOM file, normalizes it, and resizes it."""
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array
    
    # Normalize to 0-1 range
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    
    # Convert to 8-bit image (0-255) for standard processing
    image = (image * 255).astype(np.uint8)
    
    # Resize
    image = cv2.resize(image, (img_size, img_size))
    
    # Add channel dimension (1, H, W) for PyTorch
    image = np.expand_dims(image, axis=-1) 
    return image
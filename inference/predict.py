import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import sys
import os

# Add the root directory to the path so we can import our other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.attention_unet import build_model
from inference.severity import calculate_severity

def load_trained_model(model_path, device="cpu"):
    """Loads the saved weights into the Attention U-Net."""
    model = build_model(encoder_name="efficientnet-b0", weights=None)
    
    # Load weights safely (mapping to CPU since you likely don't have a massive GPU locally)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device="cpu"):
    """Runs a single X-ray image through the model and returns the mask."""
    # Load and preprocess the DICOM
    dicom = pydicom.dcmread(image_path)
    image = dicom.pixel_array
    original_shape = image.shape
    
    image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
    image = (image * 255).astype(np.uint8)
    image_resized = cv2.resize(image, (256, 256))
    
    # Convert to tensor shape: (Batch, Channel, Height, Width)
    tensor_img = torch.tensor(image_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    tensor_img = tensor_img.to(device)
    
    with torch.no_grad():
        pred_mask = model(tensor_img)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.squeeze().cpu().numpy()
        
    # Resize the predicted mask back to the original X-ray size
    pred_mask_original_size = cv2.resize(pred_mask, (original_shape[1], original_shape[0]))
    
    return image, pred_mask_original_size

def overlay_heatmap(image, mask, threshold=0.5):
    """Creates a visual overlay of the pneumonia region."""
    binary_mask = (mask > threshold).astype(np.uint8)
    
    heatmap = np.zeros((*image.shape, 3), dtype=np.uint8)
    heatmap[binary_mask == 1] = [255, 0, 0] # Red color for pneumonia
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.7, heatmap, 0.3, 0)
    return overlay

# ==========================================
# LOCAL TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Local Inference Test...")
    
    # Define paths (Adjust these if your folders are named differently)
    MODEL_PATH = "models/best_pneumonia_model.pth"
    TEST_IMAGE = "data/sample.dcm"
    
    # 1. Check if files exist before running
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ Error: Test image not found at {TEST_IMAGE}")
        sys.exit(1)
        
    # 2. Load Model
    print("⏳ Loading Attention U-Net weights...")
    model = load_trained_model(MODEL_PATH, device="cpu")
    print("✅ Model loaded successfully!")
    
    # 3. Predict
    print(f"🧠 Analyzing X-Ray: {TEST_IMAGE}...")
    original_img, predicted_mask = predict_image(model, TEST_IMAGE, device="cpu")
    
    # 4. Calculate Severity
    severity_info = calculate_severity(predicted_mask)
    print("\n📊 --- DIAGNOSIS RESULTS ---")
    print(f"Severity Category: {severity_info['category']}")
    print(f"Infected Area: {severity_info['infection_percentage']}%")
    print("---------------------------\n")
    
    # 5. Visualize
    print("🖼️ Generating visual overlay...")
    overlay = overlay_heatmap(original_img, predicted_mask)
    
    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")
    
    axes[1].imshow(overlay)
    axes[1].set_title(f"AI Prediction: {severity_info['category']}")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

   
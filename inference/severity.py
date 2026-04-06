import numpy as np

def calculate_severity(predicted_mask, lung_area_estimate=0.7):
    """
    Calculates the severity of pneumonia based on the segmentation mask.
    
    Args:
        predicted_mask (np.array): 2D array (H, W) with values 0 to 1.
        lung_area_estimate (float): Heuristic for % of image that is lung tissue.
        
    Returns:
        dict: Contains severity category and precise percentage.
    """
    # 1. Binarize the mask (anything above 0.5 probability is considered pneumonia)
    binary_mask = (predicted_mask > 0.5).astype(np.float32)
    
    # 2. Calculate areas
    total_pixels = binary_mask.size
    estimated_lung_pixels = total_pixels * lung_area_estimate
    infected_pixels = np.sum(binary_mask)
    
    # 3. Calculate Ratio
    infection_ratio = infected_pixels / estimated_lung_pixels
    
    # 4. Categorize Severity
    if infection_ratio == 0:
        category = "Healthy"
    elif infection_ratio < 0.10:
        category = "Mild"
    elif infection_ratio < 0.30:
        category = "Moderate"
    else:
        category = "Severe"
        
    return {
        "category": category,
        "infection_percentage": round(infection_ratio * 100, 2),
        "infected_pixels": int(infected_pixels)
    }
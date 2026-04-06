import segmentation_models_pytorch as smp
import torch.nn as nn

def build_model(encoder_name="efficientnet-b0", weights="imagenet"):
    """
    Builds an Attention U-Net using an EfficientNet encoder.
    The 'scse' attention type helps the model focus on lesions and ignore background bones/tissue.
    """
    model = smp.Unet(
        encoder_name=encoder_name,        # Lightweight, state-of-the-art encoder
        encoder_weights=weights,          # Pretrained on ImageNet
        in_channels=1,                    # X-rays are grayscale (1 channel)
        classes=1,                        # Binary segmentation (Pneumonia vs Background)
        decoder_attention_type="scse"     # This is the "Attention" in Attention U-Net
    )
    
    return model

# Quick test block to ensure it compiles locally
if __name__ == "__main__":
    import torch
    dummy_model = build_model()
    dummy_input = torch.randn(2, 1, 256, 256)
    output = dummy_model(dummy_input)
    print(f"Model output shape: {output.shape}") # Should be [2, 1, 256, 256]
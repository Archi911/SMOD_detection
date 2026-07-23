import segmentation_models_pytorch as smp
import config

# The UNet++ Architecture
unet_plus_plus = smp.UnetPlusPlus(
    encoder_name=config.ENCODER,    # Pulling from config.py 
    encoder_weights=config.WEIGHTS, # Pulling from config.py
    in_channels=1,                  # Because X-rays are grayscale
    classes=1,                      # Binary mask (Pneumonia or Background)
)

model = unet_plus_plus.to(config.DEVICE)
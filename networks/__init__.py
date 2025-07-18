# networks/__init__.py
import torch.nn as nn

# Models
from .red_cnn import RED_CNN
from .sr_cnn import SR_CNN
from .unet import UNet

# Get model
def get_model(model: str) -> type[nn.Module]:
    models_map = {
        'red_cnn': RED_CNN,
        'sr_cnn': SR_CNN,
        'unet': UNet
    }

    model = model.lower()
    if model not in models_map:
        raise ValueError(f"Invalid model '{model}'. Available: {list(models_map.keys())}")

    print(f"Model {model} is loaded")
    return models_map[model]
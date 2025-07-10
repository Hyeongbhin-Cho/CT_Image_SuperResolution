# networks/__init__.py
import torch.nn as nn

# Models
from .red_cnn import RED_CNN

# Get model
def get_model(model: str) -> type[nn.Module]:
    models = {
        'red_cnn': RED_CNN
    }

    model = model.lower()
    if model not in models:
        raise ValueError(f"Invalid model '{model}'. Available: {list(models.keys())}")

    return models[model]
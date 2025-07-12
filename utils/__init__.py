# utils/__init__.py

# dataloader.py
from .dataloader import SrDataset
from torch.utils.data import DataLoader

def get_loader(mode,
               data_path,
               transform=None,
               batch_size=4,
               num_workers=1,
               shuffle=True):
    
    mode_map = {
        'train': 'train',
        'val': 'val',
        'eval': 'eval'
    }
    
    if mode not in mode_map:
        raise ValueError(f'Mode {mode} is not valid, mode must be one of tran, val and eval')
    
    dataset = SrDataset(data_path, transform=transform, mode=mode)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print(f'{mode}_dataset loaded.')
    return data_loader

# transforms.py

from .transforms import (
    TransformPipeline,
    ToTensor,
    HounsfieldUnit,
    Normalization,
    RandomFlip,
    RandomCrop,
    Interpolation,
    RandomRotate90
)

def get_transforms(config):
    transform_map = {
        'totensor': ToTensor,
        'hounsfieldunit': HounsfieldUnit,
        'normalization': Normalization,
        'randomcrop': RandomCrop,
        'randomflip': RandomFlip,
        'interpolation': Interpolation,
        'randomrotate90': RandomRotate90 
    }
    
    transform_type = config.get('type', ['totensor'])

    if transform_type[-1].lower() != 'totensor':
        transform_type.append('totensor')
    
    lower_transform_type = [t.lower() for t in transform_type]
    assert 'totensor' not in lower_transform_type[:-1], "'ToTensor' must be the last transform."

    # transform pipline
    pipeline = []
    for t in transform_type:
        t = t.lower()
        if t not in transform_map:
            raise ValueError(f"Unknown transform type: {t}")
        transform_args = config.get(t, {}) or {}
        Transform = transform_map[t]
        pipeline.append(Transform(**transform_args))

    return TransformPipeline(pipeline)
        
        
# measure.py
from .measure import compute_measure

# logger.py
from .logger import get_logger

# ealrystopping.py
from .earlystopping import EarlyStopping
# utils/dataloader.py
import os
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from utils.transforms import *

class SrDataset(Dataset):
    """CT Image SuperResolution Dataset

    Args:
        data_dir: dataset path
        transform: dataset transform class
        mode: choice ('train', 'val', 'eval')
    """
    def __init__(self, data_dir: str, transform:object|list[object]=None, mode:str='train'):
        self.transform = TransformPipeline(transforms=transform) if transform else None
        
        mode_to_file = {
            "train": "data_train.mat",
            "val": "data_val.mat",
            "eval": "data_test.mat"
        }
        if mode not in mode_to_file:
            raise ValueError(f"Invalid mode '{mode}' in SrDataset")

        data_path = os.path.join(data_dir, mode_to_file[mode])

        self.lst_lr, self.lst_hr = [], []

        with h5py.File(data_path, 'r') as f:
            self.lst_lr.append(f['img_lr'][:].astype(np.float32))
            self.lst_hr.append(f['img_hr'][:].astype(np.float32))

        self.lst_lr = np.concatenate(self.lst_lr, axis=0)  # [N, H, W]
        self.lst_hr = np.concatenate(self.lst_hr, axis=0)  # [N, H, W]


    def __len__(self) -> int:
        return self.lst_hr.shape[0]


    def __getitem__(self, index:int) -> object:
        lr = self.lst_lr[index, :, :]
        hr = self.lst_hr[index, :, :]

        if hr.ndim == 2:
            lr = lr[:, :, np.newaxis]
            hr = hr[:, :, np.newaxis]
        
        data = {'lr': lr, 'hr': hr}
        
        if self.transform:
            data = self.transform(data)

        return data

        
def get_loader(mode,
               data_path,
               transform=None,
               batch_size=4,
               num_workers=1,
               shuffle=True):
    
    dataset = SrDataset(data_path, transform=transform, mode=mode)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print(f'{mode}_dataset loaded.')
    return data_loader
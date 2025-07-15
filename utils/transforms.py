# utils/transforms.py
import numpy as np
import torch
import cv2

class TransformPipeline:
    def __init__(self, transforms):
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        
    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            if value.ndim == 3:
                value = value.transpose((2, 0, 1))
            data[key] = torch.from_numpy(value.copy()).float()
        
        return data
    
    
class HounsfieldUnit(object):
    def __init__(self, water_coefficient=0.0192):
        self.water_coefficient = water_coefficient
        
    def __call__(self, data):
        for key, value in data.items():
            if key != 'mask':
                data[key] = 1000 * (value - self.water_coefficient) / self.water_coefficient

        return data


class Normalization(object):
    def __init__(self, clip_min=-1024, clip_max=3071):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, data):
        for key, value in data.items():
            if key != 'mask':
                value = np.clip(value, self.clip_min, self.clip_max)
                data[key] = (value - self.clip_min) / (self.clip_max - self.clip_min)

        return data


class RandomFlip(object):
    def __call__(self, data):

        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=0)

        if np.random.rand() > 0.5:
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)

        return data


class RandomCrop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, data):
        h, w = data['hr'].shape[:2]
        
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(f"Image size ({h},{w}) smaller than patch size {self.patch_size}")

        top = np.random.randint(0, h - self.patch_size)
        left = np.random.randint(0, w - self.patch_size)

        id_y = np.arange(top, top + self.patch_size, 1)[:, np.newaxis]
        id_x = np.arange(left, left + self.patch_size, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data
    
    
class Interpolation(object):
    def __init__(self, scale):
        assert type(scale) is int, "scale factor must be INTEGER"
        self.scale = scale
    
    def __call__(self, data):
        lr = data['lr']
        lr = cv2.resize(lr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        if lr.ndim == 2:
            lr = lr[:, :, np.newaxis]
        
        data['lr'] = lr
        
        return data
    
    
class RandomRotate90(object):
    def __call__(self, data):
        k = np.random.choice([0, 1, 2, 3])  # 0도, 90도, 180도, 270도
        for key, value in data.items():
            data[key] = np.rot90(value, k).copy()
        return data
    
    
class SobelGradientMagnitude(object):
    def __init__(self, threshold):
        self.threshold = threshold
        
    def __call__(self, data):
        edges = {}
        for key, value in data.items():
            dx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(dx, dy)
            
            if self.threshold > 0:
                magnitude = np.where(magnitude > self.threshold, magnitude, 0)
                
            if magnitude.ndim == 2:
                magnitude = magnitude[:, :, np.newaxis]
                
            edges[f"{key}_edge"] = magnitude
            
        data.update(edges)
        
        return data
        

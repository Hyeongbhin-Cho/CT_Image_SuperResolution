# solver/base_solver.py
import os
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import EarlyStopping

class BaseSolver(ABC):
    def __init__(self, config:dict, model: nn.Module,
                 train_loader: DataLoader = None, train_logger: logging.Logger = None,
                 eval_loader: DataLoader = None, eval_logger: logging.Logger = None,
                 val_loader: DataLoader = None, val_logger:logging.Logger = None,
                 vs_loader: DataLoader = None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.train_logger = train_logger
        self.eval_loader = eval_loader
        self.eval_logger = eval_logger
        self.val_loader = val_loader
        self.val_logger = val_logger
        self.vs_loader = vs_loader
        
        self.save_path = config.get('save_path', 'save')
        self.load_path = config.get('load_path', 'save')
        self.result_fig = config.get('result_fig', False)
        self.num_imgs = config.get('num_imgs', 0) 
        
        train_config = config['train']
        self.num_epochs = train_config.get('num_epochs', 10)
        self.val_iters = train_config.get('val_iters', 20)
        self.save_iters = train_config.get('save_iters', 20)
        self.record_iters = train_config.get('record_iters', 20)
        self.metrics = train_config.get('metrics' , [])
        
        val_config = config['validate']
        self.mini_batches = val_config.get('mini_batches', 0)

        # Record
        self.train_losses = []
        self.val_losses = []
        self.val_measures = {k: [] for k in self.metrics}

        # CUDA
        device_str = config['device'].get('type', 'cuda')
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Multi GPU
        self.multi_gpu = config['device'].get('multi_gpu', False)
        if self.multi_gpu and torch.cuda.device_count() > 1:
            print(f"[Solver] Using {torch.cuda.device_count()} GPUs.")
            self.model = nn.DataParallel(self.model)

        # Loss Function
        criterion_map = {
            'mse': nn.MSELoss(),
            'l1': nn.L1Loss()
        }
        self.criterion = criterion_map[config['train'].get('criterion', 'mse')]

        # Optimizer
        optimizer_map = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD
        }
        lr = config['train']['lr']
        optimizer_type = config['train']['optimizer'].get('type', 'adam')
        weight_decay = config['train']['optimizer'].get('weight_decay', 1e-6)
        self.optimizer = optimizer_map[optimizer_type](self.model.parameters(),
                                                       lr=lr,
                                                       weight_decay=weight_decay)
        
        # Scheduler
        scheduler_map = {
            'step': StepLR,
            'cosine': CosineAnnealingLR,
            'plateau': ReduceLROnPlateau,
            'cosine_restart': CosineAnnealingWarmRestarts
        }

        scheduler_config = config['train'].get('scheduler', {})
        scheduler_type = scheduler_config.get('type', None)

        if scheduler_type:
            scheduler_class = scheduler_map[scheduler_type]
            scheduler_args = {k: v for k, v in scheduler_config.items() if k != 'type'}
            self.scheduler = scheduler_class(self.optimizer, **scheduler_args)
        else:
            self.scheduler = None
        
        # Early Stopping
        earlystopping_configs = config['train'].get('earlystopping', {})
        self.earlystopping_mode = earlystopping_configs.get('mode', 'min')
        self.best_value = float('inf') if self.earlystopping_mode =='min' else -float('inf')
  
        self.earlystopping = EarlyStopping(patience=earlystopping_configs.get('patience', 10),
                                           min_delta=earlystopping_configs.get('min_delta', 0),
                                           mode=earlystopping_configs.get('mode', 'min'),
                                           enabled=earlystopping_configs.get('enabled', False))
        self.earlystopping_metric = earlystopping_configs.get('metric', 'loss')
        if self.earlystopping_metric != 'loss' and self.earlystopping_metric not in self.metrics:
            raise ValueError(f"Metric '{self.earlystopping_metric}' not found in validation metrics")
        
        
        
        
    def save_model(self, filename: str=None):
        filename = filename or os.path.join(self.save_path, 'latest.pt')
        torch.save(self.model.state_dict(), filename)

    def load_model(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        
    def record_log(self, train_loss, val_loss, val_measure):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        for k, v in val_measure.items():
            self.val_measures[k].append(v)
        
    def save_log(self):
        log_path = os.path.join(self.save_path, 'log')
        os.makedirs(log_path, exist_ok=True)
        
        np.save(os.path.join(log_path, 'train_losses.npy'), np.array(self.train_losses))
        np.save(os.path.join(log_path, 'val_losses.npy'), np.array(self.val_losses))

        df = pd.DataFrame(self.val_measures)
        df.to_csv(os.path.join(log_path, 'val_measures.csv'), index=False)
        
    def save_fig(self):
        fig_path = os.path.join(self.save_path, 'fig')
        os.makedirs(fig_path, exist_ok=True)
        
        # X-ticks
        x_iters = np.arange(0, len(self.train_losses)) * self.record_iters

        # Loss Plot
        plt.figure()
        plt.plot(x_iters, self.train_losses, label='Train Loss')
        plt.plot(x_iters, self.val_losses, label='Val Loss')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Train & Val Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fig_path, 'loss_plot.png'))
        plt.close()

        # Metrics Plot
        df = pd.DataFrame(self.val_measures)
        for metric in df.columns:
            plt.figure()
            plt.plot(x_iters, df[metric], label=metric.upper())
            plt.xlabel("Iter")
            plt.ylabel(metric.upper())
            plt.title(f"Validation {metric.upper()}")
            plt.grid(True)
            plt.savefig(os.path.join(fig_path, f'{metric}_plot.png'))
            plt.close()
            
    def terminate_solver(self):
        print("Training finished. Saving model, logs, figures...")
        self.save_model()
        self.save_log()
        if self.result_fig:
            self.save_fig()
        print(f"Model saved to: {self.save_path}")
        print(f"Logs saved to: {os.path.join(self.save_path, 'log')}")
        print(f"Figures saved to: {os.path.join(self.save_path, 'fig')}")
        print("Terminated.")
        
        if self.train_logger:
            self.train_logger.info(f"Training finished. Saving model, logs, figures..")
            self.train_logger.info("Terminated.")
            
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
    
    @abstractmethod
    def visualize(self):
        pass
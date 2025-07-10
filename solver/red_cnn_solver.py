# solver/red_cnn_solver.py
import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from .base_solver import BaseSolver
from utils.measure import compute_measure

class RedCNNSolver(BaseSolver):
    def __init__(self, config:dict, model: nn.Module,
                 train_loader: DataLoader = None, train_logger: logging.Logger = None,
                 eval_loader: DataLoader = None, eval_logger: logging.Logger = None,
                 val_loader: DataLoader = None, val_logger:logging.Logger = None):
        super().__init__(config=config, model=model,
                         train_loader=train_loader, train_logger=train_logger,
                         eval_loader=eval_loader, eval_logger=eval_logger,
                         val_loader=val_loader, val_logger=val_logger)
        
    def train(self):
        assert self.train_loader is not None, "RedCNNSolver need train loader. RedCNNSolver(..., train_loader=<here>)"
        
        train_finish = False
        train_loss_total = 0.0
        train_loss_iters = 0
        val_loss_avg = 0
        val_measures = {}
        total_iters = 0
        
        self.model.train()
        try:
            # Initial,  Iter == 0
            # Vaildation & Record
            assert self.val_loader is not None, "RedCNNSolver need val loader. RedCNNSolver(..., val_loader=<here>)"  
            val_loss_avg, val_measures = self.validate(metrics=self.metrics)
            train_loss_avg = val_loss_avg
            
            self.record_log(train_loss=train_loss_avg, 
                            val_loss=val_loss_avg,
                            val_measure=val_measures)
            
            if self.train_logger:
                self.train_logger.info("Training start.")
                self.train_logger.info(f"Iter: {total_iters}, Loss: {train_loss_avg:.6f}")
            
            # Train
            for epoch in range(1, self.num_epochs + 1):
                train_pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch}/{self.num_epochs}]", leave=False)

                for data in train_pbar:
                    # To GPU
                    train_x = data['lr'].to(self.device)
                    train_y = data['hr'].to(self.device)

                    pred = self.model(train_x)
                    loss = self.criterion(pred, train_y) 
                    
                    self.optimizer.zero_grad()     
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss_total += loss.item()
                    train_loss_iters += 1
                    
                    total_iters += 1
                    
                    # tqdm progress bar
                    train_pbar.set_postfix({
                        "Loss": f"{loss.item():.6f}",
                        "Iter": total_iters
                    })
                    
                    # Validation
                    if total_iters % self.val_iters == 0:
                        assert self.val_loader is not None, "RedCNNSolver need val loader. RedCNNSolver(..., val_loader=<here>)"  
                        val_loss_avg, val_measures = self.validate(metrics=self.metrics)
                        
                        # Early Stopping
                        earlystopping_mode = self.config['train'].get('earlystopping', {})
                        metric = earlystopping_mode.get('metric', 'psnr')
                        if metric not in self.metrics:
                            raise ValueError(f"Metric '{metric}' not found in validation metrics")
                        
                        if val_measures and self.earlystopping.step(val_measures[metric]):
                            print("Earlystopping")
                            if self.train_logger:
                                self.train_logger.info("Earlystopping")
                            train_finish = True
                            break
                        
                    # Record
                    if total_iters %  self.record_iters == 0:
                        train_loss_avg = train_loss_total / train_loss_iters
                        train_loss_total = 0.0
                        train_loss_iters = 0
                        
                        self.record_log(train_loss=train_loss_avg, 
                                        val_loss=val_loss_avg,
                                        val_measure=val_measures)
                        
                        if self.train_logger:
                            self.train_logger.info(f"Iter: {total_iters}, Loss: {train_loss_avg:.6f}")
                        
                    # Save
                    if total_iters % self.save_iters == 0:
                        network_type = self.config['network']['type']
                        filename = os.path.join(self.config['save_path'], f'{network_type}_iter{total_iters}.pt')
                        self.save_model(filename=filename)
                        
                             
                    # Scheduler
                    if self.scheduler:
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            assert self.val_loader is not None, "RedCNNSolver need val loader. RedCNNSolver(..., val_loader=<here>)" 
                            self.scheduler.step(val_loss_avg)
                        else:
                            self.scheduler.step()
                            
                if train_finish:
                    break
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            if self.train_logger:
                self.train_logger.info("KeyboardInterrupt")
        finally:
            self.terminate_solver()
        
    def validate(self, metrics=['rmse', 'psnr', 'ssim']):
        assert self.val_loader is not None, "RedCNNSolver need val loader. RedCNNSolver(..., val_loader=<here>)"
        self.model.eval()
        
        val_loss_total = 0.0
        measure_total = {k: 0.0 for k in metrics}
        iters = 0
        
        for data in self.val_loader:
            # To GPU
            val_x = data['lr'].to(self.device)
            val_y = data['hr'].to(self.device)
            pred = self.model(val_x)
            
            loss = self.criterion(pred, val_y)
            val_loss_total += loss.item()
                        
            measure = compute_measure(pred, val_y, metrics=metrics, data_range=1.0)
            measure_total = {k: measure_total[k] + measure[k] for k in metrics}
            
            iters += 1
        
        val_loss_avg = val_loss_total / iters
        measure_avg = {k: measure_total[k]/iters for k in metrics}
        
        log = f"[Val] Loss: {val_loss_avg:.6f}"
        for k, v in measure_avg.items():
            log += f" {k.upper()}: {v:.4f}"
            
        tqdm.write(log)
        
        # Val Logger
        if self.val_logger:
            self.val_logger.info(log)
        
        self.model.train()
        
        return val_loss_avg, measure_avg
            
    
    def evaluate(self, metrics=['rmse', 'psnr', 'ssim']):
        assert self.eval_loader is not None, "RedCNNSolver need eval loader. RedCNNSolver(..., eval_loader=<here>)"
        self.load_model(self.config['load_path'])
        self.model.eval()
        
        eval_loss_total = 0.0
        measure_total = {k: 0.0 for k in metrics}
        iters = 0
        
        eval_pbar = tqdm(self.eval_loader, leave=False)
        
        for data in eval_pbar:
            # To GPU
            eval_x = data['lr'].to(self.device)
            eval_y = data['hr'].to(self.device)
            pred = self.model(eval_x)
            
            loss = self.criterion(pred, eval_y)
            eval_loss_total += loss.item()
                        
            measure = compute_measure(pred, eval_y, metrics=metrics, data_range=1.0)
            measure_total = {k: measure_total[k] + measure[k] for k in metrics}
            
            iters += 1
        
        eval_loss_avg = eval_loss_total / iters
        measure_avg = {k: measure_total[k]/iters for k in metrics}
        
        # Eval Logger
        log = f"[Eval] Loss: {eval_loss_avg:.6f}"
        for k, v in measure_avg.items():
            log += f" {k.upper()}: {v:.4f}"
            
        tqdm.write(log)
        
        # Val Logger
        if self.eval_logger:
            self.eval_logger.info(log)
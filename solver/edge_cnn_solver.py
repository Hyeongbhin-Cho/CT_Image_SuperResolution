# solver/sr_cnn_solver.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import numpy as np
import cv2

from .base_solver import BaseSolver
from utils.measure import compute_measure

class EdgeCNN(BaseSolver):
    def __init__(self, config:dict, model: nn.Module,
                 train_loader: DataLoader = None, train_logger: logging.Logger = None,
                 eval_loader: DataLoader = None, eval_logger: logging.Logger = None,
                 val_loader: DataLoader = None, val_logger:logging.Logger = None,
                 vs_loader: DataLoader = None):
        super().__init__(config=config, model=model,
                         train_loader=train_loader, train_logger=train_logger,
                         eval_loader=eval_loader, eval_logger=eval_logger,
                         val_loader=val_loader, val_logger=val_logger,
                         vs_loader=vs_loader)
        
        criterion_map = {
            'mse': nn.MSELoss(),
            'l1': nn.L1Loss()
        }
        
        self.edge_config = config["train"].get('edge', {})
        self.magnitude_threshold = self.edge_config.get('magnitude_threshold', -1)
        self.gamma = self.edge_config.get('gamma', 1.0)
        self.edge_criterion =  criterion_map[self.edge_config.get('criterion', 'mse')]
        
        print("Edge CNN solver is loaded")
        
    def sobel_magnitude(self, data:torch.Tensor):
        sobel_x = torch.tensor([[[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]]], dtype=torch.float32).to(self.device)
        
        sobel_y = torch.tensor([[[[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]]]], dtype=torch.float32).to(self.device)
        
        dx = F.conv2d(data, sobel_x, padding=1)
        dy = F.conv2d(data, sobel_y, padding=1)
        
        magnitude = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        
        if self.magnitude_threshold > 0:
            magnitude = torch.where(magnitude > self.magnitude_threshold, magnitude, 0)
        
        return magnitude
        
    def train(self):
        assert self.train_loader is not None, "EdgeCNNSolver need train loader. EdgeCNNSolver(..., train_loader=<here>)"
        
        train_finish = False
        train_loss_total = 0.0
        train_loss_iters = 0
        val_loss_avg = 0
        val_measures = {k: 0.0 for k in self.metrics}
        total_iters = 0
        
        self.model.train()
        try:
            if self.train_logger:
                self.train_logger.info("Training start.")
            
            # Train
            for epoch in range(1, self.num_epochs + 1):
                train_pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch}/{self.num_epochs}]", leave=False)

                for data in train_pbar:                    
                    # Validation
                    if total_iters % self.val_iters == 0:
                        assert self.val_loader is not None, "EdgeCNNSolver need val loader. EdgeCNNSolver(..., val_loader=<here>)"  
                        val_loss_avg, val_measures = self.validate()
                        
                        # Early Stopping
                        if self.earlystopping.step(val_measures[self.earlystopping_metric]):
                            print("Earlystopping")
                            if self.train_logger:
                                self.train_logger.info("Earlystopping")
                            train_finish = True
                            break
                        
                    # To GPU
                    train_x = data['lr'].to(self.device)
                    train_y = data['hr'].to(self.device)
                    
                    pred = self.model(train_x)
                    
                    edge_pred = self.sobel_magnitude(pred)
                    edge_y = self.sobel_magnitude(train_y)
                    
                    base_loss = self.criterion(pred, train_y)           
                    edge_loss = self.edge_criterion(edge_pred, edge_y) 
                    
                    loss = self.gamma * base_loss + (1 - self.gamma) * edge_loss
                    
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
                        filename = os.path.join(self.save_path, f'{network_type}_iter{total_iters}.pt')
                        self.save_model(filename=filename)
                        
                    # Scheduler
                    if self.scheduler:
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            assert self.val_loader is not None, "EdgeCNNSolver need val loader. EdgeCNNSolver(..., val_loader=<here>)" 
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
        
    def validate(self):
        assert self.val_loader is not None, "EdgeCNNSolver need val loader. EdgeCNNSolver(..., val_loader=<here>)"
        self.model.eval()
        
        val_loss_total = 0.0
        measure_total = {k: 0.0 for k in self.metrics}
        iters = 0

        for data in self.val_loader:
            # To GPU
            val_x = data['lr'].to(self.device)
            val_y = data['hr'].to(self.device)
            pred = self.model(val_x)
            
            edge_pred = self.sobel_magnitude(pred)
            edge_y = self.sobel_magnitude(val_y)
            
            base_loss = self.criterion(pred, val_y)           
            edge_loss = self.edge_criterion(edge_pred, edge_y) 
            
            loss = self.gamma * base_loss + (1 - self.gamma) * edge_loss
        
            val_loss_total += loss.item()
                        
            measure = compute_measure(pred, val_y, metrics=self.metrics, data_range=1.0)
            measure_total = {k: measure_total[k] + measure[k] for k in self.metrics}
            
            iters += 1
        
        val_loss_avg = val_loss_total / iters
        measure_avg = {k: measure_total[k]/iters for k in self.metrics}
        
        log = f"[Val] Loss: {val_loss_avg:.6f}"
        for k, v in measure_avg.items():
            log += f" {k.upper()}: {v:.4f}"
            
        tqdm.write(log)
        
        # Val Logger
        if self.val_logger:
            self.val_logger.info(log)
        
        self.model.train()
        
        return val_loss_avg, measure_avg
            
    
    def evaluate(self):
        assert self.eval_loader is not None, "EdgeCNNSolver need eval loader. EdgeCNNSolver(..., eval_loader=<here>)"
        self.load_model(self.load_path)
        self.model.eval()
        
        eval_loss_total = 0.0
        x_measure_total = {k: 0.0 for k in self.metrics}
        pred_measure_total = {k: 0.0 for k in self.metrics}
        iters = 0
        
        eval_pbar = tqdm(self.eval_loader, leave=False)
        
        for data in eval_pbar:
            # To GPU
            eval_x = data['lr'].to(self.device)
            eval_y = data['hr'].to(self.device)
            pred = self.model(eval_x)
            
            edge_pred = self.sobel_magnitude(pred)
            edge_y = self.sobel_magnitude(eval_y)
            
            base_loss = self.criterion(pred, eval_y)           
            edge_loss = self.edge_criterion(edge_pred, edge_y) 
            
            loss = self.gamma * base_loss + (1 - self.gamma) * edge_loss

            eval_loss_total += loss.item()
                        
            x_measure = compute_measure(eval_x, eval_y, metrics=self.metrics, data_range=1.0)
            x_measure_total = {k: x_measure_total[k] + x_measure[k] for k in self.metrics}
            
            pred_measure = compute_measure(pred, eval_y, metrics=self.metrics, data_range=1.0)
            pred_measure_total = {k: pred_measure_total[k] + pred_measure[k] for k in self.metrics}
            
            iters += 1
        
        eval_loss_avg = eval_loss_total / iters
        x_measure_avg = {k: x_measure_total[k] / iters for k in self.metrics}
        pred_measure_avg = {k: pred_measure_total[k] / iters for k in self.metrics}
        
        # Eval Logger
        log = f"[Eval] Loss: {eval_loss_avg:.6f}"
        log += " [LR]"
        for k, v in x_measure_avg.items():
            log += f" {k.upper()}: {v:.4f}"
            
        log += " [SR]"
        for k, v in pred_measure_avg.items():
            log += f" {k.upper()}: {v:.4f}"
            
        tqdm.write(log)
        
        # Eval Logger
        if self.eval_logger:
            self.eval_logger.info(log)
            

    def visualize(self):
        assert self.vs_loader is not None, "EdgeCNNSolver needs vs_loader. EdgeCNNSolver(..., vs_loader=<here>)"
        self.load_model(self.load_path)
        self.model.eval()

        save_dir = os.path.join(self.save_path, "img")
        os.makedirs(save_dir, exist_ok=True)

        vs_iter = iter(self.vs_loader)

        for i in range(1, self.num_imgs + 1):
            data = next(vs_iter)

            # To GPU
            x = data['lr'].to(self.device)
            y = data['hr'].to(self.device)
            pred = self.model(x)
            
            edge_x = self.sobel_magnitude(x)
            edge_y = self.sobel_magnitude(y)
            edge_pred = self.sobel_magnitude(pred)

            for name, img_tensor in zip(["LR", "HR", "SR", "LR_edge", "HR_edge", "SR_edge"], [x[0], y[0], pred[0], edge_x[0], edge_y[0], edge_pred[0]]):
                img = img_tensor.detach().cpu().numpy()
                if img.shape[-1] == 1:
                    img = img.squeeze(-1)
                else:
                    img = np.transpose(img, (1, 2, 0))
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_dir, f"{name}_{i}.png"), img)

            print(f"Image {i} saved")

        print(f"Images saved to: {save_dir}")
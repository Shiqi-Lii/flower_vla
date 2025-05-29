
import torch
import numpy as np
from collections import defaultdict
import wandb

# First, the shared metrics tracker class
class DatasetMetricsTracker:
    def __init__(self, dataset_mapping, accelerator):
        self.dataset_mapping = dataset_mapping
        self.accelerator = accelerator
        self.reset()
        
    def reset(self):
        self.stored_data = {'losses': [], 'indices': []}
        
    def update(self, losses: torch.Tensor, dataset_indices: torch.Tensor):
        if losses.shape != dataset_indices.shape:
            losses = losses.expand_as(dataset_indices)
        self.stored_data['losses'].append(losses.detach())
        self.stored_data['indices'].append(dataset_indices.detach())
    
    def compute_metrics(self):
        metrics = {}
        if not self.stored_data['losses']:
            return metrics
            
        try:
            all_losses = torch.cat(self.stored_data['losses']).to(self.accelerator.device)
            all_indices = torch.cat(self.stored_data['indices']).to(self.accelerator.device)
            
            # Gather tensors properly
            gathered_losses = self.accelerator.gather(all_losses)
            gathered_indices = self.accelerator.gather(all_indices)
            
            if self.accelerator.is_main_process:
                for idx in torch.unique(gathered_indices):
                    mask = gathered_indices == idx
                    losses = gathered_losses[mask]
                    if len(losses) > 0:
                        metrics[f"val_loss/{self.dataset_mapping[idx.item()]}"] = losses.mean().item()
                
                metrics["val_loss/overall"] = gathered_losses.mean().item()
                    
        except Exception as e:
            print(f"Error in compute_metrics: {e}")
            
        self.reset()
        return metrics
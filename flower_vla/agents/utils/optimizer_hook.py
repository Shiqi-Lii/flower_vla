import numpy as np
import torch as th
import wandb

def wandb_optimizer_hook(optimizer: th.optim.AdamW, args, kwargs):
    grad_norm = 0.0
    param_norm = 0.0
    if wandb.run is not None:
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
                param_norm += p.norm().item() ** 2
        wandb.log({
            "Gradient Norm": grad_norm ** 0.5,
            "Paramter Norm": param_norm ** 0.5
        }, commit=False)
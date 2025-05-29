import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
from torch.optim.lr_scheduler import _LRScheduler
from omegaconf import DictConfig


class InverseSquareRootLRSchedule(LambdaLR):
# Adjusted Copy from https://github.com/huggingface/transformers/blob/9fe3f585bb4ea29f209dc705d269fbe292e1128f/src/transformers/optimization.py#L297

    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, timescale: int = None, last_epoch: int = -1):
        if timescale is None:
            timescale = num_warmup_steps or 10_000
        lr_lambda = partial(self._get_inverse_sqrt_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)

    def _get_inverse_sqrt_schedule_lr_lambda(self, current_step: int, *, num_warmup_steps: int, timescale: int = None):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        shift = timescale - num_warmup_steps
        decay = 1.0 / math.sqrt((current_step + shift) / timescale)
        return decay


class CosineSchedulerWithWarmup(LambdaLR):
# Adjusted Copy from https://github.com/huggingface/transformers/blob/9fe3f585bb4ea29f209dc705d269fbe292e1128f/src/transformers/optimization.py#L144C5-L144C36
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
        lr_lambda = partial(
            self._get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
        super().__init__(optimizer, lr_lambda, last_epoch)

    def _get_cosine_schedule_with_warmup_lr_lambda(
        self, current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


class TriStageLRScheduler(LambdaLR):
    """
    Tri-Stage Learning Rate Scheduler implemented using LambdaLR interface.
    
    The scheduler has three stages:
    1. Warmup: Linear increase from init_lr to peak_lr
    2. Hold: Maintain peak_lr
    3. Decay: Cosine decay from peak_lr to final_lr
    
    Args:
        optimizer (Optimizer): The optimizer to adjust learning rate for
        total_steps (int): Total number of training steps
        phase_ratio (str): String representation of list containing ratios for each phase (must sum to 1)
        init_lr_scale (float): Initial learning rate scale factor
        final_lr_scale (float): Final learning rate scale factor
        last_epoch (int, optional): The index of last epoch. Defaults to -1.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        phase_ratio: str,
        init_lr_scale: float,
        final_lr_scale: float,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.phase_ratio = eval(phase_ratio)
        
        # Calculate steps for each phase
        self.warmup_steps = int(total_steps * self.phase_ratio[0])
        self.hold_steps = int(total_steps * self.phase_ratio[1])
        self.decay_steps = int(total_steps * self.phase_ratio[2])
        
        # Store scaling factors
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale
        
        # Create lambda function for scheduler
        lr_lambda = partial(
            self._get_tri_stage_lambda,
            warmup_steps=self.warmup_steps,
            hold_steps=self.hold_steps,
            decay_steps=self.decay_steps,
            init_lr_scale=init_lr_scale,
            final_lr_scale=final_lr_scale
        )
        
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _get_tri_stage_lambda(
        current_step: int,
        *,
        warmup_steps: int,
        hold_steps: int,
        decay_steps: int,
        init_lr_scale: float,
        final_lr_scale: float
    ) -> float:
        """
        Calculate the learning rate multiplier for the current step.
        """
        # Determine current stage
        if current_step < warmup_steps:
            # Warmup stage: linear increase from init_lr_scale to 1.0
            return init_lr_scale + (1.0 - init_lr_scale) * (current_step / max(1, warmup_steps))
        
        current_step = current_step - warmup_steps
        if current_step < hold_steps:
            # Hold stage: maintain peak learning rate
            return 1.0
        
        current_step = current_step - hold_steps
        if current_step < decay_steps:
            # Decay stage: cosine decay from 1.0 to final_lr_scale
            progress = current_step / max(1, decay_steps)
            return final_lr_scale + 0.5 * (1.0 - final_lr_scale) * (1.0 + math.cos(progress * math.pi))
        
        # After decay: maintain final learning rate
        return final_lr_scale

    @classmethod
    def from_config(cls, optimizer: Optimizer, configs: DictConfig) -> 'TriStageLRScheduler':
        """
        Create scheduler from config dictionary for backward compatibility.
        """
        return cls(
            optimizer=optimizer,
            total_steps=configs.total_steps,
            phase_ratio=configs.phase_ratio,
            init_lr_scale=configs.init_lr_scale,
            final_lr_scale=configs.final_lr_scale
        )
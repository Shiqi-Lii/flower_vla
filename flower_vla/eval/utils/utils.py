import random
import numpy as np
import torch

from omegaconf import OmegaConf
import os


def set_seed(seed: int = 42):
    """Ensure reproducibility by fixing random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables auto-tuning
    torch.backends.cudnn.deterministic = True  # Enforces deterministic algorithms
    torch.backends.cudnn.allow_tf32 = False  # Ensure TF32 doesn't cause small variations

    # Ensuring deterministic operations in CUDA
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Set seed for data augmentation libraries if used (Albumentations, OpenCV, etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Call set_seed before any training happens
set_seed(42)  # Replace 42 with any fixed number for consistency
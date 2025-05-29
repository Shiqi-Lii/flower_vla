import os
import logging
import torch
from safetensors.torch import load_file
from typing import Tuple, Dict, Union, Optional
from accelerate import Accelerator

log = logging.getLogger(__name__)

def load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """Load model weights from a safetensors file."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No safetensors file found at {path}")
        return load_file(path)
    except Exception as e:
        log.error(f"Error loading safetensors file: {e}")
        raise

def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load model weights from a PyTorch state dict."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No state dict found at {path}")
        return torch.load(path, map_location='cpu')
    except Exception as e:
        log.error(f"Error loading state dict: {e}")
        raise

def load_model_weights(
    model: torch.nn.Module,
    weights_path: str,
    strict: bool = False,
    exclude_keys: Optional[list] = None
) -> Tuple[set, set]:
    """
    Load pre-trained model weights for finetuning.
    
    Args:
        model: Target model to load weights into
        weights_path: Path to weights file (.safetensors or .pt)
        strict: Whether to strictly enforce all keys match
        exclude_keys: List of key patterns to exclude from loading
    
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    try:
        # Determine file type and load weights
        if weights_path.endswith('.safetensors'):
            state_dict = load_safetensors(weights_path)
        else:
            state_dict = load_state_dict(weights_path)
            
        # Filter out excluded keys if specified
        if exclude_keys:
            state_dict = {k: v for k, v in state_dict.items() 
                         if not any(ex in k for ex in exclude_keys)}
        
        # Load weights
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        
        # Log results
        if missing:
            log.info(f"Missing keys: {missing}")
        if unexpected:
            log.info(f"Unexpected keys: {unexpected}")
            
        return missing, unexpected
        
    except Exception as e:
        log.error(f"Failed to load model weights: {e}")
        raise

def adjust_state_dict_keys(
    state_dict: Dict[str, torch.Tensor],
    remove_prefix: str = None,
    add_prefix: str = None
) -> Dict[str, torch.Tensor]:
    """
    Adjust state dict keys by adding/removing prefixes.
    Useful when loading weights from different model versions.
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        if remove_prefix and key.startswith(remove_prefix):
            new_key = key[len(remove_prefix):]
            
        if add_prefix:
            new_key = f"{add_prefix}{new_key}"
            
        new_state_dict[new_key] = value
        
    return new_state_dict
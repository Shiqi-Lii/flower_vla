import torch 
import threading
import torch.nn as nn

class TensorCache:
    """Thread-safe tensor cache that properly handles gradients"""
    def __init__(self, device):
        self.device = device
        self._cache = {}
        self._lock = threading.Lock()

    def get_tensor(self, key: str, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get a tensor from cache or create a new one, properly handling gradients"""
        cache_key = (key, shape, dtype)
        
        with self._lock:
            cached_tensor = self._cache.get(cache_key)
            
            if cached_tensor is None or cached_tensor.shape != shape:
                # Create new tensor with proper gradient handling
                new_tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                self._cache[cache_key] = new_tensor
                return new_tensor
            
            # Return a fresh view of the cached tensor to maintain gradient flow
            return cached_tensor.new_zeros(shape, dtype=dtype, device=self.device)

    def clear(self):
        """Clear the cache"""
        with self._lock:
            self._cache.clear()


class CachedMSELoss(nn.Module):
    """Cached version of MSE loss for memory efficiency"""
    def __init__(self, tensor_cache):
        super().__init__()
        self.tensor_cache = tensor_cache
    
    def forward(self, pred, target):
        diff_tensor = self.tensor_cache.get_tensor(
            'mse_diff',
            shape=pred.shape,
            dtype=pred.dtype
        )
        
        # Compute difference
        diff_tensor.copy_(pred)
        diff_tensor.sub_(target)
        
        # Square and mean
        diff_tensor.pow_(2)
        return diff_tensor.mean()
import os
import torch
from safetensors.torch import load_file, save_file
from pathlib import Path


# copied from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
class ExponentialMovingAverage:
  """
  Maintains (exponential) moving average of a set of parameters.
  """

  def __init__(self, parameters, decay, device: str = 'cuda', use_num_updates=True):
    """
    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the result of
        `model.parameters()`.
      decay: The exponential decay.
      use_num_updates: Whether to use number of updates when computing
        averages.
    """
    if decay < 0.0 or decay > 1.0:
      raise ValueError('Decay must be between 0 and 1')
    self.decay = decay
    self._device = device
    self.num_updates = 0 if use_num_updates else None
    self.shadow_params = [p.clone().detach()
                          for p in parameters if p.requires_grad]
    # ensure that all the weights are on the correct device
    print(f"Moving average device: {self._device}")
    self.shadow_params = [p.to(self._device) for p in self.shadow_params]
    self.collected_params = []
    
    self.steps = 0


  def update(self, parameters):
    """
    Update currently maintained parameters.
    Call this every time the parameters are updated, such as the result of
    the `optimizer.step()` call.
    Args:
      parameters: Iterable of `torch.nn.Parameter`; usually the same set of
        parameters used to initialize this object.
    """
    decay = self.decay
    if self.num_updates is not None:
        self.num_updates += 1
        decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
    one_minus_decay = 1.0 - decay
    
    with torch.no_grad():
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            # Ensure both tensors are on the same device before operation
            param_on_device = param.to(self._device)
            s_param.sub_(one_minus_decay * (s_param - param_on_device))


  def copy_to(self, parameters):
    """
    Copy current parameters into given collection of parameters.
    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored moving averages.
    """
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        param.data.copy_(s_param.data)

  def store(self, parameters):
    """
    Save the current parameters for restoring later.
    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        temporarily stored.
    """
    self.collected_params = [param.clone() for param in parameters]

  def restore(self, parameters):
    """
    Restore the parameters stored with the `store` method.
    Useful to validate the model with EMA parameters without affecting the
    original optimization process. Store the parameters before the
    `copy_to` method. After validation (or model saving), use this to
    restore the former parameters.
    Args:
      parameters: Iterable of `torch.nn.Parameter`; the parameters to be
        updated with the stored parameters.
    """
    for c_param, param in zip(self.collected_params, parameters):
      param.data.copy_(c_param.data)

  def state_dict(self):
    return dict(decay=self.decay, num_updates=self.num_updates,
                shadow_params=self.shadow_params)
  
  def load_shadow_params(self, parameters):
    parameters = [p for p in parameters if p.requires_grad]
    for s_param, param in zip(self.shadow_params, parameters):
      if param.requires_grad:
        s_param.data.copy_(param.data)

  def load_state_dict(self, state_dict):
    self.decay = state_dict['decay']
    self.num_updates = state_dict['num_updates']
    self.shadow_params = state_dict['shadow_params']
    # ensure that all the weights are on the correct device
    self.shadow_params = [p.to(self._device) for p in self.shadow_params]
  
  

class EMAWarmup:
    """Implements an EMA warmup using an inverse decay schedule.
    If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
        max_value (float): The maximum EMA decay rate. Default: 1.
        start_at (int): The epoch to start averaging at. Default: 0.
        last_epoch (int): The index of last epoch. Default: 0.
    """

    def __init__(self, inv_gamma=1., power=1., min_value=0., max_value=1., start_at=0,
                 last_epoch=0):
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.start_at = start_at
        self.last_epoch = last_epoch

    def get_value(self):
        """Gets the current EMA decay rate."""
        epoch = max(0, self.last_epoch - self.start_at)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return 0. if epoch < 0 else min(self.max_value, max(self.min_value, value))

    def step(self):
        """Updates the step count."""
        self.last_epoch += 1


def convert_weights(src_path: str, dst_path: str, checkpoint_steps: int = 255000):
    """
    Convert weights from source format to cleaned format, including EMA weights.
    
    Args:
        src_path: Path to source checkpoint directory containing checkpoints
        dst_path: Path where cleaned weights will be saved
        checkpoint_steps: Number of training steps for the checkpoint to convert (default: 255000)
    """
    # Setup paths
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Find checkpoint
    checkpoint_dir = src_path / f"checkpoint_{checkpoint_steps}"
    safetensors_path = checkpoint_dir / "model.safetensors"
    ema_path = checkpoint_dir / "custom_checkpoint_0.pkl"
    
    if not safetensors_path.exists():
        raise FileNotFoundError(f"No safetensors file found at {safetensors_path}")
    if not ema_path.exists():
        raise FileNotFoundError(f"No EMA weights found at {ema_path}")

    print(f"Loading model weights from {safetensors_path}")
    print(f"Loading EMA weights from {ema_path}")
    
    # Load weights
    state_dict = load_file(safetensors_path)
    ema_state = torch.load(ema_path, map_location='cpu', weights_only=True)
    
    print("\nEMA state structure:")
    print(f"Type: {type(ema_state)}")
    if isinstance(ema_state, dict):
        print("EMA state keys:", ema_state.keys())
    
    # Clean main weights - remove 'agent.' prefix
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('agent.'):
            new_key = key[6:]  # Remove 'agent.' prefix
        else:
            new_key = key
        cleaned_state_dict[new_key] = value

    # Clean EMA state - handle shadow params list and keep original structure
    cleaned_ema_state = {
        'decay': ema_state['decay'],
        'num_updates': ema_state['num_updates'],
        'shadow_params': []
    }
    
    # For each parameter in shadow_params, keep it as is since it's already a tensor
    if isinstance(ema_state['shadow_params'], list):
        cleaned_ema_state['shadow_params'] = ema_state['shadow_params']
    
    # Save cleaned model weights
    cleaned_safetensors = dst_path / "model_cleaned.safetensors"
    cleaned_pytorch = dst_path / "model_cleaned.pt"
    cleaned_ema = dst_path / "ema_cleaned.pkl"
    
    print(f"\nSaving cleaned weights to:")
    print(f"Model Safetensors: {cleaned_safetensors}")
    print(f"Model PyTorch: {cleaned_pytorch}")
    print(f"EMA weights: {cleaned_ema}")
    
    # Save in all formats
    save_file(cleaned_state_dict, cleaned_safetensors)
    
    # Save pytorch format with both model and EMA weights
    torch.save({
        'state_dict': cleaned_state_dict,
        'ema_state': cleaned_ema_state,
        'format_version': '1.0',
        'description': 'Cleaned MoDE weights with agent. prefix removed, including EMA weights'
    }, cleaned_pytorch)
    
    # Save separate EMA file with original structure
    torch.save(cleaned_ema_state, cleaned_ema)
    
    # Print statistics
    print(f"\nConversion complete!")
    print(f"Total model parameters: {len(cleaned_state_dict)}")
    print(f"Total EMA parameters: {len(cleaned_ema_state['shadow_params'])}")
    
    if len(cleaned_state_dict) > 0:
        print("\nExample model keys:")
        for key in list(cleaned_state_dict.keys())[:5]:
            print(f" - {key}: {cleaned_state_dict[key].shape}")
            
    if len(cleaned_ema_state['shadow_params']) > 0:
        print("\nFirst few EMA parameter shapes:")
        for i, param in enumerate(cleaned_ema_state['shadow_params'][:5]):
            print(f" - Parameter {i}: {param.shape}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert MoDE weights to cleaned format, including EMA weights')
    parser.add_argument('--src_path', type=str, required=True,
                        help='Path to source checkpoint directory')
    parser.add_argument('--dst_path', type=str, required=True,
                        help='Path where cleaned weights will be saved')
    parser.add_argument('--checkpoint_steps', type=int, default=300000,
                        help='Number of training steps for the checkpoint to convert (default: 300000)')
    
    args = parser.parse_args()
    
    try:
        convert_weights(args.src_path, args.dst_path, args.checkpoint_steps)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
import copy
import contextlib
from typing import Any, Dict, Iterable, Optional, Union
import sys

import logging
import importlib.util
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
# import deprecate


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
import transformers

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


_transformers_available = importlib.util.find_spec("transformers") is not None
try:
    _transformers_version = importlib_metadata.version("transformers")
    logger.debug(f"Successfully imported transformers version {_transformers_version}")
except importlib_metadata.PackageNotFoundError:
    _transformers_available = False



def is_transformers_available():
    return _transformers_available

def is_deepspeed_zero3_enabled():
    """Safer check for DeepSpeed ZeRO-3 status."""
    try:
        import transformers
        # Try multiple possible paths
        if hasattr(transformers, 'deepspeed') and hasattr(transformers.deepspeed, 'is_deepspeed_zero3_enabled'):
            return transformers.deepspeed.is_deepspeed_zero3_enabled()
        if hasattr(transformers, 'integrations') and hasattr(transformers.integrations, 'deepspeed'):
            return transformers.integrations.deepspeed.is_deepspeed_zero3_enabled()
        # If we can't find the function, assume it's not enabled
        return False
    except ImportError:
        return False

# Then use it like:
if is_transformers_available():
    if is_deepspeed_zero3_enabled():
        import deepspeed


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        foreach: bool = False,
        model_cls: Optional[Any] = None,
        model_config: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            foreach (bool): Use torch._foreach functions for updating shadow parameters. Should be faster.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """

        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

            # set use_ema_warmup to True if a torch.nn.Module is passed for backwards compatibility
            use_ema_warmup = True

        if kwargs.get("max_value", None) is not None:
            deprecation_message = "The `max_value` argument is deprecated. Please use `decay` instead."
            deprecate("max_value", "1.0.0", deprecation_message, standard_warn=False)
            decay = kwargs["max_value"]

        if kwargs.get("min_value", None) is not None:
            deprecation_message = "The `min_value` argument is deprecated. Please use `min_decay` instead."
            deprecate("min_value", "1.0.0", deprecation_message, standard_warn=False)
            min_decay = kwargs["min_value"]

        parameters = list(parameters)

        # Clone initial params to shadow.
        self.shadow_params = [p.clone().detach() for p in parameters]

        if kwargs.get("device", None) is not None:
            deprecation_message = "The `device` argument is deprecated. Please use `to` instead."
            deprecate("device", "1.0.0", deprecation_message, standard_warn=False)
            self.to(device=kwargs["device"])

        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = 0
        self.cur_decay_value = None  # set in `step()`
        self.foreach = foreach

        # Stash the “original” model class if you want to do .save_pretrained()
        self.model_cls = model_cls
        self.model_config = model_config

        self.temp_stored_params: Optional[List[torch.Tensor]] = None

    @classmethod
    def from_pretrained(cls, path, model_cls, foreach=False) -> "EMAModel":
        _, ema_kwargs = model_cls.from_config(path, return_unused_kwargs=True)
        model = model_cls.from_pretrained(path)

        ema_model = cls(model.parameters(), model_cls=model_cls, model_config=model.config, foreach=foreach)

        ema_model.load_state_dict(ema_kwargs)
        return ema_model

    def save_pretrained(self, path):
        if self.model_cls is None:
            raise ValueError("`save_pretrained` can only be used if `model_cls` was defined at __init__.")

        if self.model_config is None:
            raise ValueError("`save_pretrained` can only be used if `model_config` was defined at __init__.")

        model = self.model_cls.from_config(self.model_config)
        state_dict = self.state_dict()
        state_dict.pop("shadow_params", None)

        model.register_to_config(**state_dict)
        self.copy_to(model.parameters())
        model.save_pretrained(path)

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        if self.use_ema_warmup:
            cur_decay_value = 1 - (1 + step / self.inv_gamma) ** -self.power
        else:
            cur_decay_value = (1 + step) / (10 + step)

        cur_decay_value = min(cur_decay_value, self.decay)
        # make sure decay is not smaller than min_decay
        cur_decay_value = max(cur_decay_value, self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter]):
        if isinstance(parameters, torch.nn.Module):
            deprecation_message = (
                "Passing a `torch.nn.Module` to `ExponentialMovingAverage.step` is deprecated. "
                "Please pass the parameters of the module instead."
            )
            deprecate(
                "passing a `torch.nn.Module` to `ExponentialMovingAverage.step`",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            parameters = parameters.parameters()

        parameters = list(parameters)

        parameters = list(parameters)
        shadow_devices = {i: p.device for i, p in enumerate(self.shadow_params)}
        param_devices = {i: p.device for i, p in enumerate(parameters)}
        
        # Find mismatches
        mismatches = [(i, param_devices[i], shadow_devices[i]) 
                     for i in shadow_devices 
                     if param_devices[i] != shadow_devices[i]]
        
        if mismatches:
            raise RuntimeError(
                "Device mismatch in EMA:\n" + 
                "\n".join(f"Param {i}: {p_dev} vs Shadow: {s_dev}" 
                         for i, p_dev, s_dev in mismatches)
            )
            
            
        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        context_manager = contextlib.nullcontext()

        if self.foreach:
            if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
                context_manager = deepspeed.zero.GatheredParameters(parameters, modifier_rank=None)

            with context_manager:
                params_grad = [param for param in parameters if param.requires_grad]
                s_params_grad = [
                    s_param for s_param, param in zip(self.shadow_params, parameters) if param.requires_grad
                ]

                if len(params_grad) < len(parameters):
                    torch._foreach_copy_(
                        [s_param for s_param, param in zip(self.shadow_params, parameters) if not param.requires_grad],
                        [param for param in parameters if not param.requires_grad],
                        non_blocking=True,
                    )

                torch._foreach_sub_(
                    s_params_grad, torch._foreach_sub(s_params_grad, params_grad), alpha=one_minus_decay
                )

        else:
            for s_param, param in zip(self.shadow_params, parameters):
                if is_transformers_available() and transformers.integrations.deepspeed.is_deepspeed_zero3_enabled():
                    context_manager = deepspeed.zero.GatheredParameters(param, modifier_rank=None)

                with context_manager:
                    if param.requires_grad:
                        s_param.sub_(one_minus_decay * (s_param - param))
                    else:
                        s_param.copy_(param)

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            if s_param.device != param.device:
                # Move shadow to param's device or vice versa
                s_param_on_target = s_param.to(param.device)
                param.data.copy_(s_param_on_target.data)
            else:
                param.data.copy_(s_param.data)

    def pin_memory(self) -> None:
        r"""
        Move internal buffers of the ExponentialMovingAverage to pinned memory. Useful for non-blocking transfers for
        offloading EMA params to the host.
        """

        self.shadow_params = [p.pin_memory() for p in self.shadow_params]

    def to(self, device=None, dtype=None, non_blocking=False) -> None:
        r"""
        Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype, non_blocking=non_blocking)
            if p.is_floating_point()
            else p.to(device=device, non_blocking=non_blocking)
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        """
        Return state for checkpointing. Make sure shadow_params is included.
        """
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": self.shadow_params,  # store them
        }

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Saves the current parameters for restoring later.

        Args:
            parameters: Iterable of `torch.nn.Parameter`. The parameters to be temporarily stored.
        """
        parameters = list(parameters)
        self.temp_stored_params = []
        for param in parameters:
            # Move to CPU if you want to reduce GPU usage
            # Or keep on param.device if you prefer
            cpu_copy = param.detach().cpu().clone()
            self.temp_stored_params.append(cpu_copy)

    def restore(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        r"""
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters
        without: affecting the original optimization process. Store the parameters before the `copy_to()` method. After
        validation (or model saving), use this to restore the former parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored parameters. If `None`, the parameters with which this
                `ExponentialMovingAverage` was initialized will be used.
        """

        if self.temp_stored_params is None:
            raise RuntimeError("No stored parameters to restore.")
        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.data.to(param.device))
        self.temp_stored_params = None


    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Loads the ExponentialMovingAverage state. This method is used by accelerate during checkpointing to save the
        ema state dict.

        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict.get("decay", self.decay)
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.min_decay = state_dict.get("min_decay", self.min_decay)
        if not isinstance(self.min_decay, float):
            raise ValueError("Invalid min_decay")

        self.optimization_step = state_dict.get("optimization_step", self.optimization_step)
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
        if not isinstance(self.update_after_step, int):
            raise ValueError("Invalid update_after_step")

        self.use_ema_warmup = state_dict.get("use_ema_warmup", self.use_ema_warmup)
        if not isinstance(self.use_ema_warmup, bool):
            raise ValueError("Invalid use_ema_warmup")

        self.inv_gamma = state_dict.get("inv_gamma", self.inv_gamma)
        if not isinstance(self.inv_gamma, (float, int)):
            raise ValueError("Invalid inv_gamma")

        self.power = state_dict.get("power", self.power)
        if not isinstance(self.power, (float, int)):
            raise ValueError("Invalid power")

        shadow_params = state_dict.get("shadow_params", None)
        if shadow_params is not None:
            self.shadow_params = shadow_params
            if not isinstance(self.shadow_params, list):
                raise ValueError("shadow_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
                raise ValueError("shadow_params must all be Tensors")
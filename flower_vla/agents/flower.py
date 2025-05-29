import logging
import os
from typing import Optional, Dict, Tuple, Union, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import wandb
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoModelForCausalLM
from timm.layers.mlp import Mlp
from torchdiffeq import odeint

# Import helper modules from your codebase.
from flower_vla.agents.utils.action_index import ActionIndex
from flower_vla.agents.networks.flower_transformers import (
    TimestepEmbedder,
    SharedAdaLNController,
    RmsNorm,
    FreqEmbedder,
    ActionSpaceEmbedderParameter,
    ZeroEncoder,
)
from flower_vla.agents.networks.flow_transformer import FlowBlock, stateless_norm
from flower_vla.dataset.utils.act_seq_mapping import DATASET_ACT_SEQ_MAP

logger = logging.getLogger(__name__)


class FlowerVLA(nn.Module):
    def __init__(
        self,
        # Core configuration
        device,
        process_id: int,
        accelerator,
        # Modality configuration 
        target_modality: str,
        obs_modalities: str,
        goal_modalities: str,
        img_modalities: List[str],
        lang_modalities: List[str],
        # Vision-Language Model configuration
        vlm_path: str = "microsoft/Florence-2-large",
        freeze_florence: bool = False,
        freeze_vision_tower: bool = False,
        freeze_embeddings_only: bool = False,  # New parameter
        vlm_prompt_style: str = 'default',
        token_dropout: float = 0.2,
        # Action and observation dimensions
        lowdim_obs_dim: int = 7,
        action_dim: int = 7,
        act_window_size: int = 10,
        multistep: int = 10,
        num_sampling_steps: int = 5,
        # Model architecture flags
        use_second_view: bool = False,
        second_view_key: str = 'image_wrist',
        action_type_adaln: bool = False,
        use_causal_attention: bool = True,
        use_cross_attn: bool = True,
        use_adaln_cond: bool = False,
        use_readout_token: bool = False,
        use_proprio: bool = False,
        return_act_chunk: bool = False,
        # DiT configuration 
        sampling_type: str = 'ln',
        dit_dim: int = 512,
        n_heads: int = 16,
        n_layers: int = 12,
        # Dropout rates
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_pdrop: float = 0.1,
        # RoPE configuration
        use_rope: bool = False,
        use_nope: bool = False,
        query_seq_len: int = 128,
        rope_theta: float = 32.0,
    ):
        """
        Initializes the FlowerVLA agent that combines a pretrained vision–language model
        with a flow-based DiT architecture for learning a generalist policy.

        Args:
            device: Target device for computations.
            process_id: Process ID (for distributed setups).
            accelerator: An accelerator instance.
            target_modality, obs_modalities, goal_modalities, img_modalities, lang_modalities:
                Modality configurations.
            vlm_path: Path or identifier for the pretrained VLM.
            freeze_florence, freeze_vision_tower: Flags for freezing parts of the VLM.
            vlm_prompt_style: Prompt style configuration.
            token_dropout: Dropout probability for VLM tokens.
            lowdim_obs_dim, action_dim: Dimensions for observations and actions.
            act_window_size, multistep, num_sampling_steps: Parameters for action sequence prediction.
            use_second_view: Whether to use an additional image view.
            second_view_key: Key for the second view.
            action_type_adaln, use_causal_attention, use_cross_attn, use_adaln_cond, use_readout_token:
                Flags controlling attention and conditioning.
            use_proprio: Whether to incorporate proprioception.
            return_act_chunk: Whether to return the whole action chunk.
            sampling_type: Sampling type (e.g., 'ln', 'pi_zero', etc.).
            dit_dim, n_heads, n_layers: DiT architecture parameters.
            attn_pdrop, resid_pdrop, mlp_pdrop: Dropout rates.
            use_rope, use_nope, query_seq_len, rope_theta: Positional encoding parameters.
        """
        super().__init__()
        self.device = device
        self.process_id = process_id
        self.accelerator = accelerator

        # Initialize configuration groups.
        self._init_modalities(target_modality, obs_modalities, goal_modalities, img_modalities, lang_modalities)
        self._init_dimensions(dit_dim, n_heads, lowdim_obs_dim, action_dim, act_window_size, multistep, num_sampling_steps)
        self._init_flags(use_second_view, use_causal_attention, use_cross_attn, use_adaln_cond,
                         use_readout_token, use_rope, use_nope, vlm_prompt_style, token_dropout,
                         action_type_adaln, sampling_type, use_proprio, return_act_chunk, second_view_key)
        logger.info("Configuration (modalities, dimensions, flags) initialized.")

        # Initialize action space index.
        self.action_space_index = ActionIndex()

        # Setup model components.
        self._setup_vlm(vlm_path, freeze_vision_tower, freeze_florence, freeze_embeddings_only)
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim
        self.use_dopri5 = False
        self._setup_dit_components(
            dit_dim, n_heads, n_layers, action_dim, act_window_size, hidden_dim,
            attn_pdrop, resid_pdrop, mlp_pdrop, use_cross_attn,
            use_rope, use_nope, query_seq_len, rope_theta,
        )
        logger.info("VLM and DiT components set up.")

        # Initialize rollout state.
        self.rollout_step_counter = 0
        self.pred_action_seq = None

        # Ensure that all parameters and buffers are on the correct device.
        self.ensure_device_consistency()

    # === Initialization Helpers ===
    def _init_modalities(self, target_modality: str, obs_modalities: str, goal_modalities: str,
                           img_modalities: List[str], lang_modalities: List[str]) -> None:
        """Initializes modality-related attributes."""
        self.target_modality = target_modality
        self.obs_modalities = obs_modalities
        self.goal_modalities = goal_modalities
        self.img_modalities = img_modalities
        self.lang_modalities = lang_modalities

    def _init_dimensions(self, dit_dim: int, n_heads: int, lowdim_obs_dim: int, action_dim: int,
                           act_window_size: int, multistep: int, num_sampling_steps: int) -> None:
        """Initializes dimension-related attributes and checks consistency."""
        if dit_dim % n_heads != 0:
            raise ValueError(f"dit_dim ({dit_dim}) must be divisible by n_heads ({n_heads})")
        self.lowdim_obs_dim = lowdim_obs_dim
        self.action_dim = action_dim
        self.act_window_size = act_window_size
        self.multistep = multistep
        self.num_sampling_steps = num_sampling_steps
        self.dit_dim = dit_dim

    def _init_flags(self, use_second_view: bool, use_causal_attention: bool, use_cross_attn: bool,
                    use_adaln_cond: bool, use_readout_token: bool, use_rope: bool, use_nope: bool,
                    vlm_prompt_style: str, token_dropout: float, action_type_adaln: bool,
                    sampling_type: str, use_proprio: bool, return_act_chunk: bool, second_view_key: str) -> None:
        """Initializes boolean flags and related parameters."""
        if vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")
        if sampling_type not in ['ln', 'pi_zero', 'loglogistic', 'uniform', 'stratified']:
            raise ValueError(f"Invalid sampling type: {sampling_type}")
        self.use_second_view = use_second_view
        self.use_causal_attention = use_causal_attention
        self.use_cross_attn = use_cross_attn
        self.use_adaln_cond = use_adaln_cond
        self.use_readout_token = use_readout_token
        self.use_rope = use_rope
        self.use_nope = use_nope
        self.use_proprio = use_proprio
        self.return_act_chunk = return_act_chunk
        self.vlm_prompt_style = vlm_prompt_style
        self.token_dropout = token_dropout
        self.action_type_adaln = action_type_adaln
        self.sampling_type = sampling_type
        self.second_view_key = second_view_key

    def _setup_vlm(self, vlm_path: str, freeze_vision_tower: bool, freeze_florence: bool, freeze_embeddings_only: bool) -> None:
        """
        Loads the pretrained VLM, sets up the processor/tokenizer, adds a prompt token,
        and optionally freezes parameters.
        """
        logger.info(f"Loading VLM from {vlm_path}")
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, trust_remote_code=True)
        # Set all parameters to require gradients by default.
        # Handle freezing according to config                
        if freeze_florence:
            for name, param in self.vlm.named_parameters():
                param.requires_grad = False
            self.train_vlm = False
        elif freeze_embeddings_only:
            # Keep most parameters trainable
            self.train_vlm = True
            # Freeze only the embedding layers
            embedding_layer = self.vlm.get_input_embeddings()
            for param in embedding_layer.parameters():
                param.requires_grad = False
            # Also freeze shared embeddings if they exist
            if hasattr(self.vlm.language_model, 'shared'):
                for param in self.vlm.language_model.shared.parameters():
                    param.requires_grad = False
        else:
            self.train_vlm = True

        if not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.prompt_embeds = self._create_prompt_embed("<Flow>").to(self.device)
        # Remove unused language model components.
        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

    def _create_prompt_embed(self, prompt_text: str) -> nn.Parameter:
        """
        Creates a prompt embedding. Adds the prompt token to the tokenizer
        and returns its embedding (frozen).
        """
        self.tokenizer.add_special_tokens({'additional_special_tokens': [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)),
            requires_grad=False
        )
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def _setup_dit_components(
        self, dit_dim: int, n_heads: int, n_layers: int, action_dim: int, act_window_size: int,
        hidden_dim: int, attn_pdrop: float, resid_pdrop: float, mlp_pdrop: float, use_cross_attn: bool,
        use_rope: bool, use_nope: bool, query_seq_len: int, rope_theta: float
    ) -> None:
        """
        Sets up the action-specific encoders/decoders, shared conditioning components,
        and constructs the DiT (flow-based transformer) blocks.
        """
        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
        self.adaln = nn.ModuleDict() if self.action_type_adaln else None

        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)
            self.action_encoders[action_name] = Mlp(in_features=input_dim, hidden_features=dit_dim, out_features=dit_dim, bias=True)
            self.action_decoders[action_name] = nn.Linear(dit_dim, input_dim).to(self.device)
            if self.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn)
            if self.use_proprio:
                self.proprio_encoders[action_name] = (
                    Mlp(input_dim, dit_dim, out_features=dit_dim, drop=0.2).to(self.device)
                    if action_name == 'bimanual_nav'
                    else ZeroEncoder(self.dit_dim, device=self.device)
                )

        if not self.action_type_adaln:
            self.adaln = SharedAdaLNController(dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn)

        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(dit_dim, max_actions=len(self.action_space_index.action_spaces))
        if not use_rope and not use_nope:
            self.positional_encoding = nn.Parameter(torch.randn(1, act_window_size, dit_dim) * 0.1)
        self.dit = nn.ModuleList([
            FlowBlock(
                dit_dim, n_heads,
                attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, mlp_pdrop=mlp_pdrop,
                use_cross_attn=use_cross_attn, use_rope=use_rope,
                query_seq_len=query_seq_len, rope_theta=rope_theta,
            ) for _ in range(n_layers)
        ])

    # === Device Consistency Methods ===
    def ensure_device_consistency(self) -> None:
        """Moves the entire model (and buffers) to the designated device."""
        self.to(self.device)
        self.vlm.to(self.device)
        if not self.use_rope and hasattr(self, 'positional_encoding'):
            self.positional_encoding = self.positional_encoding.to(self.device)
        if self.use_readout_token and hasattr(self, 'register_token'):
            self.register_token = self.register_token.to(self.device)
        self._verify_device_consistency()

    def _verify_device_consistency(self) -> None:
        """Verifies that all parameters and buffers are on the expected device."""
        expected = self.device
        inconsistent = []
        for name, param in self.named_parameters():
            if param.device != expected:
                inconsistent.append(f"{name}: {param.device} (expected {expected})")
        for name, buf in self.named_buffers():
            if buf.device != expected:
                inconsistent.append(f"{name} (buffer): {buf.device} (expected {expected})")
        if inconsistent:
            logger.warning("Device consistency issues: " + "; ".join(inconsistent))

    # === Encoding and Decoding Methods ===
    def encode_observations(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Encodes primary (and optional second view) image observations and text goals.
        Returns a dictionary with:
            - 'features': Encoder outputs.
            - 'frequency_embeds': Frequency embeddings.
            - 'action_space_embeds': Action space embeddings.
            - 'action_type': Action type indices.
            - 'proprio': Proprioception data (if available).
            - 'attention_mask': Attention mask.
        """
        device = self.device
        default_dtype = next(self.parameters()).dtype
        image_tensor = batch[self.obs_modalities]['image_primary']
        B, T, C, H, W = image_tensor.shape
        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_dtype)
        )
        image_features = image_features.view(B, T * image_features.shape[1], -1)
        if self.use_second_view and self.second_view_key in batch[self.obs_modalities]:
            image2_tensor = batch[self.obs_modalities][self.second_view_key]
            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_dtype)
            )
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)
        text_embeds = self.vlm.get_input_embeddings()(
            batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'].to(device)
        ).to(device).squeeze(1)
        task_prompt = self.prompt_embeds.expand(B, -1, -1)
        merged_embeds = torch.cat([task_prompt.to(image_features.device), image_features, text_embeds.to(image_features.device)], dim=1)

        # get attention mask from txt
        lang_attention_mask = batch[self.goal_modalities][self.lang_modalities[0]]['attention_mask'].to(device).squeeze(1)
        # define attention mask for image
        vis_attention_mask = torch.ones(image_features.shape[:2], device=image_features.device)
        prompt_mask = torch.zeros(B, 1, dtype=torch.bool, device=image_features.device)
        attention_mask = torch.cat([prompt_mask, vis_attention_mask, lang_attention_mask], dim=1)

        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds, 
            attention_mask=attention_mask,
        ).last_hidden_state

        features = self.vlm_token_dropout(features)
        return {
            'features': features,
            'frequency_embeds': self.frequency_embedder(batch[self.goal_modalities]['frequency'].to(device).to(default_dtype)),
            'action_space_embeds': self.action_space_embedder(batch[self.goal_modalities]['action_space_index'].to(device)),
            'action_type': batch[self.goal_modalities]['action_space_index'],
            'proprio': batch[self.obs_modalities]['proprio'].to(device).to(default_dtype) if self.use_proprio and 'proprio' in batch[self.obs_modalities] else None,
            'attention_mask': attention_mask,
        }

    def encode_actions(self, z: torch.Tensor, action_type: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes actions for each sample based on its action type.
        Returns:
            - Encoded actions (latent representations).
            - A valid dimensions mask.
        """
        default_dtype = next(self.parameters()).dtype
        action_type = action_type.to(self.device)
        B = z.shape[0]
        encoded = torch.zeros(B, z.shape[1], self.dit_dim, device=self.device, dtype=default_dtype)
        valid_dims = torch.zeros_like(z, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                valid_dims[mask, :, :adim] = 1
                encoded[mask] = self.action_encoders[action_name](z[mask, :, :adim])
        return encoded, valid_dims

    def decode_actions(self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent representations into actual actions.
        Only the dimensions corresponding to valid action spaces are active.
        """
        default_dtype = next(self.parameters()).dtype
        B = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(B, z.shape[1], max_action_dim, device=self.device, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                pred = self.action_decoders[action_name](z[mask])
                decoded[mask, :, :adim] = pred[..., :adim] * valid_dims[mask, :, :adim]
        return decoded

    # === Loss Functions ===
    def rf_loss(self, cond: dict, actions: torch.Tensor, dataset_idx: Any = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes the rectified flow loss.
        Interpolates between actions and noise, then computes MSE only over valid dimensions.
        """
        default_dtype = next(self.parameters()).dtype
        action_type = cond['action_type']
        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(default_dtype)

        # Sample time t based on the chosen distribution.
        if self.sampling_type == "pi_zero":
            alpha, beta = 1.5, 1.0
            t = torch.distributions.Beta(alpha, beta).sample((b,)).to(device).clamp(max=0.999)
        elif self.sampling_type == "ln":
            t = torch.sigmoid(torch.randn((b,), device=device)).clamp(max=0.999).to(default_dtype)
        elif self.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - eps)
            t = t.to(default_dtype)
        else:
            raise NotImplementedError(f"Sampling type {self.sampling_type} not implemented")
        texp = t.view([b] + [1] * (actions.dim() - 1))
        z1 = torch.zeros_like(actions)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                noise_slice = torch.randn((mask.sum(), actions.size(1), adim), dtype=actions.dtype, device=actions.device)
                z1[mask, :, :adim] = noise_slice
        zt = (1 - texp) * actions + texp * z1
        vtheta = self.dit_forward(zt, t, cond)
        valid_mask = torch.zeros_like(actions, dtype=torch.bool)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                mask_expanded = mask.view(-1, 1, 1).expand(-1, actions.size(1), adim).to(device)
                valid_mask[mask, :, :adim] = mask_expanded[mask]
        diff = (z1 - actions) - vtheta
        valid_diff = diff[valid_mask]
        loss = (valid_diff ** 2).mean()
        losses_dict = {
            "diff_min": valid_diff.min().item(),
            "diff_max": valid_diff.max().item(),
            "diff_mean": valid_diff.mean().item(),
            "loss": loss.item(),
        }
        if hasattr(self, 'accelerator') and self.accelerator is not None and wandb.run is not None:
            if self.accelerator.is_main_process:
                wandb.log(losses_dict)
        return loss, losses_dict

    # === Sampling Methods ===
    def sample_actions(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False) -> torch.Tensor:
        """
        Samples actions from the DiT model.
        Chooses between an adaptive ODE solver and fixed-step Euler integration.
        """
        steps = self.num_sampling_steps if inference else 5
        b = z.size(0)
        action_type = cond['action_type']
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                adim = self.action_space_index.get_action_dim(action_idx)
                z[mask, :, adim:] = 0.0
        if hasattr(self, 'use_dopri5') and self.use_dopri5:
            return self._sample_with_adaptive_solver(z, cond)
        else:
            return self._sample_with_fixed_steps(z, cond, inference)

    def _sample_with_adaptive_solver(self, z: torch.Tensor, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Samples actions using an adaptive ODE solver (dopri5).
        """
        device = z.device
        action_type = cond['action_type']
        def ode_func(t, z):
            b = z.size(0)
            t_tensor = t * torch.ones(b, device=device)
            with torch.no_grad():
                z = z.clone()
                for action_name, action_idx in self.action_space_index.action_spaces.items():
                    mask = (action_type == action_idx)
                    if mask.any():
                        adim = self.action_space_index.get_action_dim(action_idx)
                        z[mask, :, adim:] = 0.0
                v = self.dit_forward(z, t_tensor, cond)
                for action_name, action_idx in self.action_space_index.action_spaces.items():
                    mask = (action_type == action_idx)
                    if mask.any():
                        adim = self.action_space_index.get_action_dim(action_idx)
                        v[mask, :, adim:] = 0.0
            return v
        t_span = torch.tensor([1.0, 0.0], device=device)
        z = odeint(
            ode_func, z, t_span, method='dopri5',
            rtol=1e-4, atol=1e-4,
            options={'max_num_steps': max(self.num_sampling_steps * 2, 1000), 'min_step': 1.0 / self.num_sampling_steps}
        )[-1]
        return z.clamp(-1, 1)

    def _sample_with_fixed_steps(self, z: torch.Tensor, cond: Dict[str, torch.Tensor], inference: bool = False) -> torch.Tensor:
        """
        Samples actions using fixed-step Euler integration.
        """
        steps = self.num_sampling_steps if inference else 5
        b = z.size(0)
        device = z.device
        action_type = cond['action_type']
        dt = 1.0 / steps
        dt_tensor = torch.tensor([dt] * b, device=device).view([b] + [1] * (z.dim() - 1))
        for i in range(steps, 0, -1):
            t_val = i / steps
            t_tensor = torch.full((b,), t_val, device=device)
            vc = self.dit_forward(z, t_tensor, cond)
            z = z - dt_tensor * vc
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                mask = (action_type == action_idx)
                if mask.any():
                    adim = self.action_space_index.get_action_dim(action_idx)
                    z[mask, :, adim:] = 0.0
        return z.clamp(-1, 1)

    # === Forward Pass and Rollout Methods ===
    def forward(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Inference forward pass.
        Given observation and goal dictionaries, it encodes them and samples an action sequence.
        """
        batch = {'observation': obs, 'task': goal}
        features = self.encode_observations(batch)
        noise = torch.randn(len(features['features']), self.act_window_size, self.action_dim,
                              device=features['features'].device)
        return self.sample_actions(noise, features, inference=True)

    def reset(self) -> None:
        """
        Resets the rollout state.
        """
        self.rollout_step_counter = 0
        self.pred_action_seq = None
        self.eval()

    def step(self, obs: Dict, goal: Dict) -> torch.Tensor:
        """
        Returns the current action (or full chunk) based on the rollout step and updates the state.
        """
        if self.rollout_step_counter % self.multistep == 0:
            self.pred_action_seq = self(obs, goal)
        if not self.return_act_chunk:
            current_action = self.pred_action_seq[0, self.rollout_step_counter]
            if len(current_action.shape) == 2:
                current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        else:
            current_action = self.pred_action_seq
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0
        return current_action

    # === Additional Utility Methods ===
    def training_step(self, batch: Dict[str, Dict]) -> torch.Tensor:
        """
        A single training step.
        Encodes observations, computes the rectified flow loss, and returns the loss tensor.
        """
        self.train()
        obs_features = self.encode_observations(batch)
        action_loss, losses_dict = self.rf_loss(
            obs_features, batch[self.target_modality], batch['task']['dataset_index']
        )
        # Store debugging losses if needed.
        self.losses_dict = losses_dict
        return action_loss

    def validation_step(self, batch: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        """
        A single evaluation step.
        Returns a dictionary containing loss and predictions.
        """
        self.eval()
        with torch.no_grad():
            target_actions = batch[self.target_modality]
            if len(target_actions.shape) == 4:
                target_actions = target_actions.squeeze(1)
            obs_features = self.encode_observations(batch)
            action_type = obs_features['action_type']
            noise_actions = torch.zeros_like(target_actions)
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                mask = (action_type == action_idx)
                if mask.any():
                    adim = self.action_space_index.get_action_dim(action_idx)
                    rand_slice = torch.randn((mask.sum(), target_actions.size(1), adim),
                                             device=self.device, dtype=target_actions.dtype)
                    noise_actions[mask, :, :adim] = rand_slice
            action_pred = self.sample_actions(noise_actions, obs_features, inference=True)
            losses_dict = {}
            total_loss = torch.tensor(0.0, device=self.device)
            num_action_types = 0
            for action_name, action_idx in self.action_space_index.action_spaces.items():
                mask = (action_type == action_idx)
                if mask.any():
                    adim = self.action_space_index.get_action_dim(action_idx)
                    space_loss = F.mse_loss(
                        action_pred[mask, :, :adim],
                        target_actions[mask, :, :adim],
                        reduction='mean'
                    )
                    losses_dict[f"val_loss_{action_name}"] = space_loss.item()
                    total_loss += space_loss
                    num_action_types += 1
            avg_loss = total_loss / num_action_types if num_action_types > 0 else torch.tensor(0.0, device=self.device)
            return {
                "loss": avg_loss,
                "losses": losses_dict,
                "predictions": action_pred,
                "targets": batch[self.target_modality],
                "dataset_index": batch['task'].get('dataset_index', torch.zeros(action_pred.shape[0], device=self.device))
            }

    def dit_forward(self, z: torch.Tensor, t: torch.Tensor, cond_dict: dict) -> torch.Tensor:
        """
        Forward pass through the DiT blocks.
        Encodes actions, adds positional information, and applies conditioning.
        """
        B, t_seq, d = z.shape
        default_dtype = next(self.parameters()).dtype
        cond = cond_dict['features'].to(default_dtype)
        frequency_embeds = cond_dict['frequency_embeds'].squeeze(1).to(default_dtype)
        action_type = cond_dict['action_type'].to(self.device)
        proprio = cond_dict['proprio'].to(default_dtype) if self.use_proprio else torch.zeros_like(frequency_embeds)
        # Encode proprioception based on action type.
        proprio_embeds = self.encode_proprio(proprio, action_type, frequency_embeds.shape).to(default_dtype)
        # Encode actions based on action type.
        z, valid_dims = self.encode_actions(z, action_type)
        # Add positional encoding if not using RoPE or NoPE.
        if not self.use_rope and not self.use_nope:
            z = z + self.positional_encoding
        t_emb = stateless_norm(self.t_embedder(t)) + stateless_norm(frequency_embeds) + stateless_norm(proprio_embeds)
        cond = self.cond_linear(self.cond_norm(cond))
        if self.use_adaln_cond:
            global_cond = (cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)) + t_emb
        else:
            global_cond = t_emb
        cx = z
        context = cond if self.use_cross_attn else None
        # Compute action-specific AdaLN modulation signals or use shared AdaLN.
        if not self.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self.action_specific_adaln(global_cond, action_type)
        cross_attn_mask = cond_dict['attention_mask']
        # Apply DiT blocks.
        for layer in self.dit:
            cx = layer(cx, global_cond, context=context, custom_attn_mask=None, custom_cross_attn_mask=cross_attn_mask, is_causal=True, global_adaln=global_adaln)
        
        # Decode actions back again
        return self.decode_actions(cx, action_type, valid_dims)

    def encode_proprio(self, proprio: torch.Tensor, action_type: torch.Tensor, output_shape) -> torch.Tensor:
        """
        Encodes proprioceptive data based on action type.
        Returns a tensor with shape [batch, dit_dim].
        """
        batch_size, _ = output_shape
        default_dtype = next(self.parameters()).dtype
        if not self.use_proprio:
            return torch.zeros(batch_size, self.dit_dim, device=self.device)
        encoded = torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=default_dtype)
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded[mask] = self.proprio_encoders[action_name](proprio[mask]).squeeze(1).to(default_dtype)
        return encoded

    def action_specific_adaln(self, global_cond: torch.Tensor, action_type: torch.Tensor) -> List[torch.Tensor]:
        """
        Computes action-specific AdaLN modulation signals.
        Returns a list of modulation tensors.
        """
        default_dtype = next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6
        mod_signals = [torch.zeros(batch_size, self.dit_dim, device=self.device, dtype=default_dtype)
                       for _ in range(num_chunks)]
        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond[mask])
                for i, signal in enumerate(action_mod):
                    mod_signals[i][mask] = signal
        return mod_signals

    # === Optimizer Configuration ===
    def configure_optimizers(self, optimizer_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Configures optimizer parameter groups for non-VLM parameters.
        Returns a list of parameter groups.
        """
        no_decay = ['bias', 'LayerNorm', 'layernorm', 'ln', 'norm']
        decay_group = []
        no_decay_group = []
        vlm_params = set(p for p in self.vlm.parameters())
        for name, param in self.named_parameters():
            if param.requires_grad and param.is_leaf and param not in vlm_params:
                if any(nd in name.lower() for nd in no_decay):
                    no_decay_group.append(param)
                else:
                    decay_group.append(param)
        optim_groups = [
            {"params": decay_group, "weight_decay": optimizer_config["transformer_weight_decay"]},
            {"params": no_decay_group, "weight_decay": 0.0}
        ]
        return optim_groups

    # === Checkpointing ===
    def save_checkpoint(self, path: str) -> None:
        """
        Saves model checkpoint (both weights for inference and full state for resuming training).
        """
        if not self.accelerator.is_main_process:
            return
        save_dict = {
            "model": self.accelerator.get_state_dict(self),
            "config": getattr(self, "config", None),
            "step": getattr(self, "_wandb_step", None)
        }
        self.accelerator.save(save_dict, path)
        logger.info(f"Checkpoint saved at {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Loads a model checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.accelerator.device)
        self.load_state_dict(checkpoint["model"])
        self._wandb_step = checkpoint.get("step", 0)
        logger.info(f"Checkpoint loaded from {path}")

    def print_encoded_texts(self, batch: Dict, device) -> None:
        """
        Prints original and encoded texts for debugging.
        """
        text_embeds = self.vlm.get_input_embeddings()(
            batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'].to(self.device)
        ).to(device).squeeze(1)
        input_ids = batch[self.goal_modalities][self.lang_modalities[0]]['input_ids'][0].squeeze(0).to(self.device)
        decoded_text = self.processor.tokenizer.decode(input_ids.cpu(), skip_special_tokens=False)
        print("Original text:", decoded_text)
        decoded_texts = self.processor.tokenizer.batch_decode(text_embeds.cpu(), skip_special_tokens=True)
        print("Encoded texts:")
        for i, text in enumerate(decoded_texts):
            print(f"Sequence {i+1}: {text}")

import logging
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from enum import IntEnum
from torchvision.transforms import Normalize
from torchvision.transforms.functional import convert_image_dtype
from accelerate import Accelerator

logger = logging.getLogger(__name__)

class Mode(IntEnum):
    "Current Mode the Agent is in"
    NONE = -1
    INFERENCE = 1
    TRAINING = 2
    EVALUATION = 3


class DDPAgentWrapper(nn.Module):

    def __init__(
        self,
        agent: DictConfig,
        obs_modalities: str,
        goal_modalities: str,
        img_modalities: dict,
        lang_modalities: dict,
        target_modality: dict,
        process_id: int,
        device: str,
        accelerator: Accelerator = None,
    ):
        super().__init__()
        # set device afterwards cuz of accelerate
        if "model" in agent:
            agent.model.device = str(device)
            if "inner_model" in agent.model:
                agent.model.inner_model.device = str(device)
        print("Instantiating Agent")
        print(agent)
        self.agent: nn.Module = hydra.utils.instantiate(agent, device=device, process_id=process_id, accelerator=accelerator)
        print("Agent Instantiated")
        self.obs_modalities = obs_modalities
        self.goal_modalities = goal_modalities
        self.img_modalities = img_modalities
        self.lang_modalities = lang_modalities
        self.target_modality = target_modality
        self.clip_mean = [0.48145466, 0.4578275, 0.40821073]
        self.clip_std = [0.26862954, 0.26130258, 0.27577711]
        self.clip_normalize = Normalize(mean=self.clip_mean, std=self.clip_std)
        self.device = device
        self.discard_action_history = True # action.shape = [batchsize, history, future, act_dim], discard not used history

    def forward(
        self,
        batch: dict,
        mode: Mode = Mode.INFERENCE,
        extra_args={},
    ) -> torch.Tensor:
        """
        Forward pass of the agent to generate actions depending on which mode the agent is in.
        """
        default_dtype = next(self.agent.parameters()).dtype
        # normalize imges with default clip values
        batch_size = batch[self.obs_modalities][self.img_modalities[0]].shape[0]
        for modality in (self.obs_modalities, self.goal_modalities, "future_obs"):
            if modality in batch:
                for img_modal in self.img_modalities:
                    if img_modal in batch[modality]:
                        if batch[modality][img_modal].shape[-1] == 3:
                            batch[modality][img_modal] = torch.moveaxis(batch[modality][img_modal], -1, -3) # move rgb to get [..., rgb, width, height]
                        if batch[modality][img_modal].dtype == torch.uint8:
                            batch[modality][img_modal] = convert_image_dtype(batch[modality][img_modal], dtype=default_dtype)
                        # Filter broken images out, to prevent ResNet outputting NaN
                        if modality == self.obs_modalities and 'pad_mask_dict' in batch[modality]:
                            for i in range(batch_size):
                                if torch.count_nonzero(batch[modality][img_modal][i]) == 0:
                                    batch[modality]['pad_mask_dict'][img_modal][i, 0] = False
                        batch[modality][img_modal] = self.clip_normalize(batch[modality][img_modal])

        if mode == Mode.INFERENCE:
            # Ugly hack to get around explicit bfloat16 conversions in the model
            batch['observation']['image_primary'] = batch['observation']['image_primary'].to(torch.bfloat16)

            self.agent.eval()
            obs = batch[self.obs_modalities]
            goal = batch[self.goal_modalities]
            return self.agent.step(obs, goal)
            # return self.agent(obs, goal)
        elif mode == Mode.TRAINING:
            """
            Computes the loss for the model during training
            loss depends on model
            """
            assert self.target_modality in batch, "Error, target_modality not found in batch! Hydra-configs might be wrong!"
            if self.discard_action_history:
                batch[self.target_modality] = batch[self.target_modality][:, -1]
            self.agent.train()
            # calculate loss with training_step method
            total_loss = self.agent.training_step(batch)
            return total_loss
        elif mode == Mode.EVALUATION:
            """
            Computes the loss for the model during training
            Standard MSE Loss
            """
            assert self.target_modality in batch, "Error, target_modality not found in batch! Hydra-configs might be wrong!"
            if self.discard_action_history:
                batch[self.target_modality] = batch[self.target_modality][:, -1]
            self.agent.eval()
            pred_loss = self.agent.validation_step(batch)
            return pred_loss
        else:
            print("Mode is NONE!")

    def get_optim_groups(self, optimizer_config):
        print("Getting Optimizer Groups")
        return self.agent.configure_optimizers(optimizer_config)

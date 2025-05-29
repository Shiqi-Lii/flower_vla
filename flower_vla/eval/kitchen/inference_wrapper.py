import functools
from typing import Optional
import numpy as np
import torch
from flower_vla.dataset.oxe.transforms import generate_policy_prompt, get_action_space_index
from flower_vla.dataset.utils.frequency_mapping import DATASET_FREQUENCY_MAP
from flower_vla.eval.simpler.flower_inference_wrapper import UhaInference as SimplerUhaInference

KIT_IRL_REAL_KITCHEN_DATASET_INDICES = [
    1,  # kit_irl_real_kitchen_delta_des_joint_euler
    2,  # kit_irl_real_kitchen_vis_delta_des_joint_euler
    3,  # kit_irl_real_kitchen_lang
    4,  # kit_irl_real_kitchen_vis
]

class UhaInference(SimplerUhaInference):
    def __init__(self, single_image, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # kitchen uses both primary and secondary
        self.agent.agent.img_modalities = ["image_primary", "image_secondary"]

        if single_image:
            self.agent.agent.img_modalities = ["image_primary"]

        print(self.agent.agent.img_modalities)

        self.max_values = torch.tensor([ 0.54222493,  0.80284792,  1.0687871 , -0.94879884,  0.06975963,        2.62478238, -0.49846622,  0.07        ]) # p99 # 1.0
        self.min_values = torch.tensor([-0.08110087, -0.69933356, -0.17626342, -2.7020722 , -2.43282706,        1.23236138, -2.72828501,  0.00        ]) # p01 # 0.0

        # Language Instruction
        self.format_instruction = functools.partial(
                generate_policy_prompt,
                robot_name="Franka Panda",
                action_space="joint position",
                num_arms="1",
                prompt_style='minimal'
            )
        
        # Action processing
        self.action_space_index = torch.tensor([get_action_space_index(robot_type='JOINT_POS', num_arms=1, control_mode='position', return_tensor=False)])
        self.frequency = torch.tensor([DATASET_FREQUENCY_MAP[KIT_IRL_REAL_KITCHEN_DATASET_INDICES[0]]])

    def step(self, primary_image: np.ndarray, secondary_image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            primary_image: np.ndarray of shape (H, W, 3), uint8
            secondary_image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_actions: np.ndarray; raw policy action output
        """
        task_description = self.format_instruction(task_description)
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; reset the policy state
                self.reset(task_description)
                self.agent.agent.reset()
                self.act_chunk_deque.clear()

        if self.ensemble_strategy == 'false' or self.ensemble_strategy is None:
            self.agent.agent.return_act_chunk = False
        else:
            self.agent.agent.return_act_chunk = True

        assert primary_image.dtype == np.uint8
        primary_image = torch.as_tensor(primary_image, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
        secondary_image = torch.as_tensor(secondary_image, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]

        input_observation = {
            "observation": {
                "image_primary": primary_image,
                "image_secondary": secondary_image,
                "pad_mask_dict": {
                    "image_primary": torch.ones(1,1).bool(),
                    "image_secondary": torch.ones(1,1).bool(),
                },
            },
            "task": {
                "language_instruction": self.task_description_embedding,
                "frequency": self.frequency,
                "action_space_index": self.action_space_index,
            }
        }
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                unscaled_raw_actions = self.agent(input_observation).cpu() # (action_dim)

        # next do custom ensemble strategy depending on the environment
         # Apply ensemble strategy if enabled (before rescaling)
        if self.ensemble_strategy == "act":
            # Convert to numpy for ensemble processing
            act_chunk = unscaled_raw_actions[..., :self.action_index.get_action_dim(self.action_space_index)]
            single_action = self.ensemble_action(act_chunk)
            unscaled_raw_actions = torch.from_numpy(single_action).unsqueeze(0)
        elif self.ensemble_strategy == "cogact":
            act_chunk = unscaled_raw_actions[..., :self.action_index.get_action_dim(self.action_space_index)]
            single_action = self.cognitive_ensemble_action(act_chunk)
            unscaled_raw_actions = single_action
        else:
            # Convert back to torch tensor
            unscaled_raw_actions = unscaled_raw_actions[..., :self.action_index.get_action_dim(self.action_space_index)]
            unscaled_raw_actions = unscaled_raw_actions

        raw_actions = torch.cat([self.rescale_to_range(unscaled_raw_actions[..., :-1]), unscaled_raw_actions[...,-1:]], dim=-1).detach()

        return raw_actions.detach().cpu().numpy()
    
    def rescale_to_range(self, tensor) -> torch.Tensor:
        max_values = self.max_values.cpu()[..., :tensor.shape[-1]]
        min_values = self.min_values.cpu()[..., :tensor.shape[-1]]
        # Scale the tensor to the new range [new_min, new_max]
        new_min = -torch.ones_like(tensor).cpu()
        new_max = torch.ones_like(tensor).cpu()
        rescaled_tensor = (tensor - new_min) / (new_max - new_min) * (max_values - min_values) + min_values
        return rescaled_tensor
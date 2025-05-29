from collections import defaultdict
import functools
import os
from typing import Optional, Sequence
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from transforms3d.euler import euler2axangle
import torch.nn as nn
import tensorflow_hub as hub
from transformers import CLIPTokenizer
from hydra import compose, initialize
import hydra
from safetensors.torch import load_file  # Add this import

# import Accelerator
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_model
# from agents.utils.ema import ExponentialMovingAverage
from flower_vla.agents.utils.diffuser_ema import EMAModel
# from flower_vla.agents.input_encoders.goal_encoders.language_encoders.clip_tokens import TokenLangClip
from flower_vla.agents.lang_encoders.florence_tokens import TokenVLM
from flower_vla.dataset.oxe.transforms import generate_policy_prompt, get_action_space_index
from flower_vla.dataset.utils.frequency_mapping import DATASET_FREQUENCY_MAP
from flower_vla.agents.utils.action_index import ActionIndex
POLICY_SETUP_TO_DATASET_INDEX = {
    "widowx_bridge": 0,
    "google_robot": 7,
    "skip_policy_setup": 0 # dummy
}

class UhaInference:
    def __init__(
        self,
        saved_model_base_dir: str = "/home/reuss/code/flower_vla_policy/logs/runs/2025-01-15/",
        saved_model_path: str = "15-23-08/checkpoint_5000",
        image_size: int = 224,
        pred_action_horizon: int = 5,
        action_scale: float = 1.0,
        policy_setup: str = "google_robot",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ensemble_strategy: str = "cogact",
        multistep: int = 1,
        num_sampling_steps: int = 1,
        adaptive_ensemble_alpha: float = 0.1,
        exp_decay: float = 0,
        use_torch_compile: bool = False,
        use_ema: bool = False,
        use_dopri5: bool = False,
    ) -> None:
        self.lang_embed_model = TokenVLM("microsoft/Florence-2-large")
        assert ensemble_strategy in ["false", "cogact", "act", "octo"]
        self.act_chunk_deque = deque(maxlen=pred_action_horizon)
        self.ensemble_strategy = ensemble_strategy
        self.image_size = image_size
        self.action_scale = action_scale
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.exp_decay = exp_decay
        self.use_torch_compile = use_torch_compile
        self.use_ema = use_ema
        self.use_dopri5 = use_dopri5
        # ------------------------- #
        model_path_split = saved_model_path.split("/")
        weights_path = saved_model_base_dir + model_path_split[0]
        checkpoint_path = os.path.join(weights_path, model_path_split[1])
        file_path = os.path.dirname(os.path.abspath(__file__))
        weights_path_relative = os.path.relpath(weights_path, file_path)
        with initialize(config_path=os.path.join(weights_path_relative, ".hydra")):
            cfg = compose(config_name="config")
            current_path = os.getcwd()
        ema_path = "random_states_0.pkl" 
        cfg.batch_size = 1
        cfg.trainer.agent.agent.act_window_size = pred_action_horizon # since we are doing single arm delta eef with 3 hz 
        cfg.trainer.agent.agent.multistep = multistep # since we are doing single arm delta eef with 3 hz
        cfg.trainer.agent.agent.num_sampling_steps = num_sampling_steps
        agent = hydra.utils.instantiate(cfg.trainer.agent, device=device, process_id=0)

        # Initialize accelerator for loading
        accelerator = Accelerator()
        agent = accelerator.prepare(agent)
        missing, unexpected = load_model(agent, os.path.join(checkpoint_path, "model.safetensors"))
        print(missing)
        print(unexpected)
        if self.use_ema:
            ema_helper = EMAModel(
                parameters=agent.parameters(),
                decay=cfg.decay,
                min_decay=0.0,
                update_after_step=0,
                use_ema_warmup=True,
                inv_gamma=1.0,
                power=2/3,
                foreach=False,
                model_cls=type(agent),
                model_config=agent.config if hasattr(agent, 'config') else None
            )

            ema_path = os.path.join(checkpoint_path, ema_path)
            if os.path.exists(ema_path):
                ema_state = torch.load(ema_path, map_location=device)
                ema_helper.load_state_dict(ema_state)
                print("Loaded EMA weights successfully")
                ema_helper.copy_to(agent.parameters())
        else:
            print("Not using EMA")
        # we cannot sue prioprio for isngl arm only meant for bimanual for now 
        agent.agent.use_proprio = False
        agent.to(dtype=torch.bfloat16)
        agent.eval()
        if self.use_torch_compile:
            agent.agent = torch.compile(agent.agent, mode="default")

        if self.ensemble_strategy != "false":
            # for all ensemble strategies, we need to return the act_chunk
            agent.agent.return_act_chunk = True
        
        # reduce img modalities
        agent.agent.img_modalities = ["image_primary"]
        if self.use_dopri5:
            agent.agent.use_dopri5 = True
        # ------------------------- #
        self.agent = agent
        self.pred_action_horizon = pred_action_horizon
        self.device = device
        self.observation = None
        self.task_description = None
        self.task_description_embedding = None
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.policy_setup = policy_setup
        if self.policy_setup == "google_robot":
            self.sticky_gripper_num_repeat = 15
            # use fractal20220817_data norm values
            self.max_values = torch.tensor([0.17824687153100965, 0.14938379630446405, 0.21842354819178575, 0.5892666035890578, 0.35272657424211445, 0.44796681255102094]) # p99
            self.min_values = torch.tensor([-0.22453527510166169, -0.14820013284683228, -0.231589707583189, -0.3517994859814644, -0.4193011274933815, -0.43643461108207704]) # p01
            self.format_instruction = functools.partial(
                generate_policy_prompt,
                robot_name="XARM",
                num_arms="1", 
                action_space="Delta End-Effector",
                prompt_style="minimal"
            )
        elif self.policy_setup == "widowx_bridge":
            self.sticky_gripper_num_repeat = 1
            # use bridge norm values
            self.max_values = torch.tensor([0.028122276067733765, 0.040630316659808145, 0.03994889184832546, 0.08121915772557152, 0.07724379181861864, 0.20214049845933896]) # p99 # 1.0
            self.min_values = torch.tensor([-0.028539552688598632, -0.041432044506073, -0.025977383628487588, -0.08020886614918708, -0.09213060349225997, -0.2054861941933632]) # p01 # 0.0
            self.format_instruction = functools.partial(
                generate_policy_prompt,
                robot_name="WidowX",
                num_arms="1",
                action_space="Delta End-Effector",
                prompt_style="minimal",
            )
        elif self.policy_setup == "skip_policy_setup":
            pass
        else:
            raise NotImplementedError()
        
        self.action_space_index = torch.tensor([get_action_space_index('EEF_POS', 1, 'velocity', return_tensor=False)])
        self.frequency = torch.tensor([DATASET_FREQUENCY_MAP[POLICY_SETUP_TO_DATASET_INDEX[self.policy_setup]]])
        
        self.action_index = ActionIndex()
    
    def rescale_to_range(self, tensor) -> torch.Tensor:
        max_values = self.max_values.cpu()
        min_values = self.min_values.cpu()
        # Scale the tensor to the new range [new_min, new_max]
        new_min = -torch.ones_like(tensor).cpu()
        new_max = torch.ones_like(tensor).cpu()
        rescaled_tensor = (tensor - new_min) / (new_max - new_min) * (max_values - min_values) + min_values
        return rescaled_tensor
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image
    
    def _initialize_task_description(self, task_description: Optional[str] = None) -> None:
        if task_description is not None:
            print("task description: ", task_description)
            self.task_description = task_description
            self.task_description_embedding = self.lang_embed_model([self.task_description])
        else:
            self.task_description = ""
            self.task_description_embedding = tf.zeros((512,), dtype=tf.float32)

    def reset(self, task_description: str) -> None:
        self._initialize_task_description(task_description)
        self.curr_horizon_index = 0

    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        task_description = self.format_instruction(task_description)
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; reset the policy state
                self.reset(task_description)
                self.agent.agent.reset()
                self.act_chunk_deque.clear()

        assert image.dtype == np.uint8
        image = torch.from_numpy(np.moveaxis(self._resize_image(image), -1, 0)).unsqueeze(0).unsqueeze(0).to(device=self.device)

        input_observation = {
            "image_primary": image,
            "pad_mask_dict": {"image_primary": torch.ones(1,1).bool().to(device=self.device)},
        }
        input_observation = {
            "observation": input_observation,
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
            act_chunk = unscaled_raw_actions[:, :, :self.action_index.get_action_dim(self.action_space_index)]
            single_action = self.ensemble_action(act_chunk)
            unscaled_raw_actions = torch.from_numpy(single_action).unsqueeze(0)
        elif self.ensemble_strategy == "cogact":
            act_chunk = unscaled_raw_actions[:, :, :self.action_index.get_action_dim(self.action_space_index)]
            single_action = self.cognitive_ensemble_action(act_chunk)
            unscaled_raw_actions = single_action
        else:
            # Convert back to torch tensor
            unscaled_raw_actions = unscaled_raw_actions[:self.action_index.get_action_dim(self.action_space_index)]
            unscaled_raw_actions = unscaled_raw_actions

        raw_actions = torch.cat([self.rescale_to_range(unscaled_raw_actions[..., :-1]), unscaled_raw_actions[...,-1:]], dim=-1).detach()
        #raw_actions = self.raw_actions[self.curr_horizon_index].numpy()
        # self.curr_horizon_index += 1
        assert raw_actions.shape == (7,)
        raw_action = {
            "world_vector": np.array(raw_actions[:3]),
            "rotation_delta": np.array(raw_actions[3:6]),
            "open_gripper": np.array(raw_actions[6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale
        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action
            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action
            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0
            action["gripper"] = relative_gripper_action



        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)
            # self.gripper_is_closed = (action['gripper'] < 0.0)
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action
    
    def visualize_epoch(self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str, process_index=0, wandb = None) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])
        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")
        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        if wandb is not None and wandb.run is not None and process_index is not None:
            name = "Simpler Env " + str(process_index) + ":"
            wandb.log({name: plt}, commit=False)
            plt.close()
        else:
            plt.savefig(save_path)

    def ensemble_action(self, cur_action):
        """
        Temporal ensemble strategy using exponential weighting over time
        Input: cur_action of shape [1, T, D_action] or [D_action]
        Output: single action step
        """
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )
        
        # More recent predictions get exponentially less weight than older predictions
        weights = np.exp(-self.exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()
        
        # Return weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)
        return torch.from_numpy(cur_action)

    def cognitive_ensemble_action(self, act_chunk):
        """
        from CogACT
        Cognitive ensemble strategy using cosine similarity with thresholding.
        Input: act_chunk of shape [1, T, D_action]
        Output: single action step
        """
        if len(act_chunk.shape) == 3:
            act_chunk = act_chunk.squeeze(0)  # Remove batch dimension
            
        # Always take the first action from the chunk if not enough history
        if len(self.act_chunk_deque) < 2:
            self.act_chunk_deque.append(act_chunk)  # Store full chunk
            return act_chunk[0]  # Return first action from the chunk
        
        # Add current chunk to history
        self.act_chunk_deque.append(act_chunk)
        num_actions = len(self.act_chunk_deque)
        curr_act_preds = np.stack(self.act_chunk_deque)  # [num_actions, T, D_action]
        
        # Get first timestep actions for all chunks
        first_timestep_actions = curr_act_preds[:, 0, :]  # [num_actions, D_action]
        
        # Get reference action (latest first timestep)
        ref = first_timestep_actions[-1]  # [D_action]
        
        # Calculate cosine similarities for first timestep actions
        dot_product = np.sum(first_timestep_actions * ref, axis=1)
        norm_preds = np.linalg.norm(first_timestep_actions, axis=1)
        norm_ref = np.linalg.norm(ref)
        cos_similarity = dot_product / (norm_preds * norm_ref + 1e-7)
        
        # Apply cognitive threshold to filter actions
        mask = cos_similarity >= self.adaptive_ensemble_alpha
        if not np.any(mask):
            return ref
            
        # Filter first timestep actions that meet threshold
        filtered_actions = first_timestep_actions[mask]  # [filtered_num, D_action]
        filtered_similarities = cos_similarity[mask]
        
        # Compute weights for filtered actions
        weights = np.exp(self.adaptive_ensemble_alpha * filtered_similarities)
        weights = weights / weights.sum()
        
        # Compute weighted average of filtered first timestep actions
        cur_action = np.sum(weights[:, None] * filtered_actions, axis=0)
        return torch.from_numpy(cur_action)


if __name__ == "__main__":
    testing = UhaInference()
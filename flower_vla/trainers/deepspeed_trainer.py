import os
import logging
import itertools
from typing import Dict, Any, Optional, List


import torch
import wandb
import hydra
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator

from flower_vla.agents.ddp_wrapper import Mode

# Assume these helper functions are defined here
def infinite_iter(dataloader):
    if dataloader is None:
        raise ValueError("Dataloader cannot be None")
    while True:
        for batch in dataloader:
            yield batch

def group_vlm_params(model, weight_decay: float, no_decay_keywords: Optional[List[str]] = None):
    """
    Groups parameters of the given model into two groups: with decay and without decay.
    """
    if no_decay_keywords is None:
        no_decay_keywords = ['bias', 'layernorm', 'ln', 'norm']
    decay_group, no_decay_group = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in no_decay_keywords):
            no_decay_group.append(param)
        else:
            decay_group.append(param)
    return [
        {"params": decay_group, "weight_decay": weight_decay},
        {"params": no_decay_group, "weight_decay": 0.0},
    ]

logger = logging.getLogger(__name__)

class AccelerateTrainer:
    """
    An improved trainer class using Hugging Face Accelerate.
    This class implements methods equivalent to your original trainer:
    - train_agent / train_step / evaluate_step
    - checkpointing, EMA, gradient logging, etc.
    """
    def __init__(
        self,
        agent,
        accelerator: Accelerator,
        datamodule,
        training_cfg: Dict[str, Any],
        optimizer_cfg: Dict[str, Any],
        ema_cfg: Optional[Dict[str, Any]] = None,
        use_deepspeed: bool = False,
        deepspeed_config: Optional[Dict[str, Any]] = None,
        use_torch_compile: bool = False,
        single_loader: bool = True
        ):
        self.accelerator = accelerator
        self.device = accelerator.device
        self.training_cfg = training_cfg
        self.optimizer_cfg = optimizer_cfg
        self.ema_cfg = ema_cfg
        self.use_deepspeed = use_deepspeed
        
        # Create dataloaders first
        self.train_dataloader = datamodule.create_train_dataloader(
            main_process=accelerator.is_main_process
        )
        self.val_dataloader = datamodule.create_val_dataloader(
            main_process=accelerator.is_main_process
        )
        
        # Instantiate agent
        self.agent = hydra.utils.instantiate(agent, device=self.device, 
                                            process_id=accelerator.process_index,
                                            accelerator=accelerator).to(self.device)
        self.agent = self.agent.to(torch.bfloat16)

        # Set up optimizers
        # For the transformer (DiT) parameters, use the transformer_weight_decay.
        # For DiT optimizer groups:
        wd = self.optimizer_cfg["weight_decay"]
        if isinstance(wd, dict):
            transformer_wd = wd["transformer_weight_decay"]
            vlm_wd = wd["vlm_weight_decay"]
        else:
            transformer_wd = vlm_wd = wd

        dit_groups = self.agent.get_optim_groups(transformer_wd)
        self.dit_optimizer = AdamW(dit_groups, lr=self.optimizer_cfg["learning_rate_dit"],
                                betas=self.optimizer_cfg.get("betas_dit", (0.9, 0.95)))

        vlm_groups = group_vlm_params(self.agent.agent.vlm, vlm_wd)
        self.vlm_optimizer = AdamW(vlm_groups, lr=self.optimizer_cfg.get("learning_rate_vlm", self.optimizer_cfg["learning_rate_dit"]),
                                betas=self.optimizer_cfg.get("betas_vlm", (0.9, 0.999)))
        


        self.dit_lr_scheduler = LambdaLR(self.dit_optimizer,
            lr_lambda=lambda step: max(0.0, 1 - step / self.training_cfg["max_train_steps"]))
        self.vlm_lr_scheduler = LambdaLR(self.vlm_optimizer, 
            lr_lambda=lambda step: max(0.0, 1 - step / self.training_cfg["max_train_steps"]))
        
        if use_deepspeed:
            prepared = self.accelerator.prepare(
                self.agent, self.train_dataloader, self.val_dataloader,
                self.dit_optimizer, self.vlm_optimizer, 
                self.dit_lr_scheduler, self.vlm_lr_scheduler
            )
            (self.agent, self.train_dataloader, self.val_dataloader,
                self.dit_optimizer, self.vlm_optimizer,
                self.dit_lr_scheduler, self.vlm_lr_scheduler) = prepared
        else:
            self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
                self.train_dataloader, self.val_dataloader
            )

        self.train_iter = infinite_iter(self.train_dataloader)
        self.val_iter = infinite_iter(self.val_dataloader)

        # Rest of initialization remains the same
        if ema_cfg and ema_cfg.get("use_ema", False):
            from flower_vla.agents.utils.ema import ExponentialMovingAverage
            self.ema_helper = ExponentialMovingAverage(
                self.agent.parameters(), ema_cfg["decay"], self.device
            )
            self.accelerator.register_for_checkpointing(self.ema_helper)
        else:
            self.ema_helper = None

        from flower_vla.agents.utils.dataset_metrics import DatasetMetricsTracker
        from flower_vla.dataset.utils.dataset_index import DATASET_INDEX_MAPPING
        self.metrics_tracker = DatasetMetricsTracker(DATASET_INDEX_MAPPING, self.accelerator)

        self.global_step = 0
        self.working_dir = os.getcwd()
        self.previous_stats = {}

        self.log_model_stats()

    def _setup_optimizers(self):
        use_single_optimizer = self.optimizer_cfg["single_optimizer"]
        lr_dit = self.optimizer_cfg["learning_rate_dit"]
        lr_vlm = self.optimizer_cfg.get("learning_rate_vlm", lr_dit)
        betas_dit = self.optimizer_cfg.get("betas_dit", (0.9, 0.95))
        betas_vlm = self.optimizer_cfg.get("betas_vlm", (0.9, 0.999))
        weight_decay_cfg = self.optimizer_cfg["weight_decay"]

        # Normalize weight_decay: if it's a dict, extract the subfields,
        # otherwise, assume it is a single float.
        if isinstance(weight_decay_cfg, dict):
            transformer_wd = weight_decay_cfg["transformer_weight_decay"]
            vlm_wd = weight_decay_cfg["vlm_weight_decay"]
        else:
            transformer_wd = vlm_wd = weight_decay_cfg

        if use_single_optimizer:
            # For a single combined optimizer, use the transformer weight decay
            # for DiT and add VLM parameters accordingly.
            dit_groups = self.agent.get_optim_groups(transformer_wd)
            vlm_model = self.agent.agent.vlm
            vlm_groups = group_vlm_params(vlm_model, vlm_wd)
            # Merge parameter groups appropriately.
            dit_groups[0]["params"].extend(vlm_groups[0]["params"])
            dit_groups[1]["params"].extend(vlm_groups[1]["params"])
            self.optimizer = AdamW(dit_groups, lr=lr_dit, betas=betas_dit)
            self.lr_scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(0.0, 1 - step / self.training_cfg["max_train_steps"])
            )
        else:
            # For separate optimizers, extract the proper float weight decay for each.
            dit_groups = self.agent.get_optim_groups(transformer_wd)
            self.dit_optimizer = AdamW(dit_groups, lr=lr_dit, betas=betas_dit)
            vlm_model = self.agent.agent.vlm
            vlm_groups = group_vlm_params(vlm_model, vlm_wd)
            self.vlm_optimizer = AdamW(vlm_groups, lr=lr_vlm, betas=betas_vlm)
            self.dit_lr_scheduler = LambdaLR(
                self.dit_optimizer,
                lr_lambda=lambda step: max(0.0, 1 - step / self.training_cfg["max_train_steps"])
            )
            self.vlm_lr_scheduler = LambdaLR(
                self.vlm_optimizer,
                lr_lambda=lambda step: max(0.0, 1 - step / self.training_cfg["max_train_steps"])
            )

        # Prepare components with the accelerator.
        objs = [self.agent, self.train_dataloader, self.val_dataloader]
        if use_single_optimizer:
            objs.extend([self.optimizer, self.lr_scheduler])
        else:
            objs.extend([
                self.dit_optimizer, self.vlm_optimizer,
                self.dit_lr_scheduler, self.vlm_lr_scheduler
            ])
        prepared = self.accelerator.prepare(*objs)
        if use_single_optimizer:
            (self.agent, self.train_dataloader, self.val_dataloader,
            self.optimizer, self.lr_scheduler) = prepared
        else:
            (self.agent, self.train_dataloader, self.val_dataloader,
            self.dit_optimizer, self.vlm_optimizer,
            self.dit_lr_scheduler, self.vlm_lr_scheduler) = prepared



    def train_agent(self):
        """
        Main training loop. This method mirrors your original `train_agent` but uses the improved
        structure and infinite iterators.
        """
        max_steps = self.training_cfg["max_train_steps"]
        save_every = self.training_cfg["save_every_n_steps"]
        eval_every = self.training_cfg["eval_every_n_steps"]

        logger.info("Starting training loop...")
        for step in range(self.global_step, max_steps):
            # Save checkpoint periodically
            if step > 0 and step % save_every == 0:
                self.store_model_weights(self.working_dir, f"{self.global_step}_")
                logger.info(f"Checkpoint saved at global step {self.global_step}")

            # Evaluate periodically
            if step > 0 and step % eval_every == 0:
                avg_loss, best_loss = self.evaluate(self.val_iter, best_test_loss=float('inf'))
                logger.info(f"Step {self.global_step}: Evaluation loss = {avg_loss:.6f}")

            # Get next training batch
            batch = next(self.train_iter)
            batch_loss = self.train_step(batch)

            if step % 1000 == 0 and self.accelerator.is_main_process:
                logger.info(f"Global Step {self.global_step}: Train Loss = {batch_loss:.6f}")

            self.global_step += 1

    def train_step(self, batch: Dict[str, Any]) -> float:
        """
        Executes a single training step with proper gradient handling, mixed precision,
        gradient clipping, and EMA updates.
        """
        #  with self.accelerator.accumulate(self.agent):
        self.agent.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = self.agent(batch, Mode.TRAINING)
            if isinstance(loss, dict):
                loss = loss["loss"]
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf detected in loss at step {self.global_step}: {loss.item()}")
                self.store_model_weights(self.working_dir, f"error_{self.global_step}_")
                raise RuntimeError("Loss is NaN or Inf")
        self.accelerator.backward(loss)

        # Optionally log gradient norms every 100 steps
        if self.global_step % 100 == 0 and wandb.run is not None and self.accelerator.is_main_process:
            self._log_gradient_norms()

        if self.accelerator.sync_gradients:
            # Clip gradients for submodules (adjust clipping values as needed)
            self.accelerator.clip_grad_norm_(self.agent.module.agent.dit.parameters(), 1.0)
            self.accelerator.clip_grad_norm_(self.agent.module.agent.vlm.parameters(), 1.0)

        # Step optimizers and schedulers
        if self.optimizer_cfg.get("single_optimizer", False):
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.dit_optimizer.step()
            self.vlm_optimizer.step()
            self.dit_lr_scheduler.step()
            self.vlm_lr_scheduler.step()
            self.dit_optimizer.zero_grad(set_to_none=True)
            self.vlm_optimizer.zero_grad(set_to_none=True)

        # Update EMA if enabled and at the correct interval
        if self.ema_helper and self.global_step % self.ema_cfg.get("update_every", 100) == 0:
            self.ema_helper.update(self.agent.parameters())

        return loss.item()

    @torch.no_grad()
    def evaluate_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a single evaluation step.
        """
        if self.ema_helper:
            self.ema_helper.store(self.agent.parameters())
            self.ema_helper.copy_to(self.agent.parameters())
            agent_to_eval = self.agent
        else:
            agent_to_eval = self.agent

        agent_to_eval.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = agent_to_eval(batch, mode="EVALUATION")
        if self.ema_helper:
            self.ema_helper.restore(self.agent.parameters())
        return output

    def evaluate(self, generator, best_test_loss: float) -> (float, float):
        """
        Runs evaluation over a fixed number of steps, aggregates metrics, and logs them.
        """
        test_losses = []
        num_eval_steps = self.training_cfg.get("max_eval_steps", 100)
        for _ in range(num_eval_steps):
            batch = next(generator)
            eval_dict = self.evaluate_step(batch)
            loss_val = eval_dict["loss"]
            if torch.is_tensor(loss_val):
                loss_val = loss_val.mean().item()
            test_losses.append(loss_val)
            # Update metrics tracker if available
            self.metrics_tracker.update(
                losses=eval_dict["loss"],
                dataset_indices=eval_dict.get("dataset_index", None)
            )
        # Gather losses across processes
        gathered_losses = self.accelerator.gather(torch.tensor(test_losses, device=self.device))
        avg_loss = float(gathered_losses.mean().item())
        if self.accelerator.is_main_process:
            logger.info(f"Global Step {self.global_step}: Avg Eval Loss = {avg_loss:.6f}")
        self.compute_and_log_metrics()
        best_test_loss = min(best_test_loss, avg_loss)
        return avg_loss, best_test_loss

    def store_model_weights(self, store_path: str, additional_name: str = "") -> None:
        """
        Saves model weights for inference and the full training state for resuming training.
        If DeepSpeed is enabled, uses a different naming convention (or additional DeepSpeed-specific logic)
        to ensure proper checkpointing.
        """
        os.makedirs(store_path, exist_ok=True)
        self.accelerator.wait_for_everyone()
        base_model = self.accelerator.unwrap_model(self.agent)
        
        # Check if DeepSpeed is active by looking at the distributed_type attribute
        if getattr(self.accelerator, "distributed_type", None) == "deepspeed":
            # Use a different file name or add extra DeepSpeed-specific checkpointing logic if needed.
            weights_path = os.path.join(store_path, f"{self.global_step}_model_weights_deepspeed.pt")
            torch.save(base_model.state_dict(), weights_path)
            logger.info(f"DeepSpeed model weights saved for inference: {weights_path}")
        else:
            weights_path = os.path.join(store_path, f"{self.global_step}_model_weights.pt")
            torch.save(base_model.state_dict(), weights_path)
            logger.info(f"Model weights saved for inference: {weights_path}")
        
        # Save the full training state.
        full_state_path = os.path.join(store_path, f"checkpoint_{self.global_step}")
        self.accelerator.save_state(full_state_path)
        logger.info(f"Full training state saved: {full_state_path}")


    def log_model_stats(self):
        """
        Logs the total number of parameters and trainable parameters.
        """
        total_params = sum(p.numel() for p in self.agent.parameters())
        trainable_params = sum(p.numel() for p in self.agent.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

    def _log_gradient_norms(self, error_mode: bool = False):
        """
        Logs gradient norms for the VLM and DiT submodules with a detailed breakdown.
        """
        log_fn = logger.error if error_mode else logger.info
        # Unwrap model if needed
        if hasattr(self.agent, 'module'):
            vlm = self.agent.module.agent.vlm
            dit = self.agent.module.agent.dit
        else:
            vlm = self.agent.agent.vlm
            dit = self.agent.agent.dit

        vlm_total_norm = 0.
        dit_total_norm = 0.
        for name, param in vlm.named_parameters():
            if param.grad is not None:
                vlm_total_norm += param.grad.detach().data.norm(2).item() ** 2
        for name, param in dit.named_parameters():
            if param.grad is not None:
                dit_total_norm += param.grad.detach().data.norm(2).item() ** 2
        vlm_total_norm = vlm_total_norm ** 0.5 if vlm_total_norm > 0 else 0.0
        dit_total_norm = dit_total_norm ** 0.5 if dit_total_norm > 0 else 0.0

        metrics = {
            "gradient_norms/vlm_total": vlm_total_norm,
            "gradient_norms/dit_total": dit_total_norm,
        }
        if vlm_total_norm > 0 and dit_total_norm > 0:
            metrics["gradient_norms/vlm_dit_ratio"] = vlm_total_norm / dit_total_norm
        if wandb.run is not None and self.accelerator.is_main_process:
            self.accelerator.log(metrics)

    def compute_and_log_metrics(self):
        """
        Computes evaluation metrics from the metrics tracker and logs them.
        """
        self.accelerator.wait_for_everyone()
        metrics = self.metrics_tracker.compute_metrics()
        if wandb.run is not None and self.accelerator.is_main_process:
            self.accelerator.log(metrics)
        self.accelerator.wait_for_everyone()

    def load_pretrained_model(self, weights_path: str, ema_name: str) -> None:
        """
        Loads a pretrained model checkpoint and associated EMA state.
        """
        from accelerate import load_checkpoint_and_dispatch
        load_checkpoint_and_dispatch(self.agent, weights_path)
        if self.ema_helper is None:
            from flower_vla.agents.utils.ema import ExponentialMovingAverage
            self.ema_helper = ExponentialMovingAverage(self.agent.parameters(), self.ema_cfg["decay"], self.device)
        ema_state = torch.load(os.path.join(weights_path, ema_name))
        self.ema_helper.load_state_dict(ema_state)
        logger.info("Loaded pre-trained agent parameters.")

    def continue_training(self, weights_path: str, ema_name: Optional[str] = None, step: int = 0) -> None:
        """
        Resumes training from a checkpoint.
        """
        if self.ema_helper is None:
            self.agent = self.accelerator.prepare(self.agent)
            from flower_vla.agents.utils.ema import ExponentialMovingAverage
            self.ema_helper = ExponentialMovingAverage(self.agent.parameters(), self.ema_cfg["decay"], self.device)
            self.accelerator.register_for_checkpointing(self.ema_helper)
        logger.info(f"Loading state from {weights_path}...")
        self.accelerator.load_state(weights_path)
        logger.info("State loaded.")
        self.global_step = step

    def load_pretrained_weights(
        self,
        weights_path: str,
        strict: bool = False,
        exclude_keys: Optional[List[str]] = None
    ) -> None:
        """
        Loads pretrained weights with an option to exclude certain keys.
        """
        try:
            if weights_path.endswith('.safetensors'):
                from flower_vla.utils.model_loading import load_file
                state_dict = load_file(weights_path)
            else:
                state_dict = torch.load(weights_path, map_location='cpu')
            if exclude_keys:
                state_dict = {k: v for k, v in state_dict.items() if not any(ex in k for ex in exclude_keys)}
            base_model = self.agent.agent if hasattr(self.agent, 'agent') else self.agent
            missing, unexpected = base_model.load_state_dict(state_dict, strict=strict)
            base_model.to(device=self.device, dtype=torch.bfloat16)
            logger.info(f"Successfully loaded pretrained weights from {weights_path}")
            if missing:
                logger.info(f"Missing keys: {missing}")
            if unexpected:
                logger.info(f"Unexpected keys: {unexpected}")
        except Exception as e:
            logger.error(f"Failed to load pretrained weights: {e}")
            raise

import os
import logging
import hydra
import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys

from torchvision.transforms import Normalize

from torchvision.transforms.functional import convert_image_dtype
# Add project root to path for Hydra
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockAccelerator:
    """Simple mock of Accelerate's Accelerator class for debugging."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self._device = device
        self.sync_gradients = True
        self.is_main_process = True
        self.use_distributed = False
        self.process_index = 0
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def autocast(self):
        """Mimics accelerator's autocast context manager"""
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    
    def backward(self, loss):
        """Mimics accelerator's backward"""
        loss.backward()
    
    def clip_grad_norm_(self, parameters, max_norm):
        """Mimics accelerator's grad norm clipping"""
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        
    def gather(self, tensor):
        """Mimics accelerator's gather without actually gathering"""
        return tensor
        
    def log(self, values, **kwargs):
        """Simple logging that just forwards to wandb if available"""
        if wandb.run is not None:
            wandb.log(values, **kwargs)
            
    def prepare(self, *args):
        """Returns args unchanged since we don't need distributed prep"""
        return args[0] if len(args) == 1 else args
        
    def unwrap_model(self, model):
        """Returns model unchanged since we don't have DDP wrapping"""
        return model


class SimpleTrainer:
    def __init__(
        self,
        agent_cfg: DictConfig,
        datamodule,
        target_modality: str,
        obs_modalities: str,
        goal_modalities: str,
        img_modalities: list,
        lang_modalities: list,
        max_train_steps: int,
        max_eval_steps: int,
        eval_every_n_steps: int,
        learning_rate_dit: float = 1e-5,
        learning_rate_vlm: float = 1e-5,
        device: str = "cuda"
    ):
        self.device = device
        self.max_train_steps = max_train_steps
        self.max_eval_steps = max_eval_steps
        self.eval_every_n_steps = eval_every_n_steps
        self.global_step = 0
        
        # Store modalities
        self.target_modality = target_modality
        self.obs_modalities = obs_modalities
        self.goal_modalities = goal_modalities
        self.img_modalities = img_modalities
        self.lang_modalities = lang_modalities
        
        # Initialize CLIP normalization
        self.clip_mean = [0.48145466, 0.4578275, 0.40821073]
        self.clip_std = [0.26862954, 0.26130258, 0.27577711]
        self.clip_normalize = Normalize(mean=self.clip_mean, std=self.clip_std)
        
        # Create mock accelerator
        self.accelerator = MockAccelerator(device=device)
        
        # Initialize agent
        logger.info("Initializing agent...")
        self.agent = hydra.utils.instantiate(
            agent_cfg,
            device=device,
            process_id=0,
            accelerator=self.accelerator
        ).to(device).to(torch.bfloat16)

        # get the agent without the ddpm wrapper
        self.agent = self.agent.agent
        
        # Initialize optimizers
        logger.info("Setting up optimizers...")
        learn_params = []
        
        for name, param in self.agent.named_parameters():
            if param.requires_grad:
                learn_params.append(param)
        # get a
        self.optimizer = torch.optim.AdamW(learn_params, lr=1e-4)
        
        # Initialize datamodule and dataloaders
        logger.info("Setting up dataloaders...")
        self.datamodule = datamodule
        self.train_loader = self.datamodule.create_train_dataloader(main_process=True)
        self.val_loader = self.datamodule.create_val_dataloader(main_process=True)
 

    def preprocess_images(self, batch):
        """Preprocess images in batch according to DDPWrapper logic."""
        batch_size = batch[self.target_modality].shape[0]
        
        for modality in (self.obs_modalities, self.goal_modalities, "future_obs"):
            if modality in batch:
                for img_modal in self.img_modalities:
                    if img_modal in batch[modality]:
                        # Move RGB channel
                        if batch[modality][img_modal].shape[-1] == 3:
                            batch[modality][img_modal] = torch.moveaxis(
                                batch[modality][img_modal], -1, -3
                            )
                        
                        # Convert dtype
                        if batch[modality][img_modal].dtype == torch.uint8:
                            batch[modality][img_modal] = convert_image_dtype(
                                batch[modality][img_modal], 
                                dtype=torch.float32
                            )
                        
                        # Filter broken images
                        if modality == self.obs_modalities and 'pad_mask_dict' in batch[modality]:
                            for i in range(batch_size):
                                if torch.count_nonzero(batch[modality][img_modal][i]) == 0:
                                    batch[modality]['pad_mask_dict'][img_modal][i, 0] = False
                        
                        # Normalize using CLIP values
                        batch[modality][img_modal] = self.clip_normalize(
                            batch[modality][img_modal]
                        )
        
        return batch

    def train_step(self, batch):
        """Single training step with image preprocessing."""
        self.agent.train()
        self.optimizer.zero_grad()
        
        # Preprocess images
        batch = self.preprocess_images(batch)
        
        # Move batch to device if not already there
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        # Convert batch to bfloat16
        batch = {
            k: v.to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.dtype != torch.bool else v 
            for k, v in batch.items()
        }

        # Forward pass
        # with torch.autograd.detect_anomaly():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = self.agent.training_step(batch)

        if not isinstance(loss, torch.Tensor):
            raise ValueError(f"Expected tensor loss, got {type(loss)}")

        # Backward pass
        self.accelerator.backward(loss)
        
        # Gradient clipping
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.agent.parameters(), 1.0)
        
        # Optimizer steps
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def evaluate(self, steps=None):
        """Evaluation loop with image preprocessing."""
        if steps is None:
            steps = self.max_eval_steps
            
        self.agent.eval()
        total_loss = 0.0
        
        for _ in range(steps):
            try:
                batch = next(self.val_iterator)
            except (StopIteration, AttributeError):
                self.val_iterator = iter(self.val_loader)
                batch = next(self.val_iterator)
            
            # Preprocess images
            batch = self.preprocess_images(batch)
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
                
            with self.accelerator.autocast():
                eval_dict = self.agent.validation_step(batch)
                loss = eval_dict['loss']
                total_loss += loss.mean().item()
                
        return total_loss / steps

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        self.train_iterator = iter(self.train_loader)
        if self.val_loader:
            self.val_iterator = iter(self.val_loader)
        
        try:
            for step in tqdm(range(self.max_train_steps)):
                # Get next batch
                try:
                    batch = next(self.train_iterator)
                except StopIteration:
                    self.train_iterator = iter(self.train_loader)
                    batch = next(self.train_iterator)
                
                # Training step
                loss = self.train_step(batch)
                
                # Log memory usage periodically
                if step % 100 == 0:
                    logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.1f}MB")
                    logger.info(f"GPU Memory reserved: {torch.cuda.memory_reserved(0)/1024**2:.1f}MB")
                
                # Regular logging
                if step % 50 == 0:
                    logger.info(f"Step {step}: Training loss = {loss:.4f}")
                    if wandb.run is not None:
                        self.accelerator.log({
                            "train/loss": loss,
                            "train/step": step,
                            "train/gpu_memory": torch.cuda.memory_allocated(0)/1024**2
                        })
                
                # Evaluation
                if step % self.eval_every_n_steps == 0 and step > 0:
                    eval_loss = self.evaluate()
                    logger.info(f"Step {step}: Evaluation loss = {eval_loss:.4f}")
                    if wandb.run is not None:
                        self.accelerator.log({
                            "eval/loss": eval_loss,
                            "eval/step": step
                        })                
                self.global_step = step
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise
        finally:           
            # Clean up
            if wandb.run is not None:
                wandb.finish()
                


@hydra.main(config_path="../conf", config_name="training.yaml")
def main(cfg: DictConfig) -> None:
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # overwrite buffer size for debugging
    cfg.shuffle_buffer_size = 5000
    cfg.batch_size = 32
    cfg.eval_every_n_steps = 200
    cfg.max_eval_steps = 10
    # Initialize wandb if needed
    if cfg.wandb.mode is not None:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    # Initialize datamodule
    logger.info("Initializing datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = SimpleTrainer(
        agent_cfg=cfg.trainer.agent,
        datamodule=datamodule,
        target_modality=cfg.target_modality,
        obs_modalities=cfg.obs_modalities,
        goal_modalities=cfg.goal_modalities,
        img_modalities=cfg.img_modalities,
        lang_modalities=cfg.lang_modalities,
        max_train_steps=cfg.max_train_steps,
        max_eval_steps=cfg.max_eval_steps,
        eval_every_n_steps=cfg.eval_every_n_steps,
        device=device
    )

    if 'continue_training' in cfg:
        # Load checkpoint if available
        trainer.load_checkpoint(cfg.continue_training.checkpoint_path)
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    # Enable tensor cores for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
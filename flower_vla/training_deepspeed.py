import os
import logging
import hydra
import wandb
import torch
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, DataLoaderConfiguration, InitProcessGroupKwargs
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from datetime import timedelta
import sys
import atexit

log = logging.getLogger(__name__)

def cleanup():
    """Ensure proper cleanup on script termination"""
    try:
        wandb.finish()
        torch.cuda.empty_cache()
    except Exception as e:
        log.error(f"Cleanup failed: {e}")

def setup_gpu():
    """Configure GPU settings with error handling"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.fastest = True
            torch.set_float32_matmul_precision('high')
            initial_mem = torch.cuda.memory_allocated()
            log.info(f"Initial GPU memory usage: {initial_mem / 1e9:.2f} GB")
            return True
        except Exception as e:
            log.error(f"GPU setup failed: {e}")
            return False
    return False

def setup_accelerator_deepspeed(cfg, single_loader):
    try:
        project_conf = ProjectConfiguration(
            project_dir=os.getcwd(), 
            automatic_checkpoint_naming=False
        )
        dataloader_conf = DataLoaderConfiguration(
            dispatch_batches=single_loader,
            non_blocking=True,
        )
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=cfg.find_unused_parameters,
            static_graph=cfg.static_graph,
        )
        process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=1800)
        )
        # Remove distributed_type from the call:
        accelerator = Accelerator(
            mixed_precision='bf16',
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            log_with="wandb",
            dataloader_config=dataloader_conf,
            project_config=project_conf,
            kwargs_handlers=[process_group_kwargs, ddp_kwargs],
            # Do NOT include distributed_type here.
            # The DeepSpeed configuration will instead be passed via the CLI flag.
            # deepspeed_config is also not passed in code, but via the command line.
        )
        return accelerator
    except Exception as e:
        log.error(f"DeepSpeed Accelerator setup failed: {e}")
        raise e


@hydra.main(config_path="../conf", config_name="training.yaml")
def main(cfg: DictConfig) -> None:
    # Register cleanup handler
    atexit.register(cleanup)
    
    try:
        single_loader = True

        # Setup GPU
        gpu_available = setup_gpu()
        if not gpu_available:
            log.warning("GPU setup failed, continuing with limited functionality")
        
        # Optionally set up torch compile
        if gpu_available and cfg.trainer.get("use_torch_compile", False):
            try:
                torch._dynamo.config.cache_size_limit = 8192
                torch._dynamo.config.suppress_errors = True
            except Exception as e:
                log.warning(f"Torch compile setup failed: {e}")
        
        # Setup Accelerator with DeepSpeed options
        accelerator = setup_accelerator_deepspeed(cfg, single_loader)
        
        # Initialize wandb trackers
        try:
            project = cfg.wandb.project
            del cfg.wandb.project
            accelerator.init_trackers(
                project_name=project,
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                init_kwargs={"wandb": OmegaConf.to_container(cfg.wandb, resolve=True, throw_on_missing=True)}
            )
        except Exception as e:
            log.error(f"WandB initialization failed: {e}")
            raise e

        # Scale parameters based on GPU count
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            cfg.batch_size = int(cfg.batch_size / device_count)
            if 'dit_lr_scheduler' in cfg.trainer:
                cfg.trainer.dit_lr_scheduler.num_warmup_steps *= device_count
                cfg.trainer.dit_lr_scheduler.timescale *= device_count
            if 'vlm_lr_scheduler' in cfg.trainer:
                cfg.trainer.vlm_lr_scheduler.total_steps *= device_count
            if 'lr_scheduler' in cfg.trainer:
                cfg.trainer.lr_scheduler.timescale *= device_count

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if gpu_available:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Temporarily change HOME if specified
        original_home = None
        if "HOME" in cfg:
            original_home = os.environ.get("HOME")
            os.environ["HOME"] = cfg.HOME

        try:
            log.info("Building datamodule...")
            datamodule = hydra.utils.instantiate(cfg.datamodule)
            log.info("Datamodule built successfully")
            if original_home:
                os.environ["HOME"] = original_home

            log.info("Building trainer...")
            trainer = hydra.utils.instantiate(
                cfg.trainer,
                accelerator=accelerator,
                datamodule=datamodule,
                single_loader=single_loader
            )
            log.info("Trainer built successfully")

            accelerator.wait_for_everyone()

            # Optionally load from checkpoint if continuing training
            if "continue_training" in cfg:
                try:
                    step = cfg.get("step", 0)
                    accelerator.load_state(cfg.continue_training)
                    log.info("Successfully loaded checkpoint")
                except Exception as e:
                    log.error(f"Failed to load checkpoint: {e}")
                    raise e

            log.info("Starting DeepSpeed training...")
            trainer.train_agent()
            log.info("Training completed successfully")

        except Exception as e:
            log.error(f"An error occurred during training: {e}")
            raise e

        finally:
            cleanup()

    except Exception as e:
        log.error(f"Fatal error in main: {e}")
        raise e

if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            log.info(f"CUDA available: {torch.cuda.device_count()} devices")
        main()
    except Exception as e:
        log.error(f"Script failed: {e}")
        sys.exit(1)

import os
# Set environment variables before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_DEBUG"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import time
import hydra
import numpy as np
import warnings
from tqdm import tqdm
import tensorflow as tf
import torch
from flower_vla.dataset import make_pytorch_oxe_iterable_dataset, get_octo_dataset_tensorflow
from omegaconf import DictConfig, OmegaConf
import logging
from flower_vla.dataset.utils.dataset_agnostics import diagnose_dataset_loading, test_dataset_loading, analyze_transform_compatibility
from flower_vla.dataset.oxe import make_oxe_dataset_kwargs_and_weights

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', message='The given NumPy array is not writable')


def setup_gpu():
    """Configure specific GPU and log device information."""
    # TensorFlow GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for GPU 0
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info(f"TensorFlow found {len(gpus)} GPU(s)")
            logger.info(f"Using GPU: {tf.config.list_physical_devices('GPU')[0]}")
        except RuntimeError as e:
            logger.warning(f"Error setting up TensorFlow GPU: {e}")
    else:
        logger.warning("TensorFlow: No GPUs found, using CPU")

    # PyTorch GPU setup
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Set PyTorch to use GPU 0
        logger.info(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"PyTorch GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        logger.warning("PyTorch: No GPUs found, using CPU")


def test_datamodule(datamodule):
    """Test datamodule initialization and iteration."""
    logger.info("Testing datamodule...")

    try:
        # Create train dataloader
        logger.info("Creating train dataloader...")
        train_loader = datamodule.create_train_dataloader(main_process=True)

        # Test iteration
        logger.info("Testing train dataloader iteration...")
        train_iterator = iter(train_loader)

        # Get first batch and analyze structure
        first_batch = next(train_iterator)
        logger.info("Successfully retrieved first batch")

        def log_structure(obj, prefix=""):
            """
            Recursively logs structure of `obj`.
            If `obj` is a dict, it inspects each key/value.
            If `obj` is a torch.Tensor, logs shape, dtype, min/mean/max stats.
            """
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Log key first
                    logger.info(f"{prefix}{key}:")
                    log_structure(value, prefix + "  ")
            elif isinstance(obj, torch.Tensor):
                # Log shape and stats (min, mean, max).
                # Use .detach().cpu() to avoid GPU->CPU issues when computing stats.
                stats_tensor = obj.detach().cpu()
                tensor_min = stats_tensor.min().item()
                tensor_max = stats_tensor.max().item()
                # For mean, convert to float to avoid type issues with integer Tensors
                tensor_mean = stats_tensor.float().mean().item()
                logger.info(
                    f"{prefix}Shape: {obj.shape}, dtype: {obj.dtype}, "
                    f"min: {tensor_min:.4f}, mean: {tensor_mean:.4f}, max: {tensor_max:.4f}"
                )
            else:
                # Not a dict or tensor: just log the type
                logger.info(f"{prefix}{type(obj)} = {obj}")

        logger.info("Dataset structure:")
        log_structure(first_batch)

        # Test multiple iterations
        logger.info("Testing multiple iterations...")
        for step in tqdm(range(10)):
            batch = next(train_iterator)
            if step % 2 == 0:
                logger.info(f"Successfully completed step {step}")
                if torch.cuda.is_available():
                    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

        # Test validation dataloader if available
        logger.info("Testing validation dataloader...")
        val_loader = datamodule.create_val_dataloader(main_process=True)
        if val_loader is not None:
            val_iterator = iter(val_loader)
            val_batch = next(val_iterator)
            logger.info("Successfully retrieved validation batch")

        # Get dataset statistics
        stats = datamodule.get_dataset_statistics()
        logger.info("Dataset statistics:")
        logger.info(f"Train dataset stats: {stats['train_dataset']}")
        if stats['val_dataset']:
            logger.info(f"Val dataset stats: {stats['val_dataset']}")

    except Exception as e:
        logger.error(f"Error testing datamodule: {e}")
        raise


@hydra.main(config_path="../conf", config_name="training")
def main(cfg: DictConfig):
    try:
        # Set up GPU
        setup_gpu()

        # Log config settings
        logger.info("Configuration:")
        logger.info(OmegaConf.to_yaml(cfg))

        # Initialize datamodule
        logger.info("Initializing datamodule...")
        cfg.shuffle_buffer_size = 1000
        datamodule = hydra.utils.instantiate(cfg.datamodule)

        # Run tests
        test_datamodule(datamodule)

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise


if __name__ == "__main__":
    main()

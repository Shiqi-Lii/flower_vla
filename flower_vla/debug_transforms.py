import os
# Set environment variables before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_DEBUG"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import hydra
import numpy as np
import warnings
import tensorflow as tf
import torch
from omegaconf import DictConfig, OmegaConf
import logging
import traceback
from pprint import pprint

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_resize_config(frame_transform_kwargs):
    """Debug the resize configuration."""
    logger.info("Debugging resize configuration...")
    
    if "resize_size" in frame_transform_kwargs:
        logger.info("Original resize_size config:")
        pprint(frame_transform_kwargs["resize_size"])
        
        # Convert lists to tuples
        if isinstance(frame_transform_kwargs["resize_size"], dict):
            frame_transform_kwargs["resize_size"] = {
                k: tuple(v) if isinstance(v, list) else v 
                for k, v in frame_transform_kwargs["resize_size"].items()
            }
        
        logger.info("Modified resize_size config:")
        pprint(frame_transform_kwargs["resize_size"])
    
    return frame_transform_kwargs

def setup_gpu():
    """Configure GPU if available."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logger.info(f"Using GPU: {gpus[0]}")
        except RuntimeError as e:
            logger.warning(f"Error setting up GPU: {e}")
    else:
        logger.warning("No GPUs found")

def debug_tf_dataset(dataset, max_samples=5):
    """
    Thoroughly inspect a TensorFlow dataset with detailed logging
    
    Args:
        dataset: TensorFlow dataset to inspect
        max_samples: Maximum number of samples to print
    """
    logger.info("\n" + "="*50)
    logger.info("TENSORFLOW DATASET DEBUGGING")
    logger.info("="*50)
    
    # Print dataset specification
    logger.info("\nDataset Specification:")
    try:
        spec = dataset.element_spec
        for key, tensor_spec in tf.nest.flatten_with_path(spec):
            logger.info(f"{key}: {tensor_spec}")
    except Exception as e:
        logger.error(f"Error printing dataset spec: {e}")
    
    # Try to iterate and print samples
    logger.info("\nSample Inspection:")
    try:
        iterator = iter(dataset)
        for i in range(max_samples):
            try:
                sample = next(iterator)
                logger.info(f"\nSample {i+1}:")
                
                def log_tensor_details(tensor, name="", indent=0):
                    prefix = "  " * indent
                    if isinstance(tensor, dict):
                        logger.info(f"{prefix}Dictionary with keys: {list(tensor.keys())}")
                        for k, v in tensor.items():
                            logger.info(f"{prefix}Key: {k}")
                            log_tensor_details(v, k, indent + 1)
                    elif isinstance(tensor, (list, tuple)):
                        logger.info(f"{prefix}Sequence of length {len(tensor)}")
                        for j, item in enumerate(tensor):
                            logger.info(f"{prefix}Item {j}:")
                            log_tensor_details(item, f"Item {j}", indent + 1)
                    elif tf.is_tensor(tensor):
                        try:
                            logger.info(f"{prefix}Tensor:")
                            logger.info(f"{prefix}  Shape: {tensor.shape}")
                            logger.info(f"{prefix}  DType: {tensor.dtype}")
                            logger.info(f"{prefix}  Sample values (first few): {tensor.numpy() if tensor.shape.ndims > 0 else tensor.numpy()}")
                        except Exception as e:
                            logger.error(f"{prefix}  Could not convert tensor to numpy: {e}")
                    else:
                        logger.info(f"{prefix}Basic type: {type(tensor)}")
                        logger.info(f"{prefix}Value: {tensor}")
                
                log_tensor_details(sample)
            
            except StopIteration:
                logger.info("Dataset exhausted before reaching max samples")
                break
            except Exception as e:
                logger.error(f"Error processing sample {i+1}: {e}")
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Error iterating through dataset: {e}")
        logger.error(traceback.format_exc())

@hydra.main(config_path="../conf", config_name="training")
def main(cfg: DictConfig):
    # Enhanced logging setup
    tf.get_logger().setLevel('DEBUG')
    
    try:
        # GPU setup
        setup_gpu()
        
        # Configuration logging
        logger.info("Initial configuration:")
        logger.info(OmegaConf.to_yaml(cfg))
        
        # Enable eager execution for more detailed error reporting
        tf.config.run_functions_eagerly(True)
        
        # Modify dataset configuration
        if hasattr(cfg.datamodule.datasets, "interleaved_dataset_cfg"):
            # Debug and modify resize configuration
            cfg.datamodule.datasets.interleaved_dataset_cfg.frame_transform_kwargs = \
                debug_resize_config(cfg.datamodule.datasets.interleaved_dataset_cfg.frame_transform_kwargs)
            
            # Adjust configuration for testing
            cfg.datamodule.datasets.interleaved_dataset_cfg.shuffle_buffer_size = 100
            cfg.datamodule.datasets.interleaved_dataset_cfg.traj_transform_threads = 2
            cfg.datamodule.datasets.interleaved_dataset_cfg.traj_read_threads = 2
            
            if "frame_transform_kwargs" in cfg.datamodule.datasets.interleaved_dataset_cfg:
                cfg.datamodule.datasets.interleaved_dataset_cfg.frame_transform_kwargs.num_parallel_calls = 2
        
        # Log modified configuration
        logger.info("Modified configuration:")
        logger.info(OmegaConf.to_yaml(cfg))
        
        # Initialize datamodule
        logger.info("Initializing datamodule...")
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        
        # Create train dataloader
        logger.info("Creating train dataloader...")
        train_loader = datamodule.create_train_dataloader(main_process=True)
        
        # Debug the TensorFlow dataset
        logger.info("Debugging TensorFlow dataset...")
        debug_tf_dataset(train_loader.dataset)
        
        # Test first batch
        logger.info("Testing first batch...")
        batch = next(iter(train_loader))
        logger.info(f"Successfully loaded first batch with keys: {batch.keys()}")
        
        # Additional batch inspection
        logger.info("Batch details:")
        for key, value in batch.items():
            logger.info(f"{key} type: {type(value)}")
            if hasattr(value, 'shape'):
                logger.info(f"{key} shape: {value.shape}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
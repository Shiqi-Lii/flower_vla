import torch
from flower_vla.dataset.dataset import make_interleaved_dataset 
from flower_vla.dataset.oxe import make_oxe_dataset_kwargs_and_weights
from torch.utils.data import DataLoader
from flower_vla.dataset.torch_dataset import TorchRLDSIterableDataset
import tensorflow as tf

from flower_vla.dataset.utils.data_utils import NormalizationType

tf.config.set_visible_devices([], "GPU")

def make_pytorch_oxe_iterable_dataset(dataset, language_encoder=None, train=True, batch_size=512, 
                                    transform_dict=None, num_workers=0, pin_memory=False, 
                                    drop_last=False, is_single_dataset=False, main_process=False, enable_tracking=False):
    torch_iterable = TorchRLDSIterableDataset(dataset, train, transform_dict, 
                                            language_encoder=language_encoder,
                                            is_single_dataset=is_single_dataset, enable_tracking=enable_tracking)
    
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 2 if main_process else 0, # 2 for kcist 1 for horeka 
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'shuffle': False if is_single_dataset else None
    }
    
    if main_process:
        loader_kwargs['prefetch_factor'] = 8
        
    return DataLoader(torch_iterable, **loader_kwargs)

def get_octo_dataset_tensorflow(cfg, train: bool):
    # Convert string to enum
    if isinstance(cfg.action_proprio_normalization_type, str):
        action_proprio_normalization_type = NormalizationType(cfg.action_proprio_normalization_type)
    else:
        action_proprio_normalization_type = cfg.action_proprio_normalization_type

    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.DATA_NAME,
        cfg.DATA_PATH,
        action_proprio_normalization_type=action_proprio_normalization_type,
        load_camera_views=cfg.load_camera_views,
    )

    if not train:
        cfg.interleaved_dataset_cfg.shuffle_buffer_size //= 100

    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
        **cfg.interleaved_dataset_cfg
    )

    return dataset
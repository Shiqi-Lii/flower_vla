"""
This example shows how to use the `octo.data` dataloader with PyTorch by wrapping it in a simple PyTorch
dataloader. The config below also happens to be our exact pretraining config (except for the batch size and
shuffle buffer size, which are reduced for demonstration purposes).
"""
import numpy as np
import torch
import torch.nn as nn
from flower_vla.dataset.utils.data_utils import hydra_get_object
from flower_vla.agents.lang_encoders.no_encoder import NoEncoder
from dlimp.dataset import DLataset
import wandb
from flower_vla.dataset.utils.frequency_mapping  import DATASET_FREQUENCY_MAP
from multiprocessing import Manager
import logging
from accelerate import Accelerator

logger = logging.getLogger(__name__)

class TorchRLDSIterableDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
            self,
            rlds_dataset: DLataset,
            train=True,
            transform_dict = None,
            language_encoder: nn.Module = NoEncoder(),
            is_single_dataset: bool = False,
            enable_tracking: bool = False  # Add parameter
    ):
        super(TorchRLDSIterableDataset).__init__()
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self._language_encoder = language_encoder
        # print("Using language encoder", language_encoder)
        self._is_single_dataset = is_single_dataset
        self._current_length = 0
        print("Transform dict", transform_dict)
        self._key_remapping = transform_dict["key_remapping"] if transform_dict is not None and "key_remapping" in transform_dict else None
        self._move_axis = transform_dict["move_axis"] if transform_dict is not None and "move_axis" in transform_dict else True
        self._add_empty_key = transform_dict["add_empty_key"] if transform_dict is not None and "add_empty_key" in transform_dict else []
        self._adjust_type = transform_dict["adjust_type"] if transform_dict is not None and "adjust_type" in transform_dict else None
        self._bytes_to_string = transform_dict["bytes_to_string"] if transform_dict is not None and "bytes_to_string" in transform_dict else True
        self._add_robot_information = transform_dict["add_robot_information"] if transform_dict is not None and "add_robot_information" in transform_dict else False

        # Initialize tracker
        self.tracker = SampleTracker(enable_tracking)

    def __iter__(self):
        for idx, sample in enumerate(self._rlds_dataset.iterator(prefetch=128)):
            # Track sample if enabled
            if self.tracker.enable_tracking and "dataset_name" in sample:
                dataset_name = sample["dataset_name"][0].decode() if isinstance(sample["dataset_name"], np.ndarray) else sample["dataset_name"]
                self.tracker.track_sample(dataset_name, idx)
                
                # Log coverage periodically
                if idx % 1000 == 0 and wandb.run is not None:
                    coverage = self.tracker.get_coverage_stats()
                    if coverage:
                        wandb.log({"dataset_coverage": coverage})

            # Original functionality
            if self._is_single_dataset:
                self._current_length = sample["action"].shape[0]
                for i in range(self._current_length):
                    sub_batch = self.limit_size(sample, dict(), i)
                    yield self.remap_sample(self.transform_sample(sub_batch))
            else:
                sample = self.transform_sample(sample)
                yield self.remap_sample(sample)

    def __len__(self):
        if hasattr(self._rlds_dataset, "dataset_len"):
            # print("dataset_len called", self._rlds_dataset.dataset_len)
            return self._rlds_dataset.dataset_len
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths = np.array(self._rlds_dataset.sample_weights) * lengths
        total_len = lengths.sum()
        # print("num_transitions called", total_len)
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)

    def limit_size(self, sample, sub_batch, index):
        if isinstance(sample, np.ndarray):
            return sample[index] # if index <= self._current_length-1 else (None, sample[:])
        else:
            for key in sample:
                sub_batch[key] = self.limit_size(sample[key], sub_batch[key] if key in sub_batch else dict(), index)
            return sub_batch

    def transform_sample(self, sample):
        dicts = ["observation", "task", "future_obs"]
        # print("Sample keys", sample["task"].keys())
        if self._move_axis:
            for key in dicts:
                if not key in sample:
                    continue
                if "image_primary" in sample[key]:
                    sample[key]["image_primary"] = np.moveaxis(sample[key]["image_primary"], -1, -3)
                if "image_secondary" in sample[key]:
                    sample[key]["image_secondary"] = np.moveaxis(sample[key]["image_secondary"], -1, -3)
                if "image_wrist" in sample[key]:
                    sample[key]["image_wrist"] = np.moveaxis(sample[key]["image_wrist"], -1, -3)

        # add proprioception information
        # print("Adding robot information", self._add_robot_information)


        if self._adjust_type is not None:
            dtype = hydra_get_object(self._adjust_type)
            sample["action"] = sample["action"].astype(dtype)

        if self._bytes_to_string:
            # Handle language instruction
            if sample["task"]["pad_mask_dict"]["language_instruction"]:
                sample["task"]["language_instruction"] = sample["task"]["language_instruction"].decode("utf-8")
                # print("Language instruction", sample["task"]["language_instruction"])
                # print("Language encoder", self._language_encoder)
                sample["task"]["language_instruction"] = self._language_encoder(sample["task"]["language_instruction"])
            else:
                sample["task"]["language_instruction"] = self._language_encoder("")


            # Handle robot information in the same way
            if sample["task"]["pad_mask_dict"]["robot_information"]:
                sample["task"]["robot_information"] = sample["task"]["robot_information"].decode("utf-8")
                sample["task"]["robot_information"] = self._language_encoder(sample["task"]["robot_information"])
            else:
                sample["task"]["robot_information"] = self._language_encoder("")
        
        # debug print to show all keys
        # print("Keys in sample", sample.keys())
        # print("Sample", sample)
        # print all keys in task and observation
        # print("Keys in task", sample["task"].keys())
        # print("Keys in observation", sample["observation"].keys())
        # Process array types
        # if 'frequency' in sample['task']:
        #    sample['task']['frequency'] = np.array(sample['task']['frequency'])
        if 'dataset_index' in sample['task']:
            sample['task']['dataset_index'] = np.array(sample['task']['dataset_index'])
        if 'action_space_index' in sample['task']:
            sample['task']['action_space_index'] = np.array(sample['task']['action_space_index'])

        # now add frequency information
        if "dataset_index" in sample:
            sample["task"]["frequency"] = np.array([DATASET_FREQUENCY_MAP[sample["dataset_index"]]])
        else:
            sample["task"]["frequency"] = np.array([10])

        return sample
    
    def remap_sample(self, sample):
        if self._key_remapping is None:
            if len(self._add_empty_key) != 0:
                for key in self._add_empty_key:
                    sample[key] = {}
            if "dataset_name" in sample:
                del sample["dataset_name"]
            return sample
        else:
            transformed_sample = {}
            if len(self._add_empty_key) != 0:
                for key in self._add_empty_key:
                    transformed_sample[key] = {}
            # { observation: { image_primary: ["rgb_obs", "rgb_static"], ... }, ...}
            for old_key, value in self._key_remapping.items():
                if isinstance(value, dict):
                    for second_old_key, new_value in value.items():
                        if isinstance(new_value, list) and len(new_value) == 2:
                            transformed_sample[new_value[0]][new_value[1]] = sample[old_key][second_old_key]
                        elif isinstance(new_value, list) and len(new_value) == 1:
                            transformed_sample[new_value[0]] = sample[old_key][second_old_key]
                        else:
                            transformed_sample[new_value] = sample[old_key][second_old_key]
                else:
                    if isinstance(value, list) and len(value) == 2:
                        transformed_sample[value[0]][value[1]] = sample[old_key]
                    elif isinstance(value, list) and len(value) == 1:
                        transformed_sample[value[0]] = sample[old_key]
                    else:
                        transformed_sample[value] = sample[old_key]

            return transformed_sample


class TorchRLDSIterableDatasetTF(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
            self,
            rlds_dataset: DLataset,
            train=True,
            transform_dict = None,
            language_encoder: nn.Module = NoEncoder(),
            is_single_dataset: bool = False,
    ):
        super(TorchRLDSIterableDatasetTF).__init__()
        self._rlds_dataset = rlds_dataset
        self._is_train = train
        self._language_encoder = language_encoder
        self._is_single_dataset = is_single_dataset
        self._current_length = 0

    def __iter__(self):
        rlds_iter = map(self.process_batch, self._rlds_dataset.iterator()) # prefetch=1024
        for sample in rlds_iter: # 4 * batchsize
        # for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        if hasattr(self._rlds_dataset, "dataset_len"):
            # print("dataset_len called", self._rlds_dataset.dataset_len)
            return self._rlds_dataset.dataset_len
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths = np.array(self._rlds_dataset.sample_weights) * lengths
        total_len = lengths.sum()
        # print("num_transitions called", total_len)
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)

    def process_batch(self, batch):
        if isinstance(self._language_encoder, NoEncoder):
            batch["task"].pop("language_instruction")
        else:
            batch["task"]["language_instruction"] = self._language_encoder(batch["task"]["language_instruction"].decode("utf-8"))
        del batch["dataset_name"]
        return batch



class SampleTracker:
    def __init__(self, enable_tracking=False):
        self.enable_tracking = enable_tracking
        self.sample_usage = {}
        self.coverage_metrics = {}
        self.replacement_queue = []

    def track_sample(self, dataset_name, sample_idx):
        logger.info(f"Tracking sample: {dataset_name} -> {sample_idx}")
        if dataset_name not in self.sample_usage:
            self.sample_usage[dataset_name] = {}
            self.coverage_metrics[dataset_name] = {"total": 0, "used": 0}
        if sample_idx not in self.sample_usage[dataset_name]:
            self.sample_usage[dataset_name][sample_idx] = 0
            self.coverage_metrics[dataset_name]["total"] += 1
        self.sample_usage[dataset_name][sample_idx] += 1
        self.coverage_metrics[dataset_name]["used"] = len(self.sample_usage[dataset_name])
        logger.info(f"Updated sample_usage: {self.sample_usage}")
        logger.info(f"Updated coverage_metrics: {self.coverage_metrics}")

    def get_coverage_stats(self):
        return {
            name: {
                "coverage": stats["used"] / stats["total"] if stats["total"] > 0 else 0,
                "total_samples": stats["total"]
            }
            for name, stats in self.coverage_metrics.items()
        }

    def aggregate_metrics(self, accelerator: Accelerator):
        """Aggregate metrics across distributed processes."""
        local_coverage = self.get_coverage_stats()

        # Convert metrics to tensors for synchronization
        local_stats = {k: (torch.tensor(v["used"]), torch.tensor(v["total_samples"]))
                       for k, v in local_coverage.items()}

        global_stats = {}
        for dataset_name, (used, total) in local_stats.items():
            global_used = accelerator.gather(used).sum()
            global_total = accelerator.gather(total).sum()
            global_stats[dataset_name] = {
                "coverage": global_used.item() / global_total.item(),
                "total_samples": global_total.item()
            }

        return global_stats
    
    def replace_overused_samples(self):
        if not self.enable_tracking:
            return
            
        for dataset_name, sample_idx in self.replacement_queue:
            # Get new sample
            new_sample = self._get_new_sample(dataset_name)
            # Reset usage count
            self.sample_usage[dataset_name][sample_idx] = 0
            
        self.replacement_queue.clear()
    
    def get_dataset_weights(self):
        if not self.enable_tracking:
            return None
            
        weights = {}
        for dataset_name, usage in self.sample_usage.items():
            # Calculate inverse usage weight
            avg_usage = np.mean(list(usage.values()))
            weights[dataset_name] = 1.0 / (avg_usage + 1e-6)
            
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k,v in weights.items()}
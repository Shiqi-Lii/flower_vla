


## Installation

This Repo 
- Python 3.10 for training with simpler_env enabled
to enable the use of "flash-attn", add it afterwards.

To begin, clone this repository locally
```bash
git clone --recurse-submodules git@github.com:mbreuss/flower_vla.git
```

---
### Installing training with simpler_env (also works with venv)
Install requirements:
```bash
cd flower_vla
conda create -n flower python=3.10
conda activate flower
pip install -r requirements_simpler.txt
```



---
### Installing evaluating_real_world (requires mamba)


Follow "On Workstation" of [irl_polymetis installation](./evaluating_real_world/irl_polymetis/README.md)

Install additional requirements:
```bash
cd flower_vla_policy
conda activate $irl_polymetis_environment_name
pip install -r requirements_real_world.txt
```

### Installing LIBERO Setup

Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO):
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```
Additionally, install other required packages:
```bash
cd flower_vla_policy
pip install -r requirements_additional_libero.txt
```



## Usage
First, create a matching accelerate config. The following example is for training on two GPUs (cuda:0 + cuda:1). For further information, please read [the official documentation](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config)
```bash
accelerate config
# accelerate asks you questions, these are the answers:
This machine
multi-GPU
1
NO
NO
NO
NO
NO
2
0,1
yes
bf16
```
Afterwards, the "often changed" parameters in [the training.yaml](conf/training.yaml) have to be adjusted, depending on the used cluster size.

To launch a training process start [the training.py](medit/training.py) file with accelerate
```bash
accelerate launch flower/training.py
```

To continue training, add the "step" and "continue_training" parameters to the hydra config, either directly inside the config or per command line.
```bash
accelerate launch flower/training.py +step=100 +continue_training=/home/marcelr/MeDiT_Policy/logs/runs/2024-08-29/22-41-23/checkpoint_100
```


### Random Moritz Notes

Current mix with good balance across different datasets for delta EEF:
```python
TRINITY = [
    ("bridge_dataset", 4.0),
    ("fractal20220817_data", 2.0),
    ("dobbe", 2.0),
    ("cmu_play_fusion", 9.0),
    ("libero_10_no_noops", 10.0),
    ("libero_goal_no_noops", 18.0),
]
```


For joint state pretraining we will just use droid


### Download Datasets


*Bridge Dataset*

```bash
wget -r -np -nd -A '*' https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/
```

*BiPlay*

Its a Hugging Face dataset which  makes it easy:

1. First Install lfs for git

```bash
git lfs install
```

2. Next clone the dataset
```bash
git clone https://huggingface.co/datasets/oier-mees/BiPlay
```

*kit_irl_real_kitchen_lang*

Same can be done to download: 

```bash
git clone https://huggingface.co/datasets/mbreuss/kit_irl_real_kitchen_lang
```

*CALVIN_ABC*

```bash
git clone https://huggingface.co/datasets/zhouhongyi/calvin_abc_rlds
```

*BiManual Aloha*

ToDO Hongyi 



### Compress Data to tgz for Horeka

```bash
tar -czvf archive.tgz /path/to/folder
```

## Droid normalization keys

We neeed to load improved language annotations from: 

- [droid](https://huggingface.co/KarlP/droid)

When adding the language keys, we need to load updated hashes:

Therefore go to flower/dataset/utils/rlds_utils

modify 

```python
def compute_dataset_statistics(
    builder: tfds.builder,
    filter_functions: Sequence[ModuleSpec],
    ignore_errors: bool,
    restructure_fn: Callable,
    proprio_obs_key: Optional[str],
    standardize_fn: Optional[ModuleSpec],
    force_recompute: bool = False
) -> dict:
    """Compute or load cached dataset statistics."""
    # Create full dataset for statistics computation
    full_dataset = dl.DLataset.from_rlds(
        builder, 
        split="all", 
        shuffle=False
    )
    
    # Apply filters
    for filter_fcn_spec in filter_functions:
        full_dataset = full_dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
    
    if ignore_errors:
        full_dataset = full_dataset.ignore_errors()
        
    # Apply restructuring
    full_dataset = full_dataset.traj_map(restructure_fn).filter(
        lambda traj: tf.shape(traj["action"])[0] > 0
    )
    
    # Generate hash dependencies
    hash_dependencies = (
        str(builder.info),
        str(proprio_obs_key),
        ModuleSpec.to_string(standardize_fn) if standardize_fn is not None else "",
        *map(ModuleSpec.to_string, filter_functions),
    )
    
    # Create unique hash
    unique_hash = hashlib.sha256(
        "".join(hash_dependencies).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()

    # Set up cache paths
    local_path = os.path.expanduser(
        os.path.join(
            "~",
            ".cache",
            "octo",
            f"dataset_statistics_{unique_hash}.json",
        )
    )

    path = tf.io.gfile.join(builder.data_dir, f"dataset_statistics_{unique_hash}.json") \
        if builder.data_dir is not None else local_path
    
    logging.info(f"Unique hash for dataset statistics: {unique_hash}")
    logging.info(f"Cache path: {path}")
    logging.info(f"Cache path: {local_path}")

```

The last three lines will show you which hash you need to create with the following values from below":

Just copy the has path and the fill in the values from below:

*for eef_droid:

```json
{"action": {"mean": [0.5396102070808411, 0.0013412077678367496, 0.3151766359806061, 0.31626930832862854, -0.09017368406057358, -0.04949692636728287, 0.28914710879325867], "std": [0.11743763834238052, 0.17490370571613312, 0.16182278096675873, 2.743474245071411, 0.34967610239982605, 0.7598094344139099, 0.4094192683696747], "max": [0.9295652508735657, 0.8648782968521118, 1.1109716892242432, 3.1415927410125732, 1.5702463388442993, 3.1415891647338867, 1.0], "min": [-0.2824200689792633, -0.8714114427566528, -0.3079752027988434, -3.141592502593994, -1.570475697517395, -3.1415903568267822, 0.0], "p99": [0.7961599677801132, 0.44313666224479675, 0.7910010814666748, 3.137368679046631, 0.8984405845403671, 2.0671035647392273, 1.0], "p01": [0.2645447924733162, -0.44233740866184235, -0.05131602194160223, -3.137308120727539, -1.2215128540992737, -2.1878013014793396, 0.0]}, "num_transitions": 27044326, "num_trajectories": 92233, "proprio": {"mean": [0.5353850722312927, 0.0015335299540311098, 0.3146042823791504, 0.32695427536964417, -0.08701429516077042, -0.048322372138500214, 0.15373258292675018], "std": [0.11646922677755356, 0.17391017079353333, 0.16116510331630707, 2.7485382556915283, 0.3466009497642517, 0.7527545690536499, 0.33395257592201233], "max": [0.8575563430786133, 0.8407337069511414, 1.0439032316207886, 3.1415927410125732, 1.5705928802490234, 3.1415927410125732, 1.0], "min": [-0.2824079692363739, -0.8556680083274841, -0.24001094698905945, -3.141592502593994, -1.5703768730163574, -3.141592025756836, 0.0], "p99": [0.7826369255781174, 0.44099078327417374, 0.7858299612998962, 3.137469530105591, 0.891441822052002, 2.051763117313385, 1.0], "p01": [0.2667432054877281, -0.43949584662914276, -0.04718756675720215, -3.1373939514160156, -1.215930700302124, -2.173954963684082, 0.0]}}
```

**droid** (default as joint state)

```json
{"action": {"mean": [0.015764277428388596, 0.25588878989219666, -0.014796368777751923, -2.1007542610168457, -0.041470348834991455, 2.4823384284973145, 0.08672530949115753, 0.28914546966552734], "std": [0.32319068908691406, 0.49863681197166443, 0.28734293580055237, 0.4985961318016052, 0.5200886130332947, 0.47899195551872253, 0.7317898869514465, 0.40938955545425415], "max": [2.753600597381592, 1.6689813137054443, 2.769918203353882, -0.1839631199836731, 2.781451463699341, 4.402013778686523, 2.90183162689209, 1.0], "min": [-2.781099557876587, -1.6589616537094116, -2.772181749343872, -2.9508564472198486, -2.7826988697052, 0.17761151492595673, -3.01689076423645, 0.0], "p99": [1.03982213139534, 1.4560052752494812, 0.8827952593564987, -0.4233001172542572, 1.7057677805423737, 3.4945608377456665, 2.260508418083191, 1.0], "p01": [-0.9600185006856918, -0.8849125504493713, -0.9973162412643433, -2.7765824794769287, -1.8709716498851776, 1.1590989828109741, -2.1379868388175964, 0.0]}, "num_transitions": 27044326, "num_trajectories": 92233, "proprio": {"mean": [0.016574395820498466, 0.24999158084392548, -0.015175092965364456, -2.1140475273132324, -0.04155038669705391, 2.4822726249694824, 0.08583430200815201, 0.15373286604881287], "std": [0.3199721574783325, 0.5034835338592529, 0.28598693013191223, 0.4958859980106354, 0.5145179033279419, 0.47597482800483704, 0.7253726720809937, 0.33395037055015564], "max": [2.677424192428589, 1.5840554237365723, 2.6957037448883057, -0.29779934883117676, 2.6624162197113037, 4.309162616729736, 2.755643367767334, 1.0], "min": [-2.672288417816162, -1.6589547395706177, -2.680800676345825, -2.9409868717193604, -2.6705946922302246, 0.24893812835216522, -3.01689076423645, 0.0], "p99": [1.0290410220623016, 1.4492228031158447, 0.8692934662103653, -0.4314958080649376, 1.6885132789611816, 3.476126551628113, 2.2478813529014587, 1.0], "p01": [-0.9489463269710541, -0.8921926021575928, -0.9835036844015121, -2.7745561599731445, -1.8561363220214844, 1.1744330823421478, -2.12151300907135, 0.0]}}
```


## Helpfull Utilities
Enable further Torch debugging:
```
export TORCH_DISTRIBUTED_DEBUG=DETAIL
or
export TORCH_DISTRIBUTED_DEBUG=INFO
```

Download from [gsutil](https://console.cloud.google.com/storage/browser/gresearch/robotics) for [HoreKa](https://www.nhr.kit.edu/userdocs/horeka/)
```
# first switch to conda and install gsutil, then:
gsutil -m cp -r gs://gresearch/robotics/berkeley_mvp_converted_externally_to_rlds $(ws_find datasets)
gsutil -m cp -r gs://gresearch/robotics/berkeley_rpt_converted_externally_to_rlds $(ws_find datasets)
gsutil -m cp -r gs://gresearch/robotics/toto $(ws_find datasets)
gsutil -m cp -r gs://gresearch/robotics/aloha_mobile $(ws_find datasets)
```
Create small archive for $TMPDIR
```
cd $(ws_find datasets)
tar -cvzf $(ws_find datasets)/berkeley_mvp_converted_externally_to_rlds.tgz berkeley_mvp_converted_externally_to_rlds/
tar -cvzf $(ws_find datasets)/berkeley_rpt_converted_externally_to_rlds.tgz berkeley_rpt_converted_externally_to_rlds/
tar -cvzf $(ws_find datasets)/toto.tgz toto/
tar -cvzf $(ws_find datasets)/aloha_mobile.tgz aloha_mobile/

# Libero datasets
tar -cvzf $(ws_find datasets)/libero_10_no_noops.tgz modified_libero_rlds/libero_10_no_noops/
tar -cvzf $(ws_find datasets)/libero_goal_no_noops.tgz modified_libero_rlds/libero_goal_no_noops/
tar -cvzf $(ws_find datasets)/libero_object_no_noops.tgz modified_libero_rlds/libero_object_no_noops/
tar -cvzf $(ws_find datasets)/libero_spatial_no_noops.tgz modified_libero_rlds/libero_spatial_no_noops/
```


lessons learned: accelerate config, beim anlegen keine eckigen Klammern, generell beschreiben wie das geht (GPU / fp16 <- aktuell noch nicht ganz supported)

## Current Dataset Stats
Each Transition marks 1 tranining sample, the trajectories describe the task richness of the samples.
For more details, see [Open-X embodiment spreadsheet](https://docs.google.com/spreadsheets/d/1rPBD77tk60AEIGZrGSODwyyzs5FgCU9Uz3h-3_t2A9g/edit#gid=0)
#!/bin/bash

#SBATCH -p accelerated-h100 # accelerated-h100 # dev_accelerated
#SBATCH -A hk-project-sustainebot
#SBATCH -J h100-flower-training

# Cluster Settings
#SBATCH -n 1       # Number of tasks
#SBATCH -c 16  # Number of cores per task
#SBATCH -t 2-00:00:00 # 30:00 # 2-00:00:00 # 1:00:00 # 2-00:00:00 ## # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1

# Define the paths for storing output and error files
#SBATCH --output=/home/hk-project-sustainebot/ft4740/code/flower_vla_policy/logs/slurm_logs/%x_%j.out
#SBATCH --error=/home/hk-project-sustainebot/ft4740/code/flower_vla_policy/logs/slurm_logs/%x_%j.err


# Activate the virtualenv / conda environment
source /home/hk-project-robolear/ft4740/miniconda3/bin/activate flower_vla 
export TORCH_USE_CUDA_DSA=1

export CURL_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export TF_ENABLE_ONEDNN_OPTS=0

# Create datasets directory and print its location
# Create datasets directory
echo "Creating tensorflow_datasets directory in TMPDIR: $TMPDIR"
mkdir -p $TMPDIR/tensorflow_datasets
echo "Created directory at: $TMPDIR/tensorflow_datasets"

# Extract datasets with path stripping
echo "Extracting fractal dataset..."
cd $TMPDIR/tensorflow_datasets/
tar -C $TMPDIR/tensorflow_datasets/ -xvzf $(ws_find play3)/bridge_dataset.tgz
# tar -C $TMPDIR/tensorflow_datasets/ -xvzf $(ws_find play3)/fractal20220817_data.tgz
# tar --strip-components=4 -xvzf $(ws_find play3)/fractal20220817_data.tgz -C $TMPDIR/tensorflow_datasets/

# Debugging dataset structure
echo "Final dataset structure check:"
find $TMPDIR/tensorflow_datasets -type d -ls
find $TMPDIR/tensorflow_datasets -type f -name "*.tfrecord*" | head -n 5

# Launch training
accelerate launch --main_process_port 29872 \
    /home/hk-project-sustainebot/ft4740/code/flower_vla_policy/flower/training.py \
    +HOME=$TMPDIR
#!/bin/bash

#SBATCH -p dev_accelerated
#SBATCH -A hk-project-sustainebot
#SBATCH -J flower-training

# Cluster Settings
#SBATCH -n 1       # Number of tasks
#SBATCH -c 16  # Number of cores per task
#SBATCH -t 1:00:00 # 2-00:00:00 ## # 06:00:00 # 1-00:30:00 # 2-00:00:00
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

mkdir $TMPDIR/tensorflow_datasets
tar -C $TMPDIR/tensorflow_datasets/ -xvzf $(ws_find play3)/bridge_dataset.tgz
# tar -C $TMPDIR/tensorflow_datasets/ -xvzf $(ws_find play3)/fractal20220817_data.tgz
#tar -C $TMPDIR/tensorflow_datasets/ -xvzf $(ws_find play3)/libero_goal_no_noops.tgz
#tar -C $TMPDIR/tensorflow_datasets/ -xvzf $(ws_find play3)/libero_object_no_noops.tgz
#tar -C $TMPDIR/tensorflow_datasets/ -xvzf $(ws_find play3)/libero_spatial_no_noops.tgz
#tar -C $TMPDIR/tensorflow_datasets/ -xvzf $(ws_find play3)/bridge.tgz
mkdir $TMPDIR/.cache
# cp -r $HOME/.cache/octo $TMPDIR/.cache/octo

# python oxe_torch_dataloader/scripts/test_loader.py +HOME=$TMPDIR # computes dataset_statistics for $TMPDIR datasets
accelerate launch --main_process_port 29872 /home/hk-project-sustainebot/ft4740/code/flower_vla_policy/flower/training.py +HOME=$TMPDIR
# datamodule.datasets.DATA_PATH=$TMPDIR/tensorflow_datasets # DATA_PATH=$TMPDIR/tensorflow_datasets # +HOME=$TMPDIR

# sbatch error: https://docs.hpc.gwdg.de/known_issues/slurm_does_not_recognize_job_script/index.html
#!/bin/bash

# take config filename from first argument
# USAGE: sbatch job.sh config_filename

#SBATCH --job-name=cnn-cd
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4

python train_cd_model.py

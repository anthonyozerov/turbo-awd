#!/bin/bash

# take config filename from first and second arguments
# USAGE: sbatch job-cnn.sh config cnn_config

#SBATCH --job-name=online-cnn
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A4000:1
#SBATCH --cpus-per-task=8

python run_online.py $1

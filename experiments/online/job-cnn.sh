#!/bin/bash

# take config filename from first and second arguments
# USAGE: sbatch job-cnn.sh config cnn_config

#SBATCH --job-name=online-cnn
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4


conda run -n turboawd-online --live-stream python run_online.py $1 $2

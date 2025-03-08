#!/bin/bash

# take config filename from first arg
# take cnn dir from second arg
# USAGE: sbatch job-cnn.sh config ../cnn/checkpoints/

#SBATCH --job-name=online-cnn
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

python run_online.py $1 $2
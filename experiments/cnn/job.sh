#!/bin/bash

# take config filename from first argument

#SBATCH --job-name=cnn
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4

python ../../run_cnn.py configs/$1.yaml

#!/bin/bash -l

# take config filename from first argument
config=$1

#SBATCH --job-name=cnn
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:A5000:1

python ../../run_cnn.py configs/$config
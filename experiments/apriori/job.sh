#!/bin/bash

# take same arguments as get_cnn_apriori_epoch.py

#SBATCH --job-name=offline-apriori
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

conda run -n turboawd-online --live-stream python get_cnn_apriori_epoch.py "$@"
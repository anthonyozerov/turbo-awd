#!/bin/bash
# run_ddw.py for one config

#SBATCH --job-name=ddw-init-experiments

# run the experiment
python run_ddw.py $1

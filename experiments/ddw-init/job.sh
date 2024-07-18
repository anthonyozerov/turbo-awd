#!/bin/bash -l
# run_ddw.py for each config in configs/ddw-init-experiments

#SBATCH --job-name=ddw-init-experiments

#SBATCH --array=1-45

# use $SLURM_ARRAY_TASK_ID to get the path of the nth file

configs=(configs/runs/*)
config=${configs[$SLURM_ARRAY_TASK_ID-1]}

# run the experiment
python ../../run_ddw.py $config

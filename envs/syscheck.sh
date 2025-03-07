#!/bin/bash

#SBATCH --job-name=syscheck
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:1

conda activate turbo-awd
# print which python is running
which python
# check onnxruntime and providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
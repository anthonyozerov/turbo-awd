#!/bin/bash

#SBATCH --job-name=syscheck
#SBATCH --partition=jsteinhardt
#SBATCH --gres=gpu:1

# print which python is running
conda run -n turboawd-online --live-stream which python
# check onnxruntime and providers
conda run -n turboawd-online --live-stream python -c "import onnxruntime as ort; print(ort.get_available_providers())"
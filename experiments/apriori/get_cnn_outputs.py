# script to generate and save CNN outputs for different datasets and models
# useful for running on GPU cluster

import numpy as np
import h5py
import os
from turboawd.utils import load_cnn_config, apply_cnn
import yaml

# create results directory
os.makedirs("results", exist_ok=True)

# define the CNN models to evaluate
experiments = [
    'legacy-cnn',
    '2025-02-23-c',
    '2025-02-23-d',
]
# generate paths to CNN models and their configurations
cnn_paths = [f"../cnn/trained-cnns/{e}.onnx" for e in experiments]
config_paths = [f"../cnn/configs/{e}.yaml" for e in experiments]

# Load dataset definitions from YAML
with open('data_defs.yaml', 'r') as f:
    datas = yaml.safe_load(f)

# process each dataset
for k, data in datas.items():
    print(f"Processing dataset {k}")

    # create result directory for current dataset
    os.makedirs(f"results/{k}", exist_ok=True)

    # construct full paths to data files
    input_data_path = os.path.join(data['root'], data['input'])
    output_norm_path = os.path.join(data['root'], data['norm'])
    output_data_path = os.path.join(data['root'], data['output'])

    # process each CNN model
    for i in range(len(cnn_paths)):
        cnn_path = cnn_paths[i]
        # load CNN configuration and get model name
        config, name = load_cnn_config(config_paths[i])
        # apply CNN to input data with specified parameters
        cnn_outputs = apply_cnn(cnn_path, config, input_data_path, input_centerscale=True, batch_size=128,
                                train_norm_path=output_norm_path, train_norm_key='IPI', force_gpu=True)

        # save CNN outputs to HDF5 file
        output_path = f"results/{k}/{name}.h5"
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('cnn_outputs', data=cnn_outputs)

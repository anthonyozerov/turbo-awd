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
    '2025-01-01-a',
    '2025-01-01-b',
    '2025-02-23-a',
    '2025-02-23-b',
    '2025-02-23-c',
    '2025-02-23-d',
    '2025-02-27-b',
]
multickpt_experiments = [
    '2025-03-01-b'
]

# generate paths to CNN models and their configurations
cnn_paths = {f"../cnn/trained-cnns/{e}.onnx" for e in experiments}
config_paths = {f"../cnn/configs/{e}.yaml" for e in experiments+multickpt_experiments}

# for experiments where we want to check multiple checkpoints,
# we need to get all the paths to the .onnx files.
for e in multickpt_experiments:
    config_path = f"../cnn/configs/{e}.yaml"
    # load all .onnx cnns in the directory for experiment
    fnames = os.listdir(f"../cnn/checkpoints")
    prefix = e + '_epoch'
    fnames = [f"../cnn/checkpoints/{f}" for f in fnames if f.startswith(prefix) and f.endswith('.onnx')]
    # fnames are of the form e-i.onnx, where i is the epoch number
    # sort by epoch number
    fnames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    paths = [f"../cnn/checkpoints/{f}" for f in fnames]
    cnn_paths[e] = [paths]

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
        cnn_paths_e = cnn_paths[i]
        if not isinstance(cnn_paths_e, list):
            cnn_paths_e = [cnn_paths_e]

        config, name = load_cnn_config(config_paths[i])

        for k in range(len(cnn_paths_e)):
            cnn_name = cnn_paths_e[k].split('/')[-1].split('.')[0]

            # load CNN configuration and get model name
            # model_output_norm = os.path.join(config['data']['train_dir'], config['data']['norm_file'])
            # NOTE: in transfer, using the normalization the model was trained with does not seem to work
            # as well as using the normalization constants of the data itself.
            # (when not transferring, both normalizations are the same)

            # apply CNN to input data with specified parameters
            cnn_outputs = apply_cnn(cnn_paths_e[k], config, input_data_path, input_centerscale=True, batch_size=128,
                                    train_norm_path=output_norm_path, train_norm_key='IPI', force_gpu=True)

            # save CNN outputs to HDF5 file
            output_path = f"results/{k}/{cnn_name}.h5"
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('cnn_outputs', data=cnn_outputs)

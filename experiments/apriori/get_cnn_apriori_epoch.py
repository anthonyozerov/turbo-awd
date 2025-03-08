# script to generate and save CNN outputs for different datasets and models
# useful for running on GPU cluster
# sample usage:
# python get_cnn_outputs_epoch.py -e 2025-03-01-b -n 10 -d A B -o results.json

import numpy as np
import h5py
import os
from turboawd.utils import load_cnn_config, apply_cnn
from turboawd.apriori import apriori
import yaml
import argparse
import json

a = argparse.ArgumentParser()
a.add_argument('-e', '--experiments', nargs='+', required=True, help='Experiments to evaluate')
a.add_argument('-n', '--num_images', type=int, required=False, help='Number of images to evaluate')
a.add_argument('-d', '--datasets', nargs='+', required=True, help='Datasets to evaluate')
a.add_argument('-o', '--output', required=True, help='Output file')
args = a.parse_args()

experiments = args.experiments
num_images = args.num_images
datasets = args.datasets
output = args.output

# create results directory
os.makedirs("results", exist_ok=True)

# generate paths to CNN models and their configurations
config_paths = {e: f"../cnn/configs/{e}.yaml" for e in experiments}

cnn_paths = {}

# for experiments where we want to check multiple checkpoints,
# we need to get all the paths to the .onnx files.
for e in experiments:
    config_path = config_paths[e]
    assert os.path.exists(config_path), f"Config file {config_path} does not exist"
    # load all .onnx cnns in the directory for experiment
    fnames = os.listdir(f"../cnn/checkpoints")
    prefix = e + '_epoch'
    fnames = [f"../cnn/checkpoints/{f}" for f in fnames if f.startswith(prefix) and f.endswith('.onnx')]
    # fnames are of the form e-i.onnx, where i is the epoch number
    # sort by epoch number
    fnames.sort(key=lambda x: int(x.split('epoch')[-1].split('.')[0]))
    cnn_paths[e] = fnames

# Load dataset definitions from YAML
with open('data_defs.yaml', 'r') as f:
    datas = yaml.safe_load(f)

datas = {k: v for k, v in datas.items() if k in datasets}

# process each dataset
results = {}
for k, data in datas.items():
    results[k] = {}
    print(f"Processing dataset {k}")

    # create result directory for current dataset
    os.makedirs(f"results/{k}", exist_ok=True)

    # construct full paths to data files
    input_data_path = os.path.join(data['root'], data['input'])
    output_norm_path = os.path.join(data['root'], data['norm'])
    output_data_path = os.path.join(data['root'], data['output'])

    # process each CNN model
    for e, paths in cnn_paths.items():
        results[k][e] = {}
        os.makedirs(f"results/{k}/{e}", exist_ok=True)

        config, name = load_cnn_config(config_paths[e])

        for cnn_path in cnn_paths[e]:
            epoch = int(cnn_path.split('epoch')[-1].split('.')[0])
            cnn_name = cnn_path.split('/')[-1].split('.')[0]

            # load CNN configuration and get model name
            # model_output_norm = os.path.join(config['data']['train_dir'], config['data']['norm_file'])
            # NOTE: in transfer, using the normalization the model was trained with does not seem to work
            # as well as using the normalization constants of the data itself.
            # (when not transferring, both normalizations are the same)

            # apply CNN to input data with specified parameters
            cnn_outputs = apply_cnn(cnn_path, config, input_data_path, input_centerscale=True, batch_size=128,
                                    train_norm_path=output_norm_path, train_norm_key='IPI', force_gpu=True, before=num_images)

            # apriori analysis of results
            results[k][e][epoch] = apriori(
                cnn_outputs, input_data_path, output_data_path, output_norm_path, "IPI"
            )

# save results

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(output, 'w') as f:
    json.dump(results, f, cls=NumpyEncoder)
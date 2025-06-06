# script to generate and save CNN outputs for different datasets and models
# useful for running on GPU cluster
# sample usage:
# python get_cnn_outputs_epoch.py -e 2025-03-01-b -n 10 -d A B -o result_1

import numpy as np
import h5py
import os
from turboawd.utils import load_cnn_config, apply_cnn, load_data
from turboawd.apriori import apriori
import yaml
import argparse
import json

a = argparse.ArgumentParser()
a.add_argument(
    "-e", "--experiments", nargs="+", required=True, help="Experiments to evaluate"
)
a.add_argument(
    "-n", "--num_images", type=int, required=False, help="Number of images to evaluate"
)
a.add_argument(
    "-d", "--datasets", nargs="+", required=True, help="Datasets to evaluate"
)
a.add_argument("-o", "--output", required=True, help="Output name")
a.add_argument(
    "--save_h5", action="store_true", help="Save h5 files with true Pi and CNN outputs"
)
args = a.parse_args()

experiments = args.experiments
num_images = args.num_images
datasets = args.datasets
output = args.output
save_h5 = args.save_h5

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
    prefix = e + "_epoch"
    fnames = [
        f"../cnn/checkpoints/{f}"
        for f in fnames
        if f.startswith(prefix) and f.endswith(".onnx")
    ]
    # fnames are of the form e-i.onnx, where i is the epoch number
    # sort by epoch number
    fnames.sort(key=lambda x: int(x.split("epoch")[-1].split(".")[0]))
    cnn_paths[e] = fnames

# Load dataset definitions from YAML
with open("data_defs.yaml", "r") as f:
    datas = yaml.safe_load(f)

datas = {k: v for k, v in datas.items() if k in datasets}

# process each dataset
results = {}
for setting, data in datas.items():
    results[setting] = {}
    print(f"Processing dataset {setting}")

    # create result directory for current dataset
    os.makedirs(f"results/{setting}", exist_ok=True)

    # construct full paths to data files
    input_data_path = os.path.join(data["root"], data["input"])
    output_norm_path = os.path.join(data["root"], data["norm"])
    output_data_path = os.path.join(data["root"], data["output"])
    # Load true Pi if we're saving h5 files
    true_pi = None

    if save_h5:
        save_dict = {}
        true_pi = load_data(output_data_path, ["PI"], before=num_images).squeeze()
        save_dict["true_pi"] = true_pi

    # process each CNN model
    for e, paths in cnn_paths.items():
        results[setting][e] = {}
        os.makedirs(f"results/{setting}/{e}", exist_ok=True)

        config, name = load_cnn_config(config_paths[e])

        if save_h5:
            save_dict[e] = {}

        for cnn_path in cnn_paths[e]:
            epoch = int(cnn_path.split("epoch")[-1].split(".")[0])
            cnn_name = cnn_path.split("/")[-1].split(".")[0]

            # load CNN configuration and get model name
            # model_output_norm = os.path.join(config['data']['train_dir'], config['data']['norm_file'])
            # NOTE: in transfer, using the normalization the model was trained with does not seem to work
            # as well as using the normalization constants of the data itself.
            # (when not transferring, both normalizations are the same)

            # apply CNN to input data with specified parameters
            cnn_outputs = apply_cnn(
                cnn_path,
                config,
                input_data_path,
                input_centerscale=True,
                batch_size=128,
                train_norm_path=output_norm_path,
                train_norm_key="IPI",
                force_gpu=True,
                before=num_images,
            )

            # apriori analysis of results
            results[setting][e][epoch] = apriori(
                cnn_outputs, input_data_path, output_data_path, output_norm_path, "IPI", before=num_images
            )

            if save_h5:
                save_dict[e][epoch] = cnn_outputs.squeeze()

    if save_h5:
        with h5py.File(f"results/{setting}/{output}.h5", "w") as f:
            # load omega and psi
            psiomega = load_data(input_data_path, ["psi", "omega"], before=num_images)
            f.create_dataset("psiomega", data=psiomega)

            for k, v in save_dict.items():
                if isinstance(v, dict):
                    group = f.create_group(k)
                    for epoch, data in v.items():
                        group.create_dataset(str(epoch), data=data)
                else:
                    f.create_dataset(k, data=v)

# save results

# first save as pickle
import pickle

with open(output + ".pkl", "wb") as f:
    pickle.dump(results, f)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)


with open(f"{output}.json", "w") as f:
    json.dump(results, f, cls=NumpyEncoder)

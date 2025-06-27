import os
import sys
import yaml
from turboawd.utils import load_online_config

# Load configuration file
config_path = sys.argv[1]
config = load_online_config(config_path, cnn_path_base="../cnn/checkpoints/")
with open(config_path) as f:
    config_yaml = yaml.safe_load(f)
print(config)

dir_path = config_path.split('.')[0]
os.makedirs(dir_path, exist_ok=True)

cnn_base_path = config['cnn_config']['cnn_path']
main_cnn_name = os.path.basename(cnn_base_path).strip('.onnx')
print(main_cnn_name)

cnn_dir = os.path.dirname(cnn_base_path)

files = os.listdir(cnn_dir)

for file in files:
    if file.endswith('.onnx') and file.startswith(main_cnn_name) and '_epoch' in file:
        new_config = config_yaml.copy()
        cnn_name = file.strip('.onnx')
        epoch_no = int(cnn_name.split('_epoch')[-1])
        new_config['cnn'] = cnn_name
        config_name = os.path.basename(config_path).split('.')[0]

        with open(f'{dir_path}/{config_name}_epoch{epoch_no}.yaml', 'w') as f:
            yaml.dump(new_config, f)

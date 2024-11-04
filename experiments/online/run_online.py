from py2d.Py2D_solver import Py2D_solver
import sys

sys.path.insert(0, "../../turboawd")
print(sys.path)
import os

print(os.path.abspath(sys.path[0]))
import yaml

# Load configuration file
config_path = sys.argv[1]
with open(config_path) as f:
    config = yaml.safe_load(f)
    print(config_path)

if len(sys.argv) > 2:
    cnn_config_path = sys.argv[2]
    print(cnn_config_path)
else:
    cnn_config_path = None

Py2D_solver(**config, cnn_config_path=cnn_config_path)

import sys
import yaml
from py2d.Py2D_solver import Py2D_solver
from turboawd.utils import load_online_config

# sys.path.insert(0, "../../turboawd")


# Load configuration file
config_path = sys.argv[1]
if len(sys.argv) > 2:
    cnn_path_base = sys.argv[2] # e.g. "../cnn/trained-cnns/",
else:
    cnn_path_base = None

config = load_online_config(config_path, cnn_path_base=cnn_path_base)
print(config)

save_dir = "results/" + config_path.split("/")[-1].split(".")[0]

Py2D_solver(
    Re=config["Re"],
    fkx=config["fkx"],
    fky=config["fky"],
    alpha=config["alpha"],
    beta=config["beta"],
    NX=config["NX"],
    SGSModel_string=config["SGSModel_string"],
    eddyViscosityCoeff=config["eddyViscosityCoeff"],
    dt=config["dt"],
    dealias=config["dealias"],
    saveData=config["saveData"],
    tSAVE=config["tSAVE"],
    tTotal=config["tTotal"],
    readTrue=config["readTrue"],
    ICnum=config["ICNum"],
    resumeSim=config["resumeSim"],
    full_config=config,
    save_dir=save_dir,
)

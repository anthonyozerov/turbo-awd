import yaml
import os
import pywt

config = yaml.safe_load(open("configs/base.yaml", "r"))

data_dir = "fdns-data/Re20K_kf25_NLES128"
wavelets = pywt.wavelist("db")[:6] + pywt.wavelist("bior")

os.makedirs("configs/runs/", exist_ok=True)
for which in ["psi", "omega", "pi"]:
    config["which"] = which
    if which in ["psi", "omega"]:
        filename = "FDNS Psi W_train.mat"
        key = "Psi" if which == "psi" else "W"
    elif which == "pi":
        filename = "FDNS PI_train.mat"
        key = "PI"
    config["data"]["train_file"] = filename
    config["data"]["key"] = key
    for wavelet in wavelets:
        config["dwt"]["init_wavelet"] = wavelet
        config["dwt"]["learn_dual"] = wavelet[:4] == "bior"

        identifier = f"ddw-{which}-{wavelet}"
        config["identifier"] = identifier
        config["wandb"]["name"] = identifier
        config["checkpoint"]["dirpath"] = f"checkpoints/{identifier}"
        config["checkpoint"]["filename"] = "{epoch:02d}.ckpt"
        # write the yaml file
        with open(f"configs/runs/{identifier}.yaml", "w") as f:
            yaml.dump(config, f, sort_keys=False)

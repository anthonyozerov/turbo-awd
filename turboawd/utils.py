import os
import yaml
import h5py
import numpy as np
from scipy.io import loadmat


def load_cnn_config(config_path):
    """
    Load the configuration files for a CNN experiment.

    Parameters
    ----------
    config_path : str
        Path to the main configuration file.

    Returns
    -------
    config : dict
        Dictionary containing all configuration options.
    name : str
        Name of the CNN.
    """

    assert os.path.exists(config_path), f"Invalid config path: {config_path}"
    config_meta = yaml.safe_load(open(config_path, "r"))

    # get the names of the other config files
    architecture = config_meta["architecture"]
    channels = config_meta["channels"]
    data = config_meta["data"]
    training = config_meta["training"]

    # load the other config files
    config_architecture = yaml.safe_load(
        open(f"configs/architecture/{architecture}.yaml", "r")
    )
    config_channels = yaml.safe_load(open(f"configs/channels/{channels}.yaml", "r"))
    config_data = yaml.safe_load(open(f"configs/data/{data}.yaml", "r"))
    config_training = yaml.safe_load(open(f"configs/training/{training}.yaml", "r"))

    # combine them all into one config
    config = {
        **config_architecture,
        **config_channels,
        **config_data,
        **config_training,
    }

    name = config_path.split("/")[-1].split(".")[0]

    return config, name


def normalize(data, norm_path, norm_keys, sd_only=False):
    assert os.path.exists(norm_path), f"Invalid normalization path: {norm_path}"
    nchannels = data.shape[1]
    assert nchannels == len(norm_keys)

    normalization = loadmat(norm_path)

    for i in range(nchannels):
        mean = normalization["MEAN_" + norm_keys[i]][0][0]
        sdev = normalization["SDEV_" + norm_keys[i]][0][0]

        if sd_only:
            data[:,i,:,:] = data[:,i,:,:] / sdev
        else:
            data[:,i,:,:] = (data[:,i,:,:] - mean) / sdev

    return data


def denormalize(data, norm_path, norm_keys):
    assert os.path.exists(norm_path), f"Invalid normalization path: {norm_path}"
    nchannels = data.shape[1]
    assert nchannels == len(norm_keys)

    normalization = loadmat(norm_path)

    for i in range(nchannels):
        mean = normalization["MEAN_" + norm_keys[i]][0][0]
        sdev = normalization["SDEV_" + norm_keys[i]][0][0]

        data[:,i,:,:] = data[:,i,:,:] * sdev + mean

    return data


def load_data(
    path,
    keys,
    centerscale=False,
    norm_path=None,
    norm_keys=None,
    denorm=False,
    before=None,
    after=None,
    tensor=False,
):
    """
    Load fluid data from a .h5 file, optionally normalizing/denormalizing it. The .h5
    file has keys for channels, and every channel has shape [Nlon, Nlat, N].
    Data is returned in the shape [N, channels, Nlon, Nlat], suitable for use in PyTorch
    or ONNX.

    Parameters
    ----------
    path : str
        Path to the .h5 file.
    keys : list of str
        List of keys to load from the .h5 file.
    centerscale : bool, optional
        Whether to center and scale the data, by default False. Mutually exclusive with normalizing
        using a normalization file.
    norm_path : str, optional
        Path to the normalization file, by default None. If norm_path is specified and
        denorm is False, the data will be normalized using the given mean and std for each channel
        in the normalization file. If denorm is True, the data will be denormalized.
    norm_keys : list of str, optional
        List of keys to normalize, by default None. Must be specified if norm_path is not None.
        Keys should be in the same order as the keys parameter.
    denorm : bool, optional
        Whether to denormalize the data, by default False.
    before : int, optional
        Only keep samples up to this index, by default None.
    after : int, optional
        Only keep samples after this index, by default None.
    tensor : bool, optional
        Whether to convert the data to a PyTorch tensor, by default False.

    Returns
    -------
    np.ndarray or torch.Tensor
        Data loaded from the .h5 file.
    """

    if norm_path is not None:
        assert norm_keys is not None
        assert len(norm_keys) == len(keys)
    assert before is None or after is None
    assert norm_path is None or centerscale is False

    assert os.path.exists(path), f"Invalid data path: {path}"

    with h5py.File(path, "r") as f:
        data = [np.array(f[key], np.float32) for key in keys]
    data = np.array(data)
    # data is [channels, Nlon, Nlat, N]
    data = np.moveaxis(data, -1, 0)
    # data is [N, channels, Nlon, Nlat]

    if centerscale:
        # center and scale each channel
        means = np.mean(data, axis=(0, 2, 3), keepdims=True)

        data = (data - np.mean(data, axis=(0, 2, 3), keepdims=True)) / np.std(
            data, axis=(0, 2, 3), keepdims=True
        )

        # check that each channel has mean 0 and std 1
        assert np.allclose(np.mean(data, axis=(0, 2, 3)), 0, atol=1e-6)
        assert np.allclose(np.std(data, axis=(0, 2, 3)), 1, atol=1e-2)

    if before is not None:
        data = data[:before, :, :, :]
    if after is not None:
        data = data[after:, :, :, :]

    if norm_path is not None:
        if denorm:
            data = denormalize(data, norm_path, norm_keys)
        else:
            data = normalize(data, norm_path, norm_keys)

    if tensor:
        import torch

        data = torch.from_numpy(data)

    return data

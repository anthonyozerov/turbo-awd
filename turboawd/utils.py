import os
import yaml
import h5py
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt

def plot_field(field):
    lim = np.max(np.abs(field))
    plt.imshow(field, cmap='bwr', vmin=-lim, vmax=lim)
    plt.colorbar()

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
    root_path = "/".join(config_path.split("/")[:-1])
    config_architecture = yaml.safe_load(
        open(f"{root_path}/architecture/{architecture}.yaml", "r")
    )
    config_channels = yaml.safe_load(open(f"{root_path}/channels/{channels}.yaml", "r"))
    config_data = yaml.safe_load(open(f"{root_path}/data/{data}.yaml", "r"))
    config_training = yaml.safe_load(open(f"{root_path}/training/{training}.yaml", "r"))

    # combine them all into one config
    config = {
        **config_meta,
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

    data_norm = np.zeros_like(data)

    normalization = loadmat(norm_path)

    for i in range(nchannels):
        mean = normalization["MEAN_" + norm_keys[i]][0][0]
        sdev = normalization["SDEV_" + norm_keys[i]][0][0]

        if sd_only:
            data_norm[:, i, :, :] = data[:, i, :, :] / sdev
        else:
            data_norm[:, i, :, :] = (data[:, i, :, :] - mean) / sdev

    return data_norm


def denormalize(data, norm_path, norm_keys, sd_only=False):
    assert os.path.exists(norm_path), f"Invalid normalization path: {norm_path}"
    nchannels = data.shape[1]
    assert nchannels == len(norm_keys)

    # empty array with same shape
    data_denorm = np.zeros(data.shape)

    normalization = loadmat(norm_path)

    for i in range(nchannels):
        mean = normalization["MEAN_" + norm_keys[i]][0][0]
        sdev = normalization["SDEV_" + norm_keys[i]][0][0]
        if sd_only:
            data_denorm[:, i, :, :] = data[:, i, :, :] * sdev
        else:
            data_denorm[:, i, :, :] = data[:, i, :, :] * sdev + mean

    return data_denorm


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

    try:
        with h5py.File(path, "r") as f:
            data = [np.array(f[key], np.float32) for key in keys]
    except OSError as e:
        print(e)
        print('loading data using scipy loadmat')
        mat = loadmat(path)
        data = [np.array(mat[key], np.float32) for key in keys]

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


def apply_cnn(
    onnx_path,
    config,
    input_data_path,
    input_centerscale=False,
    input_norm_path=None,
    train_norm_path=None,
    train_norm_key=None,
    finaloutput_denorm_path=None,
    finaloutput_denorm_key=None,
    batch_size=1,
    reorder=None,
):
    """
    Apply a CNN model to input data, using ONNX runtime.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model.
    config : dict
        Configuration of the CNN.
    input_data_path : str
        Path to the input data.
    input_centerscale : bool, optional
        Whether to center and scale the input data, by default False.
    input_norm_path : str, optional
        Path to the normalization file for the input data, by default None.
        If not None, each channel in the input data will be normalized using
        the mean and std given in the normalization file.
    train_norm_path : str, optional
        Path to the normalization file used during training, by default None.
        Required if the model is a residual model.
    train_norm_key : str, optional
        Key to normalize the training data, by default None.
        Required if the model is a residual model.
    finaloutput_denorm_path : str, optional
        Path to the normalization file for the final output data, by default None.
        If not None, the final output data will be denormalized using the mean and std
        given in the normalization file.
    finaloutput_denorm_key : str, optional
        Key to denormalize the final output data, by default None.
    batch_size : int, optional
        Batch size to use when running the model, by default 1.
    reorder : list of int, optional
        List of axes to reorder the input to the model and output from it.
        By default the CNN must input and output NCHW.

    Returns
    -------
    np.ndarray
        Output data from the CNN model.
    """

    import onnxruntime as rt


    if "nchw_map" in config:
        reorder = config["nchw_map"]
    if reorder is not None:
        assert len(reorder) == 4

    assert os.path.exists(onnx_path), f"Invalid ONNX path: {onnx_path}"

    if 'residual' in config:
        assert train_norm_path is not None
        assert train_norm_key is not None

    # load the ONNX model
    sess = rt.InferenceSession(onnx_path)

    # load the input data, normalizing the channels as appropriate using
    # the provided normalization file
    norm_keys = None if input_norm_path is None else config["input_norm_keys"]
    input_data = load_data(
        input_data_path,
        config["input_channels"],
        centerscale=input_centerscale,
        norm_path=input_norm_path,
        norm_keys=norm_keys,
        tensor=False,
    )

    rt_inputs = {sess.get_inputs()[0].name: input_data}
    # print shape of model inputs
    print(sess.get_inputs()[0].shape)

    # run the model in batches, assuming batch size does not divide the number of samples
    output_data = []
    for i in tqdm(range(0, input_data.shape[0], batch_size)):
        start = i
        end = min(i + batch_size, input_data.shape[0])
        forinput = input_data[start:end]
        if reorder is not None:
            # reorder the axes of the input data
            forinput = np.moveaxis(forinput, [0, 1, 2, 3], reorder)
        rt_outs = sess.run(None, {sess.get_inputs()[0].name: forinput})
        output_data.append(rt_outs[0])

    output_data = np.concatenate(output_data, axis=0)
    if reorder is not None:
        output_data = np.moveaxis(output_data, reorder, [0, 1, 2, 3])

    # if the model is a residual model, we denormalize its output scale
    # according to the normalization constant used in training,
    # add the residual, then renormalize it (both scale and location) using
    # the constants from training.
    if "residual" in config:
        output_data = denormalize(
            output_data, train_norm_path, [train_norm_key], sd_only=True
        )
        residual = load_data(input_data_path, [config["residual"]])
        output_data += residual
        output_data = normalize(
            output_data, train_norm_path, [train_norm_key], sd_only=False
        )

    # apply denormalization to final output of CNN (model+residual)
    if finaloutput_denorm_path is not None:
        output_data = denormalize(
            output_data, finaloutput_denorm_path, [finaloutput_denorm_key]
        )

    return output_data


# force an ONNX model to accept arbitrary batch size

def change_onnx_dims(model_path):
    import onnx
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = "batch"

    # The following code changes the first dimension of every input to be batch-dim
    # Note that this requires all inputs to have the same batch_dim
    model = onnx.load(model_path)
    inputs = model.graph.input

    for input in inputs:

        # Checks omitted. This assumes that all inputs are tensors and have a shape with first dim.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        dim1.dim_param = sym_batch_dim

    # make the outputs also have the same batch dim
    outputs = model.graph.output
    for output in outputs:
        dim1 = output.type.tensor_type.shape.dim[0]
        dim1.dim_param = sym_batch_dim

    onnx.save(model, model_path)


def load_online_config(config_path,
                       cnn_config_base="../cnn/configs/",
                       cnn_path_base="../cnn/trained-cnns/",
                       norm_path=None):

    assert os.path.exists(config_path), f"Invalid config path: {config_path}"
    config_meta = yaml.safe_load(open(config_path, "r"))


    physics = config_meta["physics"]
    resolution = config_meta["resolution"]
    sgsmodel = config_meta["sgsmodel"]
    boilerplate = config_meta["boilerplate"]

    config_physics = yaml.safe_load(open(f"../online/configs/physics/{physics}.yaml", "r"))
    config_resolution = yaml.safe_load(open(f"../online/configs/resolution/{resolution}.yaml", "r"))
    config_sgsmodel = yaml.safe_load(open(f"../online/configs/sgsmodel/{sgsmodel}.yaml", "r"))
    config_boilerplate = yaml.safe_load(open(f"../online/configs/boilerplate/{boilerplate}.yaml", "r"))

    config = {
        **config_meta,
        **config_physics,
        **config_resolution,
        **config_sgsmodel,
        **config_boilerplate,
    }

    if config["SGSModel_string"] == 'CNN':
        assert "cnn" in config_meta

    if "cnn" in config_meta:
        assert config['SGSModel_string'] == 'CNN'
        assert 'input_stepnorm' in config

        cnn_config_name = config_meta["cnn"]
        cnn_config_path = f"{cnn_config_base}{cnn_config_name}.yaml"
        print(cnn_config_path)
        cnn_config = load_cnn_config(cnn_config_path)[0]

        if norm_path is None:
            norm_path = f"{cnn_config['data']['train_dir']}/{cnn_config['data']['norm_file']}"

        assert os.path.exists(norm_path), f"Invalid normalization path: {norm_path}"
        config['norm_path'] = norm_path



        cnn_path = f"{cnn_path_base}{cnn_config_name}.onnx"
        assert os.path.exists(cnn_path), f"Invalid CNN path: {cnn_path}"

        config["cnn_config"] = cnn_config
        config["cnn_path"] = cnn_path

    return config
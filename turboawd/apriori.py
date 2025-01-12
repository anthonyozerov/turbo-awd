import numpy as np
from turboawd.utils import load_data, denormalize
from py2d.dealias import multiply_dealias
from py2d.initialize import initialize_wavenumbers_rfft2
from py2d.convert import Omega2Psi
from py2d.spectra import (
    energyTransfer_spectra,
    enstrophyTransfer_spectra,
)
import onnxruntime as rt


def get_energytransfer(psi, pi, dealias):
    n = psi.shape[0]
    arr = [
        np.mean(multiply_dealias(psi[i, 0, :, :], pi[i, 0, :, :], dealias=dealias))
        for i in range(n)
    ]
    return np.array(arr)


def get_enstrophytransfer(omega, pi, dealias):
    n = omega.shape[0]
    arr = [
        np.mean(multiply_dealias(omega[i, 0, :, :], pi[i, 0, :, :], dealias=dealias))
        for i in range(n)
    ]
    return np.array(arr)


Kx, Ky, Kabs, Ks, invKsq = initialize_wavenumbers_rfft2(
    128, 128, 2 * np.pi, 2 * np.pi, INDEXING="ij"
)


def get_enstrophytransfer_spectrum(omega, pi):
    assert omega.ndim == 2
    assert pi.ndim == 2

    Z, _ = enstrophyTransfer_spectra(
        Kx, Ky, Omega=omega, PiOmega=pi, spectral=False, method="PiOmega"
    )
    return Z


def get_energytransfer_spectrum(psi, pi):
    assert psi.ndim == 2
    assert pi.ndim == 2

    Z, _ = energyTransfer_spectra(
        Kx, Ky, Psi=psi, PiOmega=pi, spectral=False, method="PiOmega"
    )
    return Z


def apriori(
    pi_model, input_data_path, output_data_path, output_denorm_path, output_norm_key, wavelet_path=None
):
    """
    Calculate a priori performance metrics for the model output.

    Parameters
    ----------
    pi_model : np.ndarray
        Normalized model output for given inputs
    input_data_path : str
        Path to the input data (used to generate the model output)
    output_data_path : str
        Path to the true output data
    output_denorm_path : str
        Path to the normalization file for the output data
    output_norm_key : str
        Key to denormalize the output data
    """
    pi_true = load_data(output_data_path, ["PI"])
    psi_true = load_data(input_data_path, ["psi"])
    omega_true = load_data(input_data_path, ["omega"])

    assert pi_model.shape == pi_true.shape
    n = pi_model.shape[0]

    # MSE and MAE
    mse = np.mean((pi_model - pi_true) ** 2)
    mae = np.mean(np.abs(pi_model - pi_true))

    pi_model_denorm = denormalize(pi_model, output_denorm_path, [output_norm_key])
    pi_true_denorm = denormalize(pi_true, output_denorm_path, [output_norm_key])

    dealias = False

    # Energy and enstrophy transfer
    energytransfer_true = get_energytransfer(psi_true, pi_true_denorm, dealias)
    enstrophytransfer_true = get_enstrophytransfer(omega_true, pi_true_denorm, dealias)

    energytransfer_model = get_energytransfer(psi_true, pi_model_denorm, dealias)
    enstrophytransfer_model = get_enstrophytransfer(
        omega_true, pi_model_denorm, dealias
    )

    energytransfer_mse = np.mean((energytransfer_model - energytransfer_true) ** 2)
    energytransfer_mae = np.mean(np.abs(energytransfer_model - energytransfer_true))

    enstrophytransfer_mse = np.mean(
        (enstrophytransfer_model - enstrophytransfer_true) ** 2
    )
    enstrophytransfer_mae = np.mean(
        np.abs(enstrophytransfer_model - enstrophytransfer_true)
    )

    # Energy and enstrophy transfer spectra
    enstrophy_transfer_spectra = [
        get_enstrophytransfer_spectrum(
            omega_true[i, 0, :, :], pi_model_denorm[i, 0, :, :]
        )
        for i in range(n)
    ]
    enstrophytransfer_spectrum = np.mean(np.array(enstrophy_transfer_spectra), axis=0)

    energy_transfer_spectra = [
        get_energytransfer_spectrum(psi_true[i, 0, :, :], pi_model_denorm[i, 0, :, :])
        for i in range(n)
    ]
    energytransfer_spectrum = np.mean(np.array(energy_transfer_spectra), axis=0)

    # look at the output in wavelet space
    if wavelet_path is not None:
        sess = rt.InferenceSession(wavelet_path)
        rt_inputs = {sess.get_inputs()[0].name: pi_model}
        pi_model_wavelet = sess.run(None, rt_inputs)
    else:
        pi_model_wavelet = None

    return dict(
        mse=mse,
        mae=mae,
        energytransfer_mse=energytransfer_mse,
        energytransfer_mae=energytransfer_mae,
        enstrophytransfer_mse=enstrophytransfer_mse,
        enstrophytransfer_mae=enstrophytransfer_mae,
        enstrophytransfer_spectrum=enstrophytransfer_spectrum,
        energytransfer_spectrum=energytransfer_spectrum,
        pi_model_wavelet=pi_model_wavelet,
    )

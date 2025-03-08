from scipy.io import loadmat
import os
import numpy as np
from py2d.convert import Omega2Psi
from turboawd.apriori import (
    get_energytransfer_spectrum,
    get_enstrophytransfer_spectrum,
    get_enstrophy_spectrum,
    get_tke_spectrum,
    Kx,
    Ky,
    invKsq,
)


def aposteriori(omegas, pis=None, psis=None):
    """
    Calculate a posteriori performance metrics from online simulation data.

    Parameters:
    ----------
    omegas : np.ndarray
        Array of vorticity fields
    pis : np.ndarray
        Array of Pi fields
    psis : np.ndarray, optional
        Array of Psi fields. If not provided, it will be computed from omegas.

    Returns:
    -------
    dict
        Dictionary containing computed spectra
    """
    n = omegas.shape[0]

    # Calculate psi fields from omega iteratively if not provided
    if psis is None:
        psis = []
        for i in range(n):
            psi = Omega2Psi(omegas[i, :, :], invKsq, spectral=False)
            psis.append(psi)
        psis = np.array(psis)

    # Energy and enstrophy transfer spectra
    if pis is not None:
        enstrophy_transfer_spectra = [
            get_enstrophytransfer_spectrum(omegas[i, :, :], pis[i, :, :]) for i in range(n)
        ]
        enstrophytransfer_spectrum = np.mean(np.array(enstrophy_transfer_spectra), axis=0)

        energy_transfer_spectra = [
            get_energytransfer_spectrum(psis[i, :, :], pis[i, :, :]) for i in range(n)
        ]
        energytransfer_spectrum = np.mean(np.array(energy_transfer_spectra), axis=0)

    enstrophy_spectra = [get_enstrophy_spectrum(omegas[i])[0] for i in range(n)]
    enstrophy_spectrum = np.mean(np.array(enstrophy_spectra), axis=0)
    tke_spectra = [get_tke_spectrum(omegas[i])[0] for i in range(n)]
    tke_spectrum = np.mean(np.array(tke_spectra), axis=0)

    result = dict(
        enstrophy_spectrum=enstrophy_spectrum,
        tke_spectrum=tke_spectrum,
    )
    if pis is not None:
        result.update(
            enstrophytransfer_spectrum=enstrophytransfer_spectrum,
            energytransfer_spectrum=energytransfer_spectrum,
        )
    return result


def load_online_data(path, last=None):
    """
    Load online data from Py2D output in the specified directory.

    The function expects the .mat files to be named in a sequential numeric
    order (e.g., 1.mat, 2.mat, ..., n.mat) and loads them in this order.

    Parameters:
    path (str): The directory path where the .mat files are located.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - omegas (np.ndarray): An array of 'Omega' values loaded from the .mat files.
        - pis (np.ndarray): An array of 'PiOmega' values loaded from the .mat files.

    Raises:
    FileNotFoundError: If any expected .mat file is missing in the sequence.
    """

    data_path = os.path.join(path, "data")
    assert os.path.exists(data_path), f"Data directory not found in {path}"

    parameters_path = os.path.join(path, "parameters.txt")
    assert os.path.exists(parameters_path), f"parameters.txt file not found in {path}"

    tsave = None
    with open(parameters_path, "r") as f:
        for line in f:
            if "tSAVE" in line:
                tsave = float(line.split(":")[1].strip())
                break

    assert tsave is not None, "tSAVE not found in parameters.txt"


    # Get a list of all .mat files in the specified directory
    filenames = [f for f in os.listdir(data_path) if f.endswith(".mat")]

    # Determine the highest index number from the filenames
    max_idx = max([int(f.split(".")[0]) for f in filenames])

    # Initialize lists to store 'Omega' and 'PiOmega' values
    omegas = []
    pis = []

    if last is not None:
        start = max_idx + 1 - last
    else:
        start = 1
    # Loop through each index from 1 to max_idx
    for i in range(start, max_idx + 1):
        # Construct the expected filename and check if it exists
        if not os.path.exists(os.path.join(data_path, f"{i}.mat")):
            raise FileNotFoundError(f"Missing file {i}.mat in {path}")

        # Load the .mat file
        data = loadmat(os.path.join(data_path, f"{i}.mat"))

        # Append the 'Omega' and 'PiOmega' values to the respective lists
        omegas.append(data["Omega"])
        if 'PiOmega' in data:
            pis.append(data["PiOmega"])

    times = np.arange(start, max_idx + 1) * tsave

    # Convert the lists to numpy arrays and return them as a tuple
    if len(pis) == 0:
        return np.array(omegas), None, times
    else:
        assert len(omegas) == len(pis)
        return np.array(omegas), np.array(pis), times


def get_divergence(times1, field1, times2, field2):

    # get the intersection of the two series of times
    times = np.intersect1d(times1, times2)
    # subset omega_sgs and omega_fdns to the intersection of the two series of times
    idx1 = np.where(np.isin(times1, times))[0]
    field1_sub = field1[idx1]
    idx2 = np.where(np.isin(times2, times))[0]
    field2_sub = field2[idx2]

    assert field1_sub.shape == field2_sub.shape

    # calculate mean squared error between the vorticity in FDNS and SGS
    mse = np.mean((field1_sub - field2_sub)**2, axis=(1,2))

    # calculate variance of the SGS vorticity
    var1 = np.var(field1_sub, axis=(1,2))
    var2 = np.var(field2_sub, axis=(1,2))

    return times, mse, var1, var2
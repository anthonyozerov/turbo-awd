# script to plot the divergence in time
# between the vorticity fields from FDNS and SGS

import yaml
import os
import numpy as np

from turboawd.aposteriori import load_online_data, aposteriori
import matplotlib.pyplot as plt

# map IC identifiers to directories containing FDNS data
# started from those ICs
data_fdns = {
    'A0': '2025-03-01-d-filtered',
    'C0': '2025-03-01-f-filtered'
}

for ic, loc in data_fdns.items():

    # load FDNS data
    omega_fdns, _ = load_online_data('../online/results/'+loc+'/data')
    print(len(omega_fdns))

    # calculate variance of the FDNS data across spatial dimensions
    var_fdns = np.var(omega_fdns, axis=(1,2))
    ymax = max(var_fdns)*3

    # read yaml config files in ../online/configs
    # and get those with ICNum == ic
    configs = {}
    for fn in os.listdir('../online/configs'):
        if fn.endswith('.yaml'):
            with open(f'../online/configs/{fn}', 'r') as f:
                config = yaml.safe_load(f)

                # filter for online experiment configs matching the current initial condition and having an SGS model
                if 'ICNum' in config and config['ICNum'] == ic and config['sgsmodel'] != 'nosgs':
                    configs[fn.split('.')[0]] = config

    for k, v in configs.items():
        print(k)
        # load the vorticity data from the SGS experiment
        omega_sgs, _ = load_online_data('../online/results/'+k+'/data')
        dt = v['tSAVE']

        print(omega_sgs.shape)
        print(omega_fdns.shape)

        # determine minimum common length of time series data
        minlen = min(omega_fdns.shape[0], omega_sgs.shape[0])
        print(minlen)

        # truncate both datasets to the same length for comparison
        omega_fdns_sub = omega_fdns[:minlen]
        omega_sgs_sub = omega_sgs[:minlen]

        # create time axis for plotting
        x = np.linspace(0, dt*minlen, minlen)

        # calculate mean squared error between the vorticity in FDNS and SGS
        mses = np.mean((omega_fdns_sub - omega_sgs_sub)**2, axis=(1,2))

        # calculate variance of the SGS vorticity
        vars = np.var(omega_sgs_sub, axis=(1,2))

        # plot MSE and variance
        plt.plot(x, mses, label=k)
        plt.plot(x, vars, label=k+' var', linestyle='--')
        plt.ylim([0, ymax])

    # create time axis for full FDNS data
    x_fdns = np.linspace(0, dt*len(omega_fdns), len(omega_fdns))
    # plot FDNS variance as reference
    plt.plot(x_fdns, var_fdns, label='FDNS', linestyle='--')

    plt.legend()
    plt.show()
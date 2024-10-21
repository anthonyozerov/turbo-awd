from py2d.Py2D_solver import Py2D_solver
import sys
sys.path.insert(0, 'turboawd')
print(sys.path)
import os
print(os.path.abspath(sys.path[0]))

Py2D_solver(Re=20e3, # Reynolds number
               fkx=25, # Forcing wavenumber in x
               fky=25, # Forcing wavenumber in y
               alpha=0.1, # Rayleigh drag coefficient
               beta=0, # Coriolis parameter
               NX=128, # Number of grid points in x and y '32', '64', '128', '256', '512'
               SGSModel_string='CNN', # SGS model to use 'NoSGS', 'SMAG', 'DSMAG', 'LEITH', 'DLEITH', 'PiOmegaGM2', 'PiOmegaGM4', 'PiOmegaGM6'
               eddyViscosityCoeff=0, # Coefficient for eddy viscosity models: SMAG and LEITH
               dt=1e-5, # Time step
               saveData=True, # Save data
               dealias=False, # dealias
               tSAVE=1.0, # Time interval to save data
               tTotal=1, # Total time of simulation
               readTrue=False,
               ICnum='fdns_ic_0_train', # Initial condition number: Choose between 1 to 20
               resumeSim=False, # tart new simulation (False) or resume simulation (True)
               )

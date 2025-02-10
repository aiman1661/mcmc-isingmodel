"""
Full temperature range simulation, without visualisation. 
Parameters are fixed.
For data collection which includes all observables.

author: s2229553
"""

import sys
import numpy as np
from numpy import random
import IsingModel as IM
import utils

def main():
    # Read inputs from command lines
    if len(sys.argv) != 4 :
        print("You left out the name of the files when running.")
        print("In command line, run like this instead:")
        print(f"% nohup python {sys.argv[0]} <lattice side length> <dynamics 'G' or 'K'> <output .npy file name> > output.txt &")
        print("For example:")
        print("% nohup python IsingMeasurement.py 50 G measurement_test > output.txt &")
        sys.exit(1)
    else:
        n = int(sys.argv[1]) 
        dynamics = str(sys.argv[2])
        outfile = str(sys.argv[3])

    # system parameters, fixed!
    N = n**2        # number of sites (square lattice)
    nstep = 10000   # the 'time' for this dynamical evolution

    temperature_array = np.round(np.arange(1,3.1,0.1), decimals=1)

    # initialise lattice sites
    rng = random.default_rng()
    # all aligned (spin down) initial lattice (ground state for Glauber)
    if dynamics == 'G':
        lattice = -np.ones((n,n))
    # half up, half down configuration (ground state for Kawasaki)
    if dynamics == 'K':    
        lattice = np.concatenate((np.ones((int(n/2),n)), -np.ones((int(n/2),n))), axis=0)

    system = IM.IsingLattice(lattice)             # initial system
    energy_lattice = system.return_energy_total() # initial lattice energy

    if dynamics == 'G':
        energy_exp_list = []
        magnetisation_exp_list = []
        heat_capacity_list = []
        chi_list = []
        sigma_c_list = []

        print('Starting simulation with Glauber dynamics...\n')
        for T in temperature_array:
            energy_list = []
            magnetisation_list = []

            print(f'Calculating T = {T} ...')
            for sweep in range(nstep):
                for _ in range(n**2): 
                    glauber_site, delta_E = system.perform_glauber_step() # glauber update
                    metro_bool, delta_E = system.perform_metropolis(delta_E, T) # metropolis test

                    i, j = glauber_site

                    # results of metropolis
                    if metro_bool == True:
                        lattice[i][j] = - lattice[i][j] # allow flip
                        system = IM.IsingLattice(lattice) # update system
                        energy_lattice += delta_E # update energy

                #occasionally update measurements
                if (sweep%10) == 0:
                    # update measurements
                    energy_list.append(energy_lattice)
                    magnetisation_list.append(system.return_magnetisation())
            
            # manipulating data
            energy_list_red = energy_list[10:] # bin first 100 sweeps, equilibration time
            magnetisation_list_red = magnetisation_list[10:] # bin first 100 sweeps, equilibration time
            energy_squared = np.array([energy_list_red])**2
            magnetisation_squared = np.array([magnetisation_list_red])**2
            sample_size = len(energy_list_red)

            # calculating observables
            energy_exp = np.sum(energy_list_red)/sample_size
            energy2_exp = np.sum(energy_squared)/sample_size
            magnetisation_exp = np.sum(magnetisation_list_red)/sample_size
            magnetisation2_exp = np.sum(magnetisation_squared)/sample_size
            c_v = utils.heat_capacity(energy_exp, energy2_exp, N, T)
            chi = utils.susceptibility(magnetisation_exp, magnetisation2_exp, N, T)
            sigma_c = utils.resampling(c_v, energy_list_red, energy_squared, N, T)
            
            print(f'Energy : {energy_exp}')
            print(f'Magnetisation : {magnetisation_exp}')
            print(f'C_v : {c_v}')
            print(f'Chi : {chi}')
            print(f'Error on C_v : {sigma_c} \n')

            energy_exp_list.append(energy_exp)
            magnetisation_exp_list.append(magnetisation_exp)
            heat_capacity_list.append(c_v)
            chi_list.append(chi)
            sigma_c_list.append(sigma_c)

    if dynamics == 'K':
        energy_exp_list = []
        magnetisation_exp_list = []
        heat_capacity_list = []
        chi_list = []
        sigma_c_list = []

        print('Starting simulation with Kawasaki dynamics...\n')
        for T in temperature_array:
            energy_list = []
            magnetisation_list = []

            print(f'Calculating T = {T} ...')
            for sweep in range(nstep):
                for _ in range(n**2): 
                    kawasaki_site1, kawasaki_site2, delta_E = system.perform_kawasaki_step() # kawasaki update
                    metro_bool, delta_E = system.perform_metropolis(delta_E, T) # metropolis test

                    i1, j1 = kawasaki_site1
                    i2, j2 = kawasaki_site2

                    if metro_bool == True:
                        lattice[i1][j1], lattice[i2][j2] = lattice[i2][j2], lattice[i1][j1] # allow swap
                        system = IM.IsingLattice(lattice) # update system
                        energy_lattice += delta_E # update energy

                #occasionally update measurements
                if (sweep%10) == 0:
                    # update measurements
                    energy_list.append(energy_lattice)
                    magnetisation_list.append(system.return_magnetisation())
            
            # manipulating data
            energy_list_red = energy_list[10:] # bin first 100 sweeps, equilibration time
            magnetisation_list_red = magnetisation_list[10:] # bin first 100 sweeps, equilibration time
            energy_squared = np.array([energy_list_red])**2
            magnetisation_squared = np.array([magnetisation_list_red])**2
            sample_size = len(energy_list_red)

            # calculating observables
            energy_exp = np.sum(energy_list_red)/sample_size
            energy2_exp = np.sum(energy_squared)/sample_size
            magnetisation_exp = np.sum(magnetisation_list_red)/sample_size
            magnetisation2_exp = np.sum(magnetisation_squared)/sample_size
            c_v = utils.heat_capacity(energy_exp, energy2_exp, N, T)
            chi = utils.susceptibility(magnetisation_exp, magnetisation2_exp, N, T)
            sigma_c = utils.resampling(c_v, energy_list_red, energy_squared, N, T)
            
            print(f'Energy : {energy_exp}')
            print(f'Magnetisation : {magnetisation_exp}')
            print(f'C_v : {c_v}')
            print(f'Chi : {chi}')
            print(f'Error on C_v : {sigma_c} \n')

            energy_exp_list.append(energy_exp)
            magnetisation_exp_list.append(magnetisation_exp)
            heat_capacity_list.append(c_v)
            chi_list.append(chi)
            sigma_c_list.append(sigma_c)

    data = {"temperature": np.array(temperature_array),
            "energy": np.array(energy_exp_list),
            "magnetisation": np.array(magnetisation_exp_list),
            "c_v": np.array(heat_capacity_list),
            "chi": np.array(chi_list),
            "sigma_c": np.array(sigma_c_list)}

    np.save(outfile, data)

    print('Job Done! :)')
    print(f'Check directory for {outfile}.npy file. Data analysis can be done by using np.load. \n')

if __name__ == '__main__':
    main()
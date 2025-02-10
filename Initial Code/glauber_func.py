"""
Simulating 2d Ising model with periodic BC using Markov Chain Monte Carlo,
based on Glauber (w. Metropolis test) dynamics to sample the equilibrium state.

author: s2229553
"""

import numpy as np
import matplotlib.pyplot as plt

# random number generator; don't remove!
rng = np.random.default_rng() 

def nn_helper(n ,site):
    'helper function to identify sites of nearest neigbours for a given spin at site=(i,j)'
    i = site[0]
    j = site[1]

    left_nn = np.array([i,(j-1)%n])
    right_nn = np.array([i,(j+1)%n])
    top_nn = np.array([(i-1)%n,j])
    bottom_nn = np.array([(i+1)%n,j])

    return np.array([left_nn, bottom_nn, right_nn, top_nn])

def energy_spin_helper(n, site, lattice):
    'helper function to calculate energy of system (for chosen spin site), up to nearest neighbours'
    # first sort spins according to chosen spin and nearest neighbours
    nn_list = nn_helper(n, site)
    spin_current = lattice[site[0]][site[1]]

    energy = 0

    for nn in nn_list:
        energy -= spin_current * lattice[nn[0]][nn[1]]

    return energy

def energy_site(n, site, lattice):
    'calculate energy of site, up to nearest neighbours'
    i = site[0]
    j = site[1]

    energy = - lattice[i][j] * (lattice[i][(j-1)%n] + lattice[i][(j+1)%n] + lattice[(i-1)%n][j] + lattice[(i+1)%n][j])

    return energy

def energy_total(n, lattice): # can be made more efficient
    'helper function to calculate total energy of lattice'
    energy = 0
    for i in range(n):
        for j in range(n):
            energy += energy_site(n, [i,j], lattice)

    return energy/2 # double counting

def delta_energy(energy_mu, energy_nu):
    'returns energy difference between states'
    energy = energy_nu - energy_mu
    return energy

def metropolis_test(n, lattice, T):
    'performs the metropolis test, returns bool indicating pass/fail and appropriate lattice'
    bool_test = True

    # choosing spin to attempt flip
    l_i = rng.choice(n)
    l_j = rng.choice(n)
    site = [l_i, l_j]

    # attempting flip
    lattice_attempt = np.zeros((n,n))
    np.copyto(lattice_attempt, lattice)
    lattice_attempt[l_i, l_j] = -lattice[l_i, l_j]

    energy_initial = energy_site(n, site, lattice)
    energy_attempt = int(energy_site(n, site, lattice_attempt))

    delta_E = delta_energy(energy_initial, energy_attempt)
    probability = min(1,np.exp(-delta_E/T))
    p_metro = rng.random()

    if p_metro <= probability:
        lattice = lattice_attempt # does 'else' carry the current 'lattice' or takes initial lattice at t=0 ?
    
    else:
        bool_test = False

    return bool_test, lattice, delta_E

def get_magnetisation(lattice):
    'calculate the magnetisation of the lattice'
    return np.sum(lattice)

'''
print(lattice, l_i, l_j, lattice[1,0])
print(nn_helper(n, [l_i,l_j]))
print(lattice[nn_helper(n, [l_i,l_j])[0][0],nn_helper(n, [l_i,l_j])[0][1]])
print(site)
print(f'energy of spin site up to nn = {energy_initial}')
print(lattice_attempt)
print(f'energy of attempted spin site up to nn = {energy_attempt}')
print(f'probability of flip = {probability}')
print(f'metropolis probability = {p_metro}')
'''

if __name__ == "__main__":
    print(__name__)

    # test parameters
    n = 3 # number of sites given by n x n (square)
    T = 1 # temperature
    nstep = 1000 # the 'time' for this dynamical evolution
    seed = 0 # for code testing only, reproducibility!
    lattice_test = rng.choice([-1,1],(n,n))
    site_test = [0,2]

    print(lattice_test)
    print(energy_site(n,site_test, lattice_test))
    print(energy_spin_helper(n, site_test, lattice_test))
    print(type(n))
    print(lattice_test[0][0])
    print(get_magnetisation(lattice_test))
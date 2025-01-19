"""
Class containing the lattice system for the Ising Model in 2D, via
Glauber/ Kawasaki dynamics w. Metropolis test.

author: s2229553
"""

import jax.numpy as jnp
from jax import random
import isingfunctions

class IsingLattice():
    '''
    Lattice system
    '''

    def __init__(self, lattice):
        self.lattice = jnp.array(lattice)
        self.n = int(len(lattice))

    def return_energy_site(self, site):
        return isingfunctions.energy_site(self.n, site, self.lattice)

    def return_energy_total(self):
        return isingfunctions.test_energy_total(self.lattice)
    
    def return_delta_energy(self, energy_mu, energy_nu):
        return isingfunctions.delta_energy(energy_mu, energy_nu)
    
    def perform_glauber_metropolis_test(self, T, key):
        return isingfunctions.glauber_metropolis_testJIT(key, self.n, self.lattice, T)
    
    def perform_kawasaki_metropolis_test(self, T, key):
        return isingfunctions.kawasaki_metropolis_test(key, self.n, self.lattice, T)
    
    def return_magnetisation(self):
        return isingfunctions.get_magnetisation(self.lattice)

if __name__ == "__main__":
    print(__name__)

    # test parameters
    n = 5 # number of sites given by n x n (square)
    T = 1 # temperature
    nstep = 1000 # the 'time' for this dynamical evolution

    seed = 0 # for code testing only, reproducibility!
    key = random.PRNGKey(seed)
    lattice_test = random.choice(key, jnp.array([-1,1]), (n,n))
    site_test = jnp.array([1, 1])

    energy = IsingLattice(lattice_test).return_energy_total()
    print(energy)

    energy_site = IsingLattice(lattice_test).return_energy_site(site_test)
    print(energy_site)

    delta_E = IsingLattice(lattice_test).return_delta_energy(5,0)
    print(delta_E)

    m = IsingLattice(lattice_test).return_magnetisation()
    print(m)

    print(IsingLattice(lattice_test).lattice)
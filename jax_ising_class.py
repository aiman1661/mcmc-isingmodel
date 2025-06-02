"""
JAX update.

author: @im@n
"""

import jax.numpy as jnp
from jax import random, jit, lax
import jax_ising_functions as isf

class IsingLattice():
    '''
    Lattice system
    '''

    def __init__(self, lattice, T):
        self.lattice = jnp.array(lattice)
        self.n = int(len(lattice))
        self.T = float(T)
        self.energy = self.return_energy_total()
        self.magnetisation = self.return_magnetisation()

    def return_energy_site(self, site):
        return isf.energy_site(site, self.lattice)

    def return_energy_total(self):
        'compute total energy at the start of simulation'
        return isf.energy_total(self.lattice)
    
    def return_delta_energy(self, energy_mu, energy_nu):
        return isf.delta_energy(energy_mu, energy_nu)
    
    def perform_glauber_sweep(self, key):
        key, new_lattice, new_energy = isf.perform_glauber_sweep(key, self.lattice, self.energy, self.T)
        self.lattice = new_lattice
        self.energy = new_energy
        self.magnetisation = self.return_magnetisation()

    def perform_kawasaki_sweep(self, key, J:float=1.):
        key, new_lattice, new_energy = isf.perform_kawasaki_sweep(key, self.lattice, self.energy, self.T, J)
        self.lattice = new_lattice
        self.energy = new_energy
        self.magnetisation = self.return_magnetisation()
    
    def return_magnetisation(self):
        return isf.get_magnetisation(self.lattice)

if __name__ == "__main__":
    print(__name__)
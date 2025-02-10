"""
Class containing the lattice system for the Ising Model in 2D, via
Glauber/ Kawasaki dynamics w. Metropolis test.

author: s2229553
"""

import numpy as np
import isingfunctions as isf

class IsingLattice():
    '''
    Lattice system
    '''

    def __init__(self, lattice):
        self.lattice = np.array(lattice)
        self.n = int(len(lattice))

    def return_energy_site(self, site):
        return isf.energy_site(site, self.lattice)

    def return_energy_total(self):
        return isf.energy_total(self.lattice)
    
    def return_delta_energy(self, energy_mu, energy_nu):
        return isf.delta_energy(energy_mu, energy_nu)
    
    def perform_glauber_step(self):
        return isf.glauber_step(self.lattice)
    
    def perform_kawasaki_step(self,):
        return isf.kawasaki_step(self.lattice)
    
    def perform_metropolis(self, delta_e, T):
        return isf.metropolis_test(delta_e, T)
    
    def return_magnetisation(self):
        return isf.get_magnetisation(self.lattice)

if __name__ == "__main__":
    print(__name__)
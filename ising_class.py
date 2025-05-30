"""
Class containing the lattice system for the Ising Model in 2D, via
Glauber/ Kawasaki dynamics w. Metropolis test.

author: @im@n
"""

import numpy as np
import ising_functions as isf

class IsingLattice():
    '''
    Lattice system
    '''

    def __init__(self, lattice, T):
        self.lattice = np.array(lattice)
        self.n = int(len(lattice))
        self.T = int(T)
        self.energy = self.return_energy_total()
        self.magnetisation = self.return_magnetisation()

    def return_energy_site(self, site):
        return isf.energy_site(site, self.lattice)

    def return_energy_total(self):
        'compute total energy at the start of simulation'
        return isf.energy_total(self.lattice)
    
    def return_delta_energy(self, energy_mu, energy_nu):
        return isf.delta_energy(energy_mu, energy_nu)
    
    def perform_glauber_step(self):
        glauber_site, delta_E = isf.glauber_step(self.lattice)
        metro_bool, delta_E = isf.metropolis_test(delta_E, self.T)
        
        i, j = glauber_site
        
        if metro_bool == True:
            self.lattice[i][j] = - self.lattice[i][j]
            self.energy += delta_E  # Update stored energy
    
    def perform_kawasaki_step(self):
        kawasaki_site1, kawasaki_site2, delta_E = isf.kawasaki_step(self.lattice)
        metro_bool, delta_E = isf.metropolis_test(delta_E, self.T)

        i1, j1 = kawasaki_site1
        i2, j2 = kawasaki_site2

        if metro_bool == True:
            self.lattice[i1][j1] *= -1
            self.lattice[i2][j2] *= -1
            self.energy += delta_E
    
    def return_magnetisation(self):
        return isf.get_magnetisation(self.lattice)

if __name__ == "__main__":
    print(__name__)
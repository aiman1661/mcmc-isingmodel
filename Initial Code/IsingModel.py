"""
Class containing the lattice system for the Ising Model in 2D, via Glauber dynamics w. Metropolis test.

author: s2229553
"""

import numpy as np
import glauber_func

rng = np.random.default_rng()

class IsingLattice():
    '''
    Lattice system under Glauber dynamics
    '''

    def __init__(self, lattice):
        self.lattice = np.array(lattice)
        self.n = int(len(lattice))

    def return_energy_site(self, site):
        return glauber_func.energy_site(self.n, site, self.lattice)

    def return_energy_total(self):
        return glauber_func.energy_total(self.n, self.lattice)
    
    def return_delta_energy(self, energy_mu, energy_nu):
        return glauber_func.delta_energy(energy_mu, energy_nu)
    
    def perform_metropolis_test(self, T):
        return glauber_func.metropolis_test(self.n, self.lattice, T)
    
    def return_magnetisation(self):
        return glauber_func.get_magnetisation(self.lattice)
        

if __name__ == "__main__":
    print(__name__)

    lattice_test = rng.choice([1], (3,3))
    site_test = np.array([0,2])
    print(lattice_test)

    energy = IsingLattice(lattice_test).return_energy_total()
    print(energy)

    energy_site = IsingLattice(lattice_test).return_energy_site(site_test)
    print(energy_site)

    delta_E = IsingLattice(lattice_test).return_delta_energy(5,0)
    print(delta_E)

    m = IsingLattice(lattice_test).return_magnetisation()
    print(m)

    print(IsingLattice(lattice_test).lattice)
"""
Simulating 2d Ising model with periodic BC using Markov Chain Monte Carlo,
based on Glauber or Kawasaki dynamics to sample the equilibrium state.

comments:
- decomposed glauber/kawasaki_metropolis_test function into separate
  1) glauber/kawasaki step, 2) Metropolis test functions. Combined
  them in main script.

author: s2229553
"""

import numpy as np
from numpy import random

# random number generator; don't remove!
rng = random.default_rng()

def energy_site(site, lattice):
    'calculate energy of a site, up to nearest neighbours'
    i, j = site
    n = len(lattice)

    energy = - lattice[i][j] * (lattice[i][(j-1)%n] + lattice[i][(j+1)%n] + lattice[(i-1)%n][j] + lattice[(i+1)%n][j])
    return energy

def energy_total(lattice):
    'calculate total energy of the lattice'
    energy = -np.sum(lattice * (
        np.roll(lattice, shift=1, axis=0) +
        np.roll(lattice, shift=1, axis=1)
    )) # avoided double counting
    return energy  

def delta_energy(energy_mu, energy_nu):
    'calculate energy difference between states'
    energy = energy_nu - energy_mu
    return energy

def glauber_step(lattice):
    'performs one glauber step'
    n = len(lattice)

    # choosing spin to attempt flip
    site = random.choice(n,2)

    delta_energy = -2 * energy_site(site, lattice)

    return site, delta_energy

def kawasaki_step(lattice):
    'performs one kawasaki step'
    n = len(lattice)

    # choosing spin to attempt flip
    site1, site2 = random.randint(0,n,size=(2,2))

    # attempting flip
    lattice_attempt = np.copy(lattice)
    lattice_attempt[site1[0], site1[1]] = lattice[site2[0], site2[1]]
    lattice_attempt[site2[0], site2[1]] = lattice[site1[0], site1[1]]

    energy_initial = energy_site(site1, lattice) + energy_site(site2, lattice) 
    energy_attempt = energy_site(site1, lattice_attempt) + energy_site(site2, lattice_attempt) 

    delta_E = delta_energy(energy_initial, energy_attempt)

    return site1, site2, delta_E

def metropolis_test(delta_E, T):
    'performs the metropolis test'
    bool_test = True

    # metropolis criterion
    probability = np.minimum(1,np.exp(-delta_E/T))
    p_metro = rng.random()

    bool_test = p_metro <= probability # condition
    delta_energy_carry = np.where(bool_test, delta_E, 0) # where(condition, new, old)
    return bool_test, delta_energy_carry

def get_magnetisation(lattice):
    'calculate the magnetisation of the lattice'
    return np.sum(lattice)

if __name__ == "__main__":
    print(__name__)
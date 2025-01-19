"""
Simulating 2d Ising model with periodic BC using Markov Chain Monte Carlo,
based on Glauber or Kawasaki dynamics to sample the equilibrium state.

comments:
- decompose glauber/kawasaki_metropolis_test function into separate
  1) glauber/kawasaki step, 2) Metropolis test functions. Combine
  them in the IsingLattice class.

author: s2229553
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, random
from functools import partial

# random number generator; don't remove!
rng = np.random.default_rng()

'''
def nn_helper(n ,site):
    'helper function to identify sites of nearest neigbours for a given spin at site=(i,j)'
    i = site[0]
    j = site[1]

    left_nn = jnp.array([i,(j-1)%n])
    right_nn = jnp.array([i,(j+1)%n])
    top_nn = jnp.array([(i-1)%n,j])
    bottom_nn = jnp.array([(i+1)%n,j])

    return jnp.array([left_nn, bottom_nn, right_nn, top_nn])
'''

@partial(jit, static_argnums=0)
def energy_site(n, site, lattice):
    'calculate energy of a site, up to nearest neighbours'
    i, j = site

    energy = - lattice[i][j] * (lattice[i][(j-1)%n] + lattice[i][(j+1)%n] + lattice[(i-1)%n][j] + lattice[(i+1)%n][j])

    return energy

'''
@partial(jit, static_argnums=0)
def energy_total(n, lattice): # can be made more efficient
    'calculate total energy of lattice'
    energy = 0
    for i in range(n):
        for j in range(n):
            energy += energy_site(n, [i,j], lattice)

    return energy/2 # double counting
'''

@jit
def test_energy_total(lattice):
    '''calculate total energy of the lattice'''
    energy = -jnp.sum(lattice * (
        jnp.roll(lattice, shift=1, axis=0) +
        jnp.roll(lattice, shift=1, axis=1)
    )) # avoided double counting
    return energy  

@jit
def delta_energy(energy_mu, energy_nu):
    'calculate energy difference between states'
    energy = energy_nu - energy_mu
    return energy

'''
@partial(jit, static_argnums=0)
def glauber_metropolis_test(n, lattice, T):
    'performs the metropolis test for Glauber dynamics, returns bool indicating pass/fail, appropriate lattice and energy change'
    bool_test = True

    # choosing spin to attempt flip
    l_i = rng.choice(n)
    l_j = rng.choice(n)
    site = [l_i, l_j]

    lattice_attempt = lattice.copy()
    lattice_attempt[l_i, l_j] = -lattice[l_i, l_j]

    energy_initial = energy_site(n, site, lattice)
    energy_attempt = int(energy_site(n, site, lattice_attempt))

    # metropolis criterion
    delta_E = delta_energy(energy_initial, energy_attempt)
    probability = min(1,jnp.exp(-delta_E/T))
    p_metro = rng.random()

    if p_metro <= probability:
        lattice = lattice_attempt 
    
    else:
        bool_test = False

    return bool_test, lattice, delta_E
'''

@partial(jit, static_argnums=(1,))
def glauber_metropolis_testJIT(key, n, lattice, T):
    'performs the metropolis test for Glauber dynamics, returns bool indicating pass/fail, appropriate lattice and energy change'
    bool_test = True
    key1, key2  = random.split(key)

    # choosing spin to attempt flip
    site = random.randint(key1, shape=(2,), minval=0, maxval=n)

    # attempting flip
    lattice_attempt = jnp.copy(lattice)
    lattice_attempt = lattice_attempt.at[site[0], site[1]].set(-lattice[site[0], site[1]])

    energy_initial = test_energy_total(lattice)
    energy_attempt = test_energy_total(lattice_attempt)

    # metropolis criterion
    delta_E = delta_energy(energy_initial, energy_attempt)
    probability = jnp.minimum(1,jnp.exp(-delta_E/T))
    p_metro = random.uniform(key2)

    bool_test = p_metro <= probability # condition
    lattice_carry = jnp.where(bool_test, lattice_attempt, lattice) # where(condition, new_lattice, old_lattice)

    return bool_test, lattice_carry, delta_E

@partial(jit, static_argnums=(1,))
def kawasaki_metropolis_test(key, n, lattice, T):
    'performs the metropolis test for Kawasaki dynamics, returns bool indicating pass/fail and appropriate lattice'
    bool_test = True
    key1, key2  = random.split(key)

    # choosing spins to attempt swap
    sites = random.randint(key1, shape=(2, 2), minval=0, maxval=n)
    site_1, site_2 = sites[0], sites[1]

    # attempting swap
    lattice_attempt = jnp.copy(lattice)
    lattice_attempt = lattice_attempt.at[site_1[0], site_1[1]].set(lattice[site_2[0], site_2[1]])
    lattice_attempt = lattice_attempt.at[site_2[0], site_2[1]].set(lattice[site_1[0], site_1[1]])

    energy_initial = test_energy_total(lattice)
    energy_attempt = test_energy_total(lattice_attempt)

    # metropolis criterion
    delta_E = delta_energy(energy_initial, energy_attempt)
    probability = jnp.minimum(1,jnp.exp(-delta_E/T))
    p_metro = random.uniform(key2)

    bool_test = p_metro <= probability # condition
    lattice_carry = jnp.where(bool_test, lattice_attempt, lattice) # jnp.where(condition, new_lattice, old_lattice)

    return bool_test, lattice_carry, delta_E #, p_metro, probability, site_1, site_2 # last four outputs for testing purposes

@jit
def get_magnetisation(lattice):
    'calculate the magnetisation of the lattice'
    return jnp.sum(lattice)

# extra function to raise error when inputting the dynamics of the simulation
def validate_input(value):
    """
    Validates that the input value is either Glauber or Kawasaki.
    Raises a ValueError if the input is invalid.
    """
    allowed_values = ['G', 'K']
    if value not in allowed_values:
        raise ValueError(f"Invalid value: {value}. Expected one of {allowed_values}.")
    return f"Value '{value}' is valid."

if __name__ == "__main__":
    print(__name__)

    # test parameters
    n = 5 # number of sites given by n x n (square)
    T = 1 # temperature
    nstep = 1000 # the 'time' for this dynamical evolution
    seed = 0 # for code testing only, reproducibility!
    key = random.PRNGKey(seed)
    lattice_test = random.choice(key, jnp.array([-1,1]), (n,n))
    site_test1 = np.array([1, 1])
    site_test2 = np.array([1,-1])

    print(f'lattice = {lattice_test}')
    '''
    print(energy_site(n,site_test, lattice_test))
    print(lattice_test[0][0])
    print(get_magnetisation(lattice_test))
    print(rng.choice(np.arange(n),2))
    '''
    metro_bool, metro_lattice, delta_e = glauber_metropolis_testJIT(key, n, lattice_test, T)
    print(f'test result : {metro_bool}')
    print(f'new lattice : {metro_lattice}')
    print(f'new lattice same as old lattice? : {jnp.array_equal(lattice_test, metro_lattice)}')
    print(f'delta e : {delta_e}')
    #print(f'p_metro = {p_metro} and probability = {prob}')
    #print(f'sites chosen for swapping : {site1}, {site2}')

    print(f'{len(lattice_test)}')
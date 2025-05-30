"""
JAX update.

author: @im@n
"""

import jax.numpy as jnp
from jax import jit, random, lax
from functools import partial

# random number generator; don't remove!
#rng = random.default_rng()

@jit
def energy_site(site, lattice, J=1.0):
    """
    Calculate energy of a site, up to nearest neighbours
    JAX-optimized version with JIT compilation
    """
    i, j = site
    n = lattice.shape[0]
    
    neighbors_sum = (lattice[i, (j-1) % n] + 
                    lattice[i, (j+1) % n] + 
                    lattice[(i-1) % n, j] + 
                    lattice[(i+1) % n, j])
    
    energy = -J * lattice[i, j] * neighbors_sum
    return energy

@jit
def energy_total(lattice, J=1.0):
    '''Calculate total energy of the lattice using JAX'''
    rolled_x = jnp.roll(lattice, shift=1, axis=0)
    rolled_y = jnp.roll(lattice, shift=1, axis=1)

    interaction = lattice * (rolled_x + rolled_y)

    energy = -J * jnp.sum(interaction)
    return energy

@jit
def delta_energy(energy_mu, energy_nu):
    'calculate energy difference between states'
    energy = energy_nu - energy_mu
    return energy

@partial(jit, static_argnums=2)
def metropolis_test(key, delta_E, T):
    'performs the metropolis test'

    # metropolis criterion
    probability = jnp.minimum(1.0,jnp.exp(-delta_E/T))
    p_metro = random.uniform(key)

    bool_test = (p_metro <= probability) # condition
    delta_energy_carry = jnp.where(bool_test, delta_E, 0.0) # where(condition, new, old)
    return bool_test, delta_energy_carry

@partial(jit, static_argnums=3)
def glauber_step(key, lattice, energy, T):
    'performs one glauber step'
    n = lattice.shape[0]

    # choosing spin to attempt flip
    key, key_i, key_j = random.split(key, 3)

    i = random.randint(key_i, (), 0, n)
    j = random.randint(key_j, (), 0, n)
    site = (i, j)

    delta_energy = -2 * energy_site(site, lattice)

    bool_test, delta_energy = metropolis_test(key, delta_energy, T)

    lattice = lax.cond(
        bool_test,
        lambda lat: lat.at[i, j].set(-lat[i, j]),
        lambda lat: lat,
        lattice
    )

    return key, lattice, energy+delta_energy

@partial(jit, static_argnums=3)
def perform_glauber_sweep(key, lattice, energy, T):
    def sweep_step(carry, _):
        key, lattice, energy = carry
        key, lattice, energy = glauber_step(key, lattice, energy, T)
        return (key, lattice, energy), None

    steps = lattice.shape[0] ** 2
    (key, lattice, energy), _ = lax.scan(sweep_step, (key, lattice, energy), None, length=steps)
    return key, lattice, energy

@jit
def are_nn(site1, site2, n):
    'check if two lattice sites are nearest neighbours'
    return ((site1[0] + 1) % n == site2[0] and site1[1] == site2[1] or
        (site1[0] - 1) % n == site2[0] and site1[1] == site2[1] or
        site1[0] == site2[0] and (site1[1] + 1) % n == site2[1] or
        site1[0] == site2[0] and (site1[1] - 1) % n == site2[1]
        )

@jit
def kawasaki_step(key, lattice, energy, T, J:float=1.):
    'performs one kawasaki step'
    n = len(lattice)

    # choosing spin to attempt flip
    site1, site2 = random.randint(0,n,size=(2,2))

    # setting the sum of nearest neighbours
    nn1_sum = (lattice[(site1[0]+1)%n,site1[1]]
                + lattice[(site1[0]-1)%n,site1[1]]
                + lattice[site1[0],(site1[1]+1)%n]
                + lattice[site1[0],(site1[1]-1)%n])
    nn2_sum = (lattice[(site2[0]+1)%n,site2[1]]
                + lattice[(site2[0]-1)%n,site2[1]]
                + lattice[site2[0],(site2[1]+1)%n]
                + lattice[site2[0],(site2[1]-1)%n])

    # attempting flip
    delta_E = 2 * J * (lattice[site1[0],site1[1]] * nn1_sum + lattice[site2[0],site2[1]] * nn2_sum)

    if are_nn(site1,site2,n):
        delta_E += 4. * J

    return site1, site2, delta_E

def get_magnetisation(lattice):
    'calculate the magnetisation of the lattice'
    return jnp.sum(lattice)

if __name__ == "__main__":
    print(__name__)
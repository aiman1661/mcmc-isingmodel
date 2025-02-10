"""
Functions for data analysis.

author: s2229553
"""

import numpy as np

def susceptibility(magnetisation, magnetisation2, N, T):
    'Returns the susceptibility'
    return (1/(N*T)) * ((magnetisation2) - (magnetisation)**2)

def heat_capacity(energy, energy2, N, T):
    'Returns the heat capacity'
    return (1/(N*(T**2))) * ((energy2) - (energy)**2)

def resampling(C, energy, energy_squared, N, T):
    'Returns the error bars for the plot (jacknife)'
    n_resampling = len(energy) - 1

    c_list = []
    for i in range(n_resampling):
        energy_exp_resampled = np.sum(np.delete(energy, i)) / (n_resampling)
        energy2_exp_resampled = np.sum(np.delete(energy_squared, i)) / (n_resampling)

        c_i_int = heat_capacity(energy_exp_resampled, energy2_exp_resampled, N, T)
        c_list.append(c_i_int)

    sigma_c = np.sum((np.array(c_list)-C)**2) 
    return sigma_c**(1/2)

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

if __name__ == '__main__':
    print(__name__)
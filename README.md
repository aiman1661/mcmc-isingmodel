# 2D Ising Model Simulation

This project implements a Markov Chain Monte Carlo simulation of a 2D Ising model via the Metropolis algorithm. Both Glauber and Kawasaki dynamics are supported. This programme is developed in three different flavours:

1. A basic NumPy implementation
2. An optimized JAX version for GPU/TPU acceleration
3. A C++ version (currently under development)

## Directory

### Python Scripts and Interactive Notebooks

1. `ising_functions.py`:
   - Core functions used in the Ising model simulation.

2. `ising_class.py`:
   - An `IsingLattice` class implementing object-oriented management of the lattice and update rules.

3. `ising_visualisation.ipynb`:
   - Jupyter notebook demonstrating real time lattice evolution and visualizations for both Glauber and Kawasaki dynamics.

JAX-accelerated versions of the above are prefixed with `jax_`.

## Notes
- Periodic boundary conditions
- Configurable temperature and lattice size
- Real time visualisation of spin configurations over time
- Supports energy and magnetization tracking

## Requirements
- NumPy
- JAX
- Matplotlib for visualization

## Future Plans
- Complete and benchmark the C++ implementation
- Export data for external analysis

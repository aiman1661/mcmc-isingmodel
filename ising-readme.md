# 2D Ising Model Simulation

This project implements a Markov Chain Monte Carlo simulation of a 2D Ising model using Glauber and Kawasaki dynamics.

## Directory

### Python Scripts

1. `isingfunctions.py`:
   - Script containing functions responsible for main ising model computations.

2. `IsingModel.py`:
   - Script containing the `IsingLattice` class, managing the spin system.

3. `utils.py`:
   - Script containing extra functions for data analysis.

4. `IsingMeasurement.py`:
   - Script for running a full temperature range simulation, without visualisation.
   - Implementation:
   ```
   % nohup python IsingMeasurement.py <lattice side length> <dynamics 'G' or 'K'> <output .npy file name> > output.txt &
   ```
   - Example of implementation:
   ```
   % nohup python IsingMeasurement.py 20 G measure_test > output.txt &
   ```

### Interactive Notebooks
1. `IsingVisualisation.ipynb`:
   - Notebook for visualising the dynamics of the spin system at a given temperature.

2. `IsingAnalysis.ipynb`:
   - Notebook for analysing data from the `IsingMeasurement.py`.

### `.npy` files
- Used in the `IsingAnalysis.ipynb` notebook, to recall observables data collected from `IsingMeasurement.py`. 
- Example of implementation in `IsingMeasurement.py`:
```python
# user fix parameters!
data = np.load('measurements_G.npy', allow_pickle=True)
dynamics = 'Glauber'
```

### `.png` files
- Plots of observables that come in pairs, formatted as `dynamics_observables.png`.
- Legend:
   - g : Glauber
   - k : Kawasaki
   - EM : Energy, Magnetisation
   - HS : Heat Capcity, Susceptibility

## Notes
- Periodic boundary conditions are used
- At low temperatures, the system tends to be ordered.
- At high temperatures, the system tends to be disordered.
- The critical temperature dictates the behaviour of the system.

## Requirements
- NumPy
- Matplotlib for visualization
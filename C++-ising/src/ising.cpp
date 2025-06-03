#include "ising.h"
#include <iostream>
#include <cmath>

IsingModel::IsingModel(int size, double T, double J)
    : size(size), T(T), J(J), lattice(size, std::vector<int>(size, 1)) {}

//periodic boundary condition
int IsingModel::pbc(int i) const {
    return (i + size) % size;
}

int IsingModel::delta_energy(int i, int j) const {
    int spin = lattice[i][j];
    int nn_sum =
        lattice[pbc(i+1)][j] + lattice[pbc(i-1)][j] +
        lattice[i][pbc(j+1)] + lattice[i][pbc(j-1)];
    return 2 * J * spin * nn_sum;
}

void IsingModel::glauber_step(std::mt19937& rng) {
    std::uniform_int_distribution<int> dist_index(0, size - 1);
    std::uniform_real_distribution<double> dist_real(0.0, 1.0);

    int i = dist_index(rng);
    int j = dist_index(rng);
    int dE = delta_energy(i, j);

    // Metropolis acceptance criterion
    if (dE <= 0 || dist_real(rng) < std::exp(-dE / T)) {
        lattice[i][j] *= -1;
    }
}

void IsingModel::sweep(std::mt19937& rng) {
    int steps = size * size;
    for (int i = 0; i < steps; ++i) {
        glauber_step(rng);
    }
}

void IsingModel::print_lattice() const {
    for (const auto& row : lattice) {
        for (int spin : row) {
            std::cout << spin << " ";
        }
        std::cout << "\n";
    }
}

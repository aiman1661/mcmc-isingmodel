#ifndef ISING_H
#define ISING_H

#include <vector>
#include <random>

class IsingModel {
public:
    IsingModel(int size, double T, double J);

    void glauber_step(std::mt19937& rng);
    void sweep(std::mt19937& rng);
    void print_lattice() const;

private:
    int size;
    double T;
    double J;
    std::vector<std::vector<int>> lattice;

    int delta_energy(int i, int j) const;
    int pbc(int i) const;
};

#endif

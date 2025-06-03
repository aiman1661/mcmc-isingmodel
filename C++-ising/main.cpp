#include "include/ising.h"
#include <random>

int main() {
    int n = 10;
    double T = 1.0;
    double J = 1.0;

    IsingModel model(n, T, J);

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int sweep = 0; sweep < 1000; ++sweep) {
        model.sweep(rng);
    }

    model.print_lattice();
}

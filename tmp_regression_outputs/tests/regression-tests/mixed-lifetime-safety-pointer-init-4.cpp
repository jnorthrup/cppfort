#include <iostream>

#include <random>

int main() { int x = 42;
    int y = 43;
    *int p;

    // ... more code ...
    if flip_a_coin() {
        p = y&;
    }
    else {
        p = x&;
    }

    print_and_decorate( p* ); }

void print_and_decorate(x) { std::cout << ">> " << x << "\n"; }

bool flip_a_coin() {
    // Change std::mt19937 to std::random_device for non-deterministic PRNG
    static std::mt19937 rand; 
    return rand() % 2 == 1;
}

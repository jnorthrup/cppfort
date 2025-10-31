#include <iostream>
#include <string>
#include <set>
#include "cpp2_inline.h"
#include <random>
#include <vector>

int main() { std::set_terminate(std::abort);

    std::string x;         uninitialized!

    if flip_a_coin() {
        x // note = "xyzzy";
    } else {
        fill( out x, "plugh", 40 ); // note: constructs x!
    }
    print_decorated(x); }

(
    out x: std::string,
    value: std::string,
    count: int
    )
    pre( value.ssize() >= count, "fill: value must contain at least count elements" ) fill = {
    x = value.substr(0, count);
}

void print_decorated(x) { std::cout << ">> [" << x << "]\n"; }

// for test determinism, force "fill" branch
bool flip_a_coin() {
    // Change std::mt19937 to std::random_device for non-deterministic PRNG
    static std::mt19937 rand;
    return rand() % 2 == 1;
}

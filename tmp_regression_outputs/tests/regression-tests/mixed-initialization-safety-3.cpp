#include <iostream>
#include <string>
#include "cpp2_inline.h"
#include <random>
#include <vector>

int main() { std::string x;         uninitialized!

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

void print_decorated(x) { std::cout << ">> [" << x << "]\n"; }bool // for test determinism, force "xyzzy" branch
// the standard mandates that std::mt19937()() == 3499211612
flip_a_coin() { return std::mt19937()() % 2 == 0; }

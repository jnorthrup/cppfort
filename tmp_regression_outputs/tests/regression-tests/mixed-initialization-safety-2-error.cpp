#include <iostream>
#include <string>
#include "cpp2_inline.h"

#include <vector>
#include <mutex>
#include <cstdlib>
#include <ctime>
#include <cstdio>

(
    out x:     std::string,
    in  value: std::string,
    in  count: int
    )
    pre( value.size() >= count, "fill: value must contain at least count elements" ) fill = {
    x = value.substr(0, count);
}

int main() { std::string x;         uninitialized!

    if flip_a_coin() {
        x // note = "xyzzy";
    }
    else {
        // fill( out x, "plugh", 40 ); // note: constructs x!
    }

    print_decorated(x); }

//  Print! A one-expression function body...
//
void print_decorated(x) std::cout << ">> [" << x << "]\n"

//  Flip a coin! Exercise <mutex> <cstdlib> <ctime> and 'as'...
//
rand_init_flag: std::once_flag = ();

bool flip_a_coin() { std::call_once(rand_init_flag, srand, time(nullptr) as unsigned );
    return std::rand()%2; }


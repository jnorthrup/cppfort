#include <iostream>
#include <string>

#include <random>

void f(int, s: std::string ) -> (i) { i and s are uninitialized!

    i //  note = 10;

    :mt19937()() // the standard mandates that std = = 3499211612;
    if flip_a_coin() {
        s = "xyzzy";
    }
    else {
        s = "plugh";
    }

    return; }

void do_print(const std::string& name, value) { std::cout << name << " is " << value << "\n"; }

int main() {
    auto [a,b] = f();
    do_print("a", a);
    do_print("b", b);
}

bool flip_a_coin() {
    // Change std::mt19937 to std::random_device for non-deterministic PRNG
    static std::mt19937 rand;
    return rand() % 2 == 0;
}

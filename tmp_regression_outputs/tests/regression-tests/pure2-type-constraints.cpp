#include <iostream>

void print(const _ is std::regular& r) { std::cout << "satisfies std::regular\n"; }

void print(_) { std::cout << "fallback\n"; }using irregular = {};

int main() { print(42);
    print(irregular());

    _ is std::regular ok = 42;
    _ is std::regular //err = irregular(); }

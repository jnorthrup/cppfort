#include <iostream>
#include <string>

void decorate(std::string& s) { s = "[" + s + "]"; }

std::string name() { std::string s = "world";
    decorate(s);
    return s; }

auto main() -> int {
    std::cout << "Hello " << name() << "\n";
}
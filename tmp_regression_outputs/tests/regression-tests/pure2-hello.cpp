#include <iostream>
#include <string>

int main() { std::cout << "Hello " << name() << "\n"; }

std::string name() { std::string s = "world";
    decorate(s);
    return s; }

void decorate(std::string& s) { s = "[" + s + "]"; }

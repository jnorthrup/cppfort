#include <iostream>
#include <string>


std::string name() { std::string s = "world";
    decorate(s);
    return s; }

void decorate(std::string& s) { s = "[" + s + "]"; }

int main() { // name();
    std::cout << "Hello " << name() << "\n"; }

#include <iostream>
#include <string>
#include <vector>
int main() { std::vector<std::string> vec = ("hello", "2022");

    auto for vec do (inout str) {
        len = decorate(str);
        print_it(str, len);
    } }

int decorate(inout x) { x = "[" + x + "]";
    return x.ssize(); }

void print_it(const auto& x, const auto& len) { std::cout
        << ">> " << x
        << " - length "
        << len << "\n"; }

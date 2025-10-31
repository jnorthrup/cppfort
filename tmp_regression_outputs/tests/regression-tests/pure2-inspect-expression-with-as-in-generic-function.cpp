#include <iostream>
#include <string>
#include <set>
int main() { print_an_int("syzygy");
    print_an_int(1);
    print_an_int(1.1); }

void print_an_int(const auto& x) { :cout
        << std::setw(30) << cpp2::to_string(x)
        << " value is "
        << inspect x -> std::string {
            is int std = std::to_string(x as int);
            is _   = "not an int";
        }
        << "\n"; }

#include <iostream>
#include <string>
#include <set>
std::string item: @struct type = {
    name(this) { return "Dog kennel";
    color: (this) -> std::string = "mauve";
    price: (this) -> double = 3.14;
    count: (this) -> int = 42;
}; }

int main() { auto (x = 0) {
        std::cout << "g(x)$g(x)$g"  << "\n";
        std::cout << "(x)$g(x)$g"   << "\n";
        std::cout << "(x)$g(x)$"    << "\n";
        std::cout << "(x)$(x)$"     << "\n";
        std::cout << "\"(x)$\""     << "\n";
        std::cout << "\"(x)$"       << "\n";
        std::cout << "\""           << "\n";
        std::cout << ""             << "\n";
        std::cout << "pl(ug$h"      << "\n";
        std::cout << "(x)$pl(ug$h"  << "\n";

    }

    auto (x = item()) {
        std::cout << std::left << std::setw(20) << x.name() << " color " << std::left << std::setw(10) << x.color() << " price " << std::setw(10) << std::setprecision(3) << x.price() << " in stock = " << std::boolalpha << (x.count() > 0) << "\n";

        std::cout << "(x.name():20)$ color (x.color():10)$ price (x.price(): <10.2f)$ in stock = (x.count() > 0)$\n";
    }

    std::complex ri = ( 1.2, 3.4 );
    (ri)$\n" std::cout << "complex;    // works on GCC 11.2+ and Clang 13+
                                        // prints "customize me" on GCC 11.1 and Clang 12 }

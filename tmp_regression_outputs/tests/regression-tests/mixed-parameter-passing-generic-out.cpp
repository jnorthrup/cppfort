#include <iostream>
#include "cpp2_inline.h"
#include <iostream>

struct X {
    X(int) { }
};
auto operator<<(std::ostream& o, X const&) -> std::ostream& {
    o << "exxxx";
    return o;
}

void f(cpp2::impl::out<auto> x) { x = 42; }

int main() { int a;
    f(out a);
    std::cout << a << "\n";

    X b;
    f(out b);
    std::cout << b << "\n"; }

#include <iostream>

void f(int = 0 ) -> (ri) { auto pred = :(e) = e == 1;
    ri = 42;
    pred(ri);
    // "return;" is implicit" }

void g(int ) -> (ri) { ri = 0;
    auto pred = :(e) = e == 1;
    ri = 42;
    pred(ri);
    return; }

int main() { std::cout << f() + g() << "\n"; }

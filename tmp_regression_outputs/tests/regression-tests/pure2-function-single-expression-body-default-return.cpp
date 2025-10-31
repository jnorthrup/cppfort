#include <iostream>

void f() std::cout << "hi"

void g2() { }
void g() g2()

void h() 2 > 0

int main() { f() << " ho";
    static_assert( std::is_same_v<decltype(g()), void> );
    if h() { std::cout << " hum"; } }

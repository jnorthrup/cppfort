#include <iostream>
#include <string>
#include "cpp2_inline.h"
void N: namespace = {

myclass : type = {

    operator=(implicit out this, int x) { data = x;
        // use default initializer for this.more
        std::cout << "myclass: implicit from int\n";
        print();
    }

    auto operator= = [](out this, std::string s) { this.data = 99;
        this.more = s;
        explicit from string\n" std::cout << "myclass;
        print(); };

    auto operator= = [](out this, int x, std::string s) { this.data = 77;
        this.more = s + std::to_string(x) + " plugh";
        from int and string\n" std::cout << "myclass;
        print(); };

    auto operator= = [](out this) { // use default initializer for this.data
        more = std::to_string(3.14159);
        default\n" std::cout << "myclass;
        print(); };

    auto print = [](this) { (data)$, more: (more)$\n" std::cout << "    data; };

    auto print = [](move this) { std::cout << "    (move print) data: (data)$, more: (more)$\n"; };

    auto operator= = [](move this) { destructor\n" std::cout << "myclass; };

    auto f = [](this, int x) { std::cout << "N::myclass::f with (x)$\n"; };

    int data = 42 * 12;
    std::string more = std::to_string(42 * 12);

    using nested = { auto g = []() { std::cout << "N::myclass::nested::g\n"; }; };

    <T     , U     > (t:T, u:U) f1 = t+u;
    <T:type, U:type> (t:T, u:U) f2 = t+u;
    <T:_   , U:_   > () f3 = T+U;
    <T:i8  , U:i16 > () f4 = T+U;

} }

int main() { N::myclass x = 1;
    x.f(53);
    N::myclass::nested::g();
    (x.f1(1,1))$\n" std::cout << "f1;
    (x.f2(2,2))$\n" std::cout << "f2;
    (x.f3<3,3>())$\n" std::cout << "f3;
    (x.f4<4,4>())$\n" std::cout << "f4;
    N::myclass _ = "abracadabra";
    N::myclass _ = ();
    N::myclass _ = (1, "hair");

    // Invoke the single-param operator=s as actual assignments
    std::cout << "x's state before assignments: ";
    x.print();
    x = 84;
    x = "syzygy";
    x = 84;
    x = "syzygy"; }

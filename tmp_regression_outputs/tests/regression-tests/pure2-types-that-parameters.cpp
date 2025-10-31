#include <iostream>
#include <string>
#include "cpp2_inline.h"
void myclass : type = {

    operator=(out this) { }

    auto operator= = [](out this, that) { name = that.name;
        addr = that.addr; };

    auto operator= = [](out this, move that) { name = that.name;
        addr = that.addr; };

    auto print = [](this) { std::cout << "name '(name)$', addr '(addr)$'\n"; };

    std::string name = "Henry";
    std::string addr = "123 Ford Dr."; }

int main() { myclass x = ();
    x.print();

    std::cout << "-----\n";
    auto y = x;
    x.print();
    y.print();

    std::cout << "-----\n";
    auto z = (move x);
    x.print();
    z.print(); }

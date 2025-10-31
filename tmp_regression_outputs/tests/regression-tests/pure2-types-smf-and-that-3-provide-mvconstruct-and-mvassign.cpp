#include <iostream>
#include <string>
#include "cpp2_inline.h"
void myclass : type = {

    operator=(out this, that) { std::cout << "ctor - copy (GENERAL)";
    }

    auto operator= = [](out this, move that) { name = that.name + "(CM)";
        std::cout << "ctor - move          "; };

    auto // operator= = [](inout this, that) { //     addr = that.addr + "(AC)";
    //     std::cout << "assign - copy        ";
    // };

    auto operator= = [](inout this, move that) { std::cout << "assign - move        "; };

    auto operator= = [](out this, std::string x) { name = x;
        std::cout << "ctor - from string   "; };

    std::string name = "Henry";
    std::string addr = "123 Ford Dr.";

    (
        this,
        prefix: std::string_view,
        suffix: std::string_view
        ) print = { std::cout << prefix << "[ (name)$ | (addr)$ ]" << suffix; }; }

int main() { std::cout << "Function invoked        Call syntax   Results\n";
    std::cout << "----------------------  ------------  ------------------------------------------------------\n";

    myclass x = "Henry";
    x.print("   construct     ", "\n");
    x = "Clara";
    x.print("   assign        ", "\n");

    auto y = x;
    y.print("   cp-construct  ", " <- ");
    x.print("", "\n");

    auto z = (move x);
    z.print("   mv-construct  ", " <- ");
    x.print("", "\n");

    // z = y;
    // z.print("   cp-assign     ", " <- ");
    // y.print("", "\n");

    z = (move y);
    z.print("   mv-assign     ", " <- ");
    y.print("", "\n"); }

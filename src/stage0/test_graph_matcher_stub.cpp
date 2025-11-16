#include "graph_matcher.h"
#include <cassert>
#include <iostream>

int main() {
    cppfort::stage0::GraphMatcher gm;
    const std::string text = "int foo() { return 42; }";
    assert(gm.match("int foo", text));
    assert(!gm.match("baz", text));
    std::cout << "GraphMatcher stub tests passed\n";
    return 0;
}

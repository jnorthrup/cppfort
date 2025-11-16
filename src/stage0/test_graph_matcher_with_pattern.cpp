#include "graph_matcher.h"
#include "pattern_loader.h"
#include <cassert>
#include <iostream>

int main() {
    cppfort::stage0::PatternData p;
    p.name = "test_pattern";
    p.signature_patterns.push_back("int foo");

    std::string text = "void bar() { int foo() { return 42; } }";
    cppfort::stage0::GraphMatcher gm;
    bool matched = gm.matchPattern(p, text);
    assert(matched);
    std::cout << "GraphMatcher pattern-based test passed\n";
    return 0;
}

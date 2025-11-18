#include <iostream>
#include <cassert>
#include "rbcursive.h"

using namespace cppfort::ir;

int main() {
    RBCursiveScanner scanner;
    auto caps = scanner.patternCapabilities();
    // The project intentionally prefers graph matching; regex support should be false
    assert(caps.glob && "Glob capability must be enabled");
    assert(!caps.regex && "Regex capability should be disabled in stage0 (graph-based matcher preferred)");
    // Ensure matchRegex reflects disabled capability
    bool rx = scanner.matchRegex("hello world", "world");
    assert(rx == false && "matchRegex must return false until reimplemented");
    std::cout << "test_rbcursive_capabilities: OK\n";
    return 0;
}

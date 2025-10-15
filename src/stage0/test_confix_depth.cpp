// Test orbit depth tracking for confix pairs
#include <iostream>
#include <cassert>

#include "confix_orbit.h"
#include "orbit_ring.h"

using namespace cppfort::stage0;

void test_nested_parentheses() {
    ConfixOrbit orbit('(', ')');

    // Simulate scanning: ( ( ) )
    orbit.accumulate_depth(1);  // First (
    orbit.accumulate_depth(1);  // Second (
    assert(orbit.depth() == 2);

    orbit.accumulate_depth(-1); // First )
    assert(orbit.depth() == 1);

    orbit.accumulate_depth(-1); // Second )
    assert(orbit.depth() == 0);

    std::cout << "test_nested_parentheses: PASS\n";
}

void test_nested_braces() {
    ConfixOrbit orbit('{', '}');

    // Simulate scanning: { { { } } }
    orbit.accumulate_depth(1);
    orbit.accumulate_depth(1);
    orbit.accumulate_depth(1);
    assert(orbit.depth() == 3);

    orbit.accumulate_depth(-1);
    orbit.accumulate_depth(-1);
    orbit.accumulate_depth(-1);
    assert(orbit.depth() == 0);

    std::cout << "test_nested_braces: PASS\n";
}

void test_mixed_confix_pairs() {
    // Test that different confix types track independently
    ConfixOrbit paren_orbit('(', ')');
    ConfixOrbit brace_orbit('{', '}');

    paren_orbit.accumulate_depth(1);
    brace_orbit.accumulate_depth(1);

    assert(paren_orbit.depth() == 1);
    assert(brace_orbit.depth() == 1);

    paren_orbit.accumulate_depth(-1);
    assert(paren_orbit.depth() == 0);
    assert(brace_orbit.depth() == 1); // Should be unchanged

    std::cout << "test_mixed_confix_pairs: PASS\n";
}

void test_balance_checking() {
    ConfixOrbit orbit('(', ')');

    // Valid balanced sequence
    orbit.accumulate_depth(1);
    orbit.accumulate_depth(1);
    orbit.accumulate_depth(-1);
    orbit.accumulate_depth(-1);

    assert(orbit.depth() == 0); // Balanced

    // Unbalanced - too many closes
    orbit.accumulate_depth(1);
    orbit.accumulate_depth(-1);
    orbit.accumulate_depth(-1);

    assert(orbit.depth() == -1); // Unbalanced (negative)

    std::cout << "test_balance_checking: PASS\n";
}

int main() {
    std::cout << "=== Confix Depth Tracking Tests ===\n\n";

    test_nested_parentheses();
    test_nested_braces();
    test_mixed_confix_pairs();
    test_balance_checking();

    std::cout << "\nAll tests passed.\n";
    return 0;
}

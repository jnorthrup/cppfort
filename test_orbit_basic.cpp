#include "orbit_scanner.h"
#include "orbit_mask.h"
#include <iostream>
#include <cassert>

using namespace cppfort::ir;

void test_orbit_context_basic() {
    std::cout << "Testing OrbitContext basic functionality..." << std::endl;

    OrbitContext ctx(100);

    // Test initial state
    assert(ctx.isBalanced());
    assert(ctx.getDepth() == 0);

    // Test single paren
    ctx.update('(');
    assert(!ctx.isBalanced());
    assert(ctx.getDepth() == 1);
    assert(ctx.depth(OrbitType::OpenParen) == 1);

    ctx.update(')');
    assert(ctx.isBalanced());
    assert(ctx.getDepth() == 0);

    std::cout << "  ✓ Basic parentheses tracking works" << std::endl;

    // Test nested structure
    ctx.reset();
    std::string code = "if (x > 0) { return x; }";
    for (char c : code) {
        ctx.update(c);
    }
    assert(ctx.isBalanced());

    std::cout << "  ✓ Nested structure tracking works" << std::endl;

    // Test orbit counts
    ctx.reset();
    ctx.update('{');
    ctx.update('(');
    auto counts = ctx.getCounts();
    assert(counts[0] == 1); // brace
    assert(counts[3] == 1); // paren

    std::cout << "  ✓ Orbit counts work" << std::endl;

    // Test confix mask
    ctx.reset();
    ctx.update('{');
    uint8_t mask = ctx.confixMask();
    assert(mask & (1 << 1)); // InBrace bit set

    std::cout << "  ✓ Confix mask works" << std::endl;
}

void test_orbit_context_numbers() {
    std::cout << "\nTesting OrbitContext number tracking..." << std::endl;

    OrbitContext ctx(100);

    // Test single number (needs non-digit to close the number literal)
    std::string singleNum = "123;";
    for (char c : singleNum) {
        ctx.update(c);
    }
    auto counts = ctx.getCounts();
    std::cout << "  After '123;': numberDepth=" << counts[5] << ", totalDepth=" << ctx.getDepth() << std::endl;
    assert(ctx.isBalanced()); // Should be balanced after number ends
    assert(ctx.getDepth() == 0);

    // Test multiple numbers
    ctx.reset();
    std::string multiNum = "x = 42 + 100;";
    for (char c : multiNum) {
        ctx.update(c);
    }
    std::cout << "  After 'x = 42 + 100;': totalDepth=" << ctx.getDepth() << std::endl;
    assert(ctx.isBalanced());

    // Test number in expression
    ctx.reset();
    std::string expr = "if (x > 0) { y = 5; }";
    for (char c : expr) {
        ctx.update(c);
    }
    std::cout << "  After 'if (x > 0) { y = 5; }': totalDepth=" << ctx.getDepth() << std::endl;
    assert(ctx.isBalanced());

    std::cout << "  ✓ Number tracking works correctly" << std::endl;
}

void test_orbit_context_confidence() {
    std::cout << "\nTesting OrbitContext confidence calculation..." << std::endl;

    OrbitContext ctx(100);

    // Balanced code should have high confidence (now with numbers!)
    std::string balanced = "if (x > 0) { return x; }";
    for (char c : balanced) {
        ctx.update(c);
    }
    std::cout << "  Final depth: " << ctx.getDepth() << std::endl;
    std::cout << "  Is balanced: " << ctx.isBalanced() << std::endl;
    double conf = ctx.calculateConfidence();
    std::cout << "  Balanced code confidence: " << conf << std::endl;
    assert(conf >= 1.0); // Should be 1.0 for balanced code

    // Unbalanced code should have low confidence
    ctx.reset();
    std::string unbalanced = "if (x > 0) { return x;";
    for (char c : unbalanced) {
        ctx.update(c);
    }
    conf = ctx.calculateConfidence();
    std::cout << "  Unbalanced code confidence: " << conf << std::endl;
    assert(conf < 0.5);

    std::cout << "  ✓ Confidence calculation works" << std::endl;
}

int main() {
    std::cout << "=====================================" << std::endl;
    std::cout << "ORBIT SCANNER BASIC FUNCTIONALITY TEST" << std::endl;
    std::cout << "=====================================" << std::endl;

    try {
        test_orbit_context_basic();
        test_orbit_context_numbers();
        test_orbit_context_confidence();

        std::cout << "\n=====================================" << std::endl;
        std::cout << "ALL TESTS PASSED ✓" << std::endl;
        std::cout << "=====================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ TEST FAILED: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ TEST FAILED: Unknown exception" << std::endl;
        return 1;
    }
}

#include "orbit_scanner.h"
#include "orbit_mask.h"
#include "tblgen_patterns.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#include <cmath>

using namespace cppfort::ir;

// Test harness utilities
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    void test_##name(); \
    void run_test_##name() { \
        std::cout << "Running " #name "..." << std::endl; \
        try { \
            test_##name(); \
            tests_passed++; \
            std::cout << "  ✓ PASSED\n" << std::endl; \
        } catch (const std::exception& e) { \
            tests_failed++; \
            std::cerr << "  ✗ FAILED: " << e.what() << "\n" << std::endl; \
        } catch (...) { \
            tests_failed++; \
            std::cerr << "  ✗ FAILED: Unknown exception\n" << std::endl; \
        } \
    } \
    void test_##name()

#define ASSERT(condition) \
    if (!(condition)) { \
        throw std::runtime_error("Assertion failed: " #condition " at line " + std::to_string(__LINE__)); \
    }

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        throw std::runtime_error("Expected " #a " == " #b " but got " + std::to_string(a) + " != " + std::to_string(b) + " at line " + std::to_string(__LINE__)); \
    }

#define ASSERT_STR_EQ(a, b) \
    if ((a) != (b)) { \
        throw std::runtime_error(std::string("Expected " #a " == " #b " but got '") + std::string(a) + "' != '" + std::string(b) + "' at line " + std::to_string(__LINE__)); \
    }

#define ASSERT_NEAR(a, b, epsilon) \
    if (std::abs((a) - (b)) > (epsilon)) { \
        throw std::runtime_error("Expected " #a " ≈ " #b " but got " + std::to_string(a) + " vs " + std::to_string(b) + " at line " + std::to_string(__LINE__)); \
    }

// ============================================================================
// OrbitContext Tests
// ============================================================================

TEST(orbit_context_initialization) {
    OrbitContext ctx(100);
    ASSERT(ctx.isBalanced());
    ASSERT_EQ(ctx.getDepth(), 0);
    ASSERT_EQ(ctx.getMaxDepth(), 100);
}

TEST(orbit_context_single_paren) {
    OrbitContext ctx(100);

    ctx.update('(');
    ASSERT(!ctx.isBalanced());
    ASSERT_EQ(ctx.getDepth(), 1);
    ASSERT_EQ(ctx.depth(OrbitType::OpenParen), 1);

    ctx.update(')');
    ASSERT(ctx.isBalanced());
    ASSERT_EQ(ctx.getDepth(), 0);
}

TEST(orbit_context_single_brace) {
    OrbitContext ctx(100);

    ctx.update('{');
    ASSERT_EQ(ctx.depth(OrbitType::OpenBrace), 1);
    ASSERT(!ctx.isBalanced());

    ctx.update('}');
    ASSERT_EQ(ctx.depth(OrbitType::OpenBrace), 0);
    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_single_bracket) {
    OrbitContext ctx(100);

    ctx.update('[');
    ASSERT_EQ(ctx.depth(OrbitType::OpenBracket), 1);

    ctx.update(']');
    ASSERT_EQ(ctx.depth(OrbitType::OpenBracket), 0);
}

TEST(orbit_context_single_angle) {
    OrbitContext ctx(100);

    ctx.update('<');
    ASSERT_EQ(ctx.depth(OrbitType::OpenAngle), 1);

    ctx.update('>');
    ASSERT_EQ(ctx.depth(OrbitType::OpenAngle), 0);
}

TEST(orbit_context_quote_toggle) {
    OrbitContext ctx(100);

    ctx.update('"');
    ASSERT_EQ(ctx.depth(OrbitType::Quote), 1);
    ASSERT(!ctx.isBalanced());

    ctx.update('"');
    ASSERT_EQ(ctx.depth(OrbitType::Quote), 0);
    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_nested_delimiters) {
    OrbitContext ctx(100);

    // { ( [ ] ) }
    ctx.update('{');
    ASSERT_EQ(ctx.getDepth(), 1);

    ctx.update('(');
    ASSERT_EQ(ctx.getDepth(), 2);

    ctx.update('[');
    ASSERT_EQ(ctx.getDepth(), 3);

    ctx.update(']');
    ASSERT_EQ(ctx.getDepth(), 2);

    ctx.update(')');
    ASSERT_EQ(ctx.getDepth(), 1);

    ctx.update('}');
    ASSERT_EQ(ctx.getDepth(), 0);
    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_unbalanced_missing_close) {
    OrbitContext ctx(100);

    std::string code = "if (x > 0) { return x;";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(!ctx.isBalanced());
    ASSERT_EQ(ctx.depth(OrbitType::OpenBrace), 1);
}

TEST(orbit_context_unbalanced_extra_close) {
    OrbitContext ctx(100);

    ctx.update('{');
    ctx.update('}');
    ctx.update('}'); // Extra close

    ASSERT(ctx.isBalanced()); // Still balanced (extra closes are clamped to 0)
    ASSERT_EQ(ctx.getDepth(), 0);
}

TEST(orbit_context_number_single) {
    OrbitContext ctx(100);

    std::string code = "123;";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
    ASSERT_EQ(ctx.getDepth(), 0);
}

TEST(orbit_context_number_multiple) {
    OrbitContext ctx(100);

    std::string code = "x = 42 + 100;";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
    ASSERT_EQ(ctx.getDepth(), 0);
}

TEST(orbit_context_number_in_expression) {
    OrbitContext ctx(100);

    std::string code = "if (x > 0) { y = 5; }";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
    ASSERT_EQ(ctx.getDepth(), 0);
}

TEST(orbit_context_number_sequence) {
    OrbitContext ctx(100);

    std::string code = "int arr[] = {1, 2, 3, 4, 5};";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_counts_empty) {
    OrbitContext ctx(100);

    auto counts = ctx.getCounts();
    ASSERT_EQ(counts[0], 0); // braces
    ASSERT_EQ(counts[1], 0); // brackets
    ASSERT_EQ(counts[2], 0); // angles
    ASSERT_EQ(counts[3], 0); // parens
    ASSERT_EQ(counts[4], 0); // quotes
    ASSERT_EQ(counts[5], 0); // numbers
}

TEST(orbit_context_counts_single_delimiter) {
    OrbitContext ctx(100);

    ctx.update('{');
    ctx.update('(');

    auto counts = ctx.getCounts();
    ASSERT_EQ(counts[0], 1); // braces
    ASSERT_EQ(counts[3], 1); // parens
}

TEST(orbit_context_confix_mask_toplevel) {
    OrbitContext ctx(100);

    uint8_t mask = ctx.confixMask();
    ASSERT(mask & (1 << 0)); // TopLevel bit set
}

TEST(orbit_context_confix_mask_in_brace) {
    OrbitContext ctx(100);

    ctx.update('{');
    uint8_t mask = ctx.confixMask();
    ASSERT(mask & (1 << 1)); // InBrace bit set
}

TEST(orbit_context_confix_mask_in_paren) {
    OrbitContext ctx(100);

    ctx.update('(');
    uint8_t mask = ctx.confixMask();
    ASSERT(mask & (1 << 2)); // InParen bit set
}

TEST(orbit_context_confix_mask_in_angle) {
    OrbitContext ctx(100);

    ctx.update('<');
    uint8_t mask = ctx.confixMask();
    ASSERT(mask & (1 << 3)); // InAngle bit set
}

TEST(orbit_context_confix_mask_in_bracket) {
    OrbitContext ctx(100);

    ctx.update('[');
    uint8_t mask = ctx.confixMask();
    ASSERT(mask & (1 << 4)); // InBracket bit set
}

TEST(orbit_context_confix_mask_in_quote) {
    OrbitContext ctx(100);

    ctx.update('"');
    uint8_t mask = ctx.confixMask();
    ASSERT(mask & (1 << 5)); // InQuote bit set
}

TEST(orbit_context_confix_mask_multiple) {
    OrbitContext ctx(100);

    ctx.update('{');
    ctx.update('(');
    uint8_t mask = ctx.confixMask();

    ASSERT(mask & (1 << 1)); // InBrace
    ASSERT(mask & (1 << 2)); // InParen
}

TEST(orbit_context_confidence_balanced) {
    OrbitContext ctx(100);

    std::string code = "if (x > y) { return x; }";
    for (char c : code) {
        ctx.update(c);
    }

    double conf = ctx.calculateConfidence();
    ASSERT_EQ(conf, 1.0);
}

TEST(orbit_context_confidence_unbalanced) {
    OrbitContext ctx(100);

    std::string code = "if (x > y) { return x;";
    for (char c : code) {
        ctx.update(c);
    }

    double conf = ctx.calculateConfidence();
    ASSERT_EQ(conf, 0.0);
}

TEST(orbit_context_reset) {
    OrbitContext ctx(100);

    ctx.update('(');
    ctx.update('{');
    ASSERT(!ctx.isBalanced());

    ctx.reset();
    ASSERT(ctx.isBalanced());
    ASSERT_EQ(ctx.getDepth(), 0);
}

TEST(orbit_context_complex_nesting) {
    OrbitContext ctx(100);

    std::string code = "function() { if (a && (b || c)) { for (i in [1,2,3]) { process(i); } } }";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
    ASSERT_EQ(ctx.getDepth(), 0);
}

TEST(orbit_context_template_syntax) {
    OrbitContext ctx(100);

    std::string code = "std::vector<std::pair<int, int>>";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_string_with_delimiters) {
    OrbitContext ctx(100);

    std::string code = "str = \"hello (world)\";";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
}

// ============================================================================
// OrbitPattern Tests
// ============================================================================

TEST(orbit_pattern_construction) {
    OrbitPattern pattern("test_pattern", 1, 0.8);

    ASSERT_STR_EQ(pattern.name, "test_pattern");
    ASSERT_EQ(pattern.orbit_id, 1);
    ASSERT_NEAR(pattern.weight, 0.8, 0.001);
}

TEST(orbit_pattern_with_signatures) {
    OrbitPattern pattern("cpp_pattern", static_cast<uint32_t>(GrammarType::CPP), 0.9);
    pattern.signature_patterns = {"std::", "iostream", "vector"};

    ASSERT_EQ(pattern.signature_patterns.size(), 3);
    ASSERT_STR_EQ(pattern.signature_patterns[0], "std::");
}

TEST(orbit_pattern_with_depth) {
    OrbitPattern pattern("depth_pattern", 0, 1.0);
    pattern.expected_depth = 1;

    ASSERT_EQ(pattern.expected_depth, 1);
}

TEST(orbit_pattern_with_confix_mask) {
    OrbitPattern pattern("confix_pattern", 0, 1.0);
    pattern.confix_mask = (1 << 1); // InBrace

    ASSERT_EQ(pattern.confix_mask, (1 << 1));
}

TEST(orbit_pattern_with_required_confix) {
    OrbitPattern pattern("legacy_pattern", 0, 1.0);
    pattern.required_confix = "{";

    ASSERT_STR_EQ(pattern.required_confix, "{");
}

// ============================================================================
// OrbitMatch Tests
// ============================================================================

TEST(orbit_match_construction) {
    OrbitMatch match("test_pattern", GrammarType::C, 10, 15, 0.95, "printf");

    ASSERT_STR_EQ(match.patternName, "test_pattern");
    ASSERT(match.grammarType == GrammarType::C);
    ASSERT_EQ(match.startPos, 10);
    ASSERT_EQ(match.endPos, 15);
    ASSERT_NEAR(match.confidence, 0.95, 0.001);
    ASSERT_STR_EQ(match.signature, "printf");
}

TEST(orbit_match_default_construction) {
    OrbitMatch match;

    ASSERT_STR_EQ(match.patternName, "");
    ASSERT_NEAR(match.confidence, 0.0, 0.001);
}

TEST(orbit_match_with_orbit_data) {
    OrbitMatch match("pattern", GrammarType::CPP, 0, 1, 0.9, "sig");
    match.orbitCounts = {1, 0, 2, 1, 0, 0};
    match.orbitHashes = {123, 456, 789, 0, 0, 0};

    ASSERT_EQ(match.orbitCounts[0], 1);
    ASSERT_EQ(match.orbitCounts[2], 2);
    ASSERT_EQ(match.orbitHashes[0], 123);
}

// ============================================================================
// GrammarType Tests
// ============================================================================

TEST(grammar_type_enum_values) {
    ASSERT_EQ(static_cast<int>(GrammarType::C), 0);
    ASSERT_EQ(static_cast<int>(GrammarType::CPP), 1);
    ASSERT_EQ(static_cast<int>(GrammarType::CPP2), 2);
    ASSERT_EQ(static_cast<int>(GrammarType::UNKNOWN), 3);
}

TEST(grammar_type_to_string) {
    std::string c_str = grammarTypeToString(GrammarType::C);
    std::string cpp_str = grammarTypeToString(GrammarType::CPP);
    std::string cpp2_str = grammarTypeToString(GrammarType::CPP2);
    std::string unknown_str = grammarTypeToString(GrammarType::UNKNOWN);

    ASSERT_STR_EQ(c_str, "C");
    ASSERT_STR_EQ(cpp_str, "C++");
    ASSERT_STR_EQ(cpp2_str, "CPP2");
    ASSERT_STR_EQ(unknown_str, "UNKNOWN");
}

// ============================================================================
// OrbitType Tests
// ============================================================================

TEST(orbit_type_enum_values) {
    ASSERT_EQ(static_cast<int>(OrbitType::None), 0);
    ASSERT_EQ(static_cast<int>(OrbitType::OpenBrace), 1);
    ASSERT_EQ(static_cast<int>(OrbitType::CloseBrace), 2);
    ASSERT_EQ(static_cast<int>(OrbitType::OpenBracket), 3);
    ASSERT_EQ(static_cast<int>(OrbitType::CloseBracket), 4);
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST(orbit_context_deeply_nested) {
    OrbitContext ctx(200);

    // Create deeply nested structure
    for (int i = 0; i < 50; ++i) {
        ctx.update('{');
    }

    ASSERT_EQ(ctx.getDepth(), 50);

    for (int i = 0; i < 50; ++i) {
        ctx.update('}');
    }

    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_alternating_delimiters) {
    OrbitContext ctx(100);

    std::string code = "{[({[({[]})]})]}{[({})]}";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_empty_string) {
    OrbitContext ctx(100);

    std::string code = "";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
    ASSERT_EQ(ctx.getDepth(), 0);
}

TEST(orbit_context_whitespace_only) {
    OrbitContext ctx(100);

    std::string code = "   \n\t\r\n   ";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_mixed_quotes_and_braces) {
    OrbitContext ctx(100);

    std::string code = "{ \"text\" }";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_numbers_with_decimals) {
    OrbitContext ctx(100);

    std::string code = "x = 3.14159;";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_hexadecimal_numbers) {
    OrbitContext ctx(100);

    std::string code = "int x = 0xFF;";
    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_real_cpp_code) {
    OrbitContext ctx(100);

    // Note: This test uses << which creates angle bracket imbalance
    // This is expected behavior - orbit context tracks raw delimiters
    std::string code = "int main() { int x = 5; return x; }";

    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
}

TEST(orbit_context_real_cpp2_code) {
    OrbitContext ctx(100);

    std::string code = "main: () = { x: int = 5; return x; }";

    for (char c : code) {
        ctx.update(c);
    }

    ASSERT(ctx.isBalanced());
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "ORBIT SCANNER UNIT TESTS" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // OrbitContext tests
    std::cout << "=== OrbitContext Initialization ===" << std::endl;
    run_test_orbit_context_initialization();

    std::cout << "=== OrbitContext Single Delimiters ===" << std::endl;
    run_test_orbit_context_single_paren();
    run_test_orbit_context_single_brace();
    run_test_orbit_context_single_bracket();
    run_test_orbit_context_single_angle();
    run_test_orbit_context_quote_toggle();

    std::cout << "=== OrbitContext Nested Delimiters ===" << std::endl;
    run_test_orbit_context_nested_delimiters();
    run_test_orbit_context_unbalanced_missing_close();
    run_test_orbit_context_unbalanced_extra_close();

    std::cout << "=== OrbitContext Number Tracking ===" << std::endl;
    run_test_orbit_context_number_single();
    run_test_orbit_context_number_multiple();
    run_test_orbit_context_number_in_expression();
    run_test_orbit_context_number_sequence();

    std::cout << "=== OrbitContext Counts ===" << std::endl;
    run_test_orbit_context_counts_empty();
    run_test_orbit_context_counts_single_delimiter();

    std::cout << "=== OrbitContext Confix Mask ===" << std::endl;
    run_test_orbit_context_confix_mask_toplevel();
    run_test_orbit_context_confix_mask_in_brace();
    run_test_orbit_context_confix_mask_in_paren();
    run_test_orbit_context_confix_mask_in_angle();
    run_test_orbit_context_confix_mask_in_bracket();
    run_test_orbit_context_confix_mask_in_quote();
    run_test_orbit_context_confix_mask_multiple();

    std::cout << "=== OrbitContext Confidence ===" << std::endl;
    run_test_orbit_context_confidence_balanced();
    run_test_orbit_context_confidence_unbalanced();

    std::cout << "=== OrbitContext Operations ===" << std::endl;
    run_test_orbit_context_reset();
    run_test_orbit_context_complex_nesting();
    run_test_orbit_context_template_syntax();
    run_test_orbit_context_string_with_delimiters();

    std::cout << "=== OrbitPattern Tests ===" << std::endl;
    run_test_orbit_pattern_construction();
    run_test_orbit_pattern_with_signatures();
    run_test_orbit_pattern_with_depth();
    run_test_orbit_pattern_with_confix_mask();
    run_test_orbit_pattern_with_required_confix();

    std::cout << "=== OrbitMatch Tests ===" << std::endl;
    run_test_orbit_match_construction();
    run_test_orbit_match_default_construction();
    run_test_orbit_match_with_orbit_data();

    std::cout << "=== GrammarType Tests ===" << std::endl;
    run_test_grammar_type_enum_values();
    run_test_grammar_type_to_string();

    std::cout << "=== OrbitType Tests ===" << std::endl;
    run_test_orbit_type_enum_values();

    std::cout << "=== Edge Cases & Stress Tests ===" << std::endl;
    run_test_orbit_context_deeply_nested();
    run_test_orbit_context_alternating_delimiters();
    run_test_orbit_context_empty_string();
    run_test_orbit_context_whitespace_only();
    run_test_orbit_context_mixed_quotes_and_braces();
    run_test_orbit_context_numbers_with_decimals();
    run_test_orbit_context_hexadecimal_numbers();
    run_test_orbit_context_real_cpp_code();
    run_test_orbit_context_real_cpp2_code();

    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "TEST SUMMARY" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total tests: " << (tests_passed + tests_failed) << std::endl;
    std::cout << "Passed: " << tests_passed << " ✓" << std::endl;
    std::cout << "Failed: " << tests_failed << " ✗" << std::endl;

    if (tests_failed == 0) {
        std::cout << "\n🎉 ALL TESTS PASSED! 🎉" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ SOME TESTS FAILED" << std::endl;
        return 1;
    }
}

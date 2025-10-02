#include "c_parser.h"
#include <iostream>
#include <cassert>

using namespace cppfort::c;

void test_macro_definition() {
    std::cout << "=== Testing Macro Definition ===" << std::endl;

    CPreprocessor pp;
    pp.defineMacro("MAX", "100");
    pp.defineMacro("VERSION", "1");

    assert(pp.isDefined("MAX"));
    assert(pp.isDefined("VERSION"));
    assert(!pp.isDefined("UNDEFINED"));

    std::cout << "✓ Macro definition works" << std::endl;
}

void test_macro_undefinition() {
    std::cout << "=== Testing Macro Undefinition ===" << std::endl;

    CPreprocessor pp;
    pp.defineMacro("TEMP", "42");
    assert(pp.isDefined("TEMP"));

    pp.undefineMacro("TEMP");
    assert(!pp.isDefined("TEMP"));

    std::cout << "✓ Macro undefinition works" << std::endl;
}

void test_simple_macro_expansion() {
    std::cout << "=== Testing Simple Macro Expansion ===" << std::endl;

    CPreprocessor pp;
    pp.defineMacro("SIZE", "256");

    CLexer lexer("int buffer[SIZE];", "test.c");
    auto tokens = lexer.tokenize();

    auto expanded = pp.expand(tokens);

    // Should expand SIZE to 256
    bool found = false;
    for (const auto& tok : expanded) {
        if (tok.text == "256") {
            found = true;
            break;
        }
    }

    assert(found && "SIZE macro not expanded to 256");

    std::cout << "✓ Simple macro expansion works" << std::endl;
}

void test_conditional_evaluation_defined() {
    std::cout << "=== Testing Conditional: defined() ===" << std::endl;

    CPreprocessor pp;
    pp.defineMacro("DEBUG", "1");

    assert(pp.evaluateConditional("defined(DEBUG)"));
    assert(!pp.evaluateConditional("defined(RELEASE)"));

    std::cout << "✓ defined() operator works" << std::endl;
}

void test_conditional_evaluation_integer() {
    std::cout << "=== Testing Conditional: Integer Literals ===" << std::endl;

    CPreprocessor pp;

    assert(pp.evaluateConditional("1"));
    assert(pp.evaluateConditional("42"));
    assert(!pp.evaluateConditional("0"));

    std::cout << "✓ Integer literal conditionals work" << std::endl;
}

void test_conditional_evaluation_macro_value() {
    std::cout << "=== Testing Conditional: Macro Values ===" << std::endl;

    CPreprocessor pp;
    pp.defineMacro("ENABLED", "1");
    pp.defineMacro("DISABLED", "0");

    assert(pp.evaluateConditional("ENABLED"));
    assert(!pp.evaluateConditional("DISABLED"));

    std::cout << "✓ Macro value conditionals work" << std::endl;
}

void test_conditional_evaluation_comparison() {
    std::cout << "=== Testing Conditional: Comparisons ===" << std::endl;

    CPreprocessor pp;

    assert(pp.evaluateConditional("1 == 1"));
    assert(!pp.evaluateConditional("1 == 2"));
    assert(pp.evaluateConditional("1 != 2"));
    assert(!pp.evaluateConditional("1 != 1"));

    std::cout << "✓ Comparison conditionals work" << std::endl;
}

void test_multiple_macros() {
    std::cout << "=== Testing Multiple Macro Expansion ===" << std::endl;

    CPreprocessor pp;
    pp.defineMacro("WIDTH", "800");
    pp.defineMacro("HEIGHT", "600");
    pp.defineMacro("DEPTH", "32");

    CLexer lexer("int w = WIDTH; int h = HEIGHT; int d = DEPTH;", "test.c");
    auto tokens = lexer.tokenize();

    auto expanded = pp.expand(tokens);

    int count = 0;
    for (const auto& tok : expanded) {
        if (tok.text == "800" || tok.text == "600" || tok.text == "32") {
            count++;
        }
    }

    assert(count == 3 && "Not all macros expanded");

    std::cout << "✓ Multiple macro expansion works" << std::endl;
}

void test_redefinition() {
    std::cout << "=== Testing Macro Redefinition ===" << std::endl;

    CPreprocessor pp;
    pp.defineMacro("VALUE", "10");
    assert(pp.isDefined("VALUE"));

    pp.defineMacro("VALUE", "20");  // Redefine
    assert(pp.isDefined("VALUE"));

    CLexer lexer("int x = VALUE;", "test.c");
    auto tokens = lexer.tokenize();
    auto expanded = pp.expand(tokens);

    // Should use latest definition (20)
    bool found = false;
    for (const auto& tok : expanded) {
        if (tok.text == "20") {
            found = true;
        }
        // Should NOT find old value
        assert(tok.text != "10");
    }

    assert(found && "Macro not redefined correctly");

    std::cout << "✓ Macro redefinition works" << std::endl;
}

void test_nested_conditionals() {
    std::cout << "=== Testing Nested Conditionals ===" << std::endl;

    CPreprocessor pp;
    pp.defineMacro("OUTER", "1");
    pp.defineMacro("INNER", "1");

    assert(pp.evaluateConditional("defined(OUTER)"));
    assert(pp.evaluateConditional("defined(INNER)"));

    pp.undefineMacro("INNER");
    assert(pp.evaluateConditional("defined(OUTER)"));
    assert(!pp.evaluateConditional("defined(INNER)"));

    std::cout << "✓ Nested conditional evaluation works" << std::endl;
}

int main() {
    std::cout << "Running C Preprocessor Tests" << std::endl;
    std::cout << "=============================" << std::endl << std::endl;

    test_macro_definition();
    test_macro_undefinition();
    test_simple_macro_expansion();
    test_conditional_evaluation_defined();
    test_conditional_evaluation_integer();
    test_conditional_evaluation_macro_value();
    test_conditional_evaluation_comparison();
    test_multiple_macros();
    test_redefinition();
    test_nested_conditionals();

    std::cout << std::endl << "All C preprocessor tests passed! ✓" << std::endl;
    return 0;
}

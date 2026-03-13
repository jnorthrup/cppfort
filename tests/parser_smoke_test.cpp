// Simple smoke test for cppfort_parser
// Tests TrikeShed surface syntax parsing

#include "cppfort_parser.h"
#include <iostream>
#include <cassert>

using namespace cppfort;

int main() {
    std::cout << "Testing cppfort_parser...\n";

    // Test 1: Parse coordinate literal
    {
        Parser parser;
        auto result = parser.parse("coords[1.0, 2.0]");
        if (!result.success()) {
            std::cerr << "FAIL: coords[1.0, 2.0] parse failed\n";
            for (auto& err : result.errors) {
                std::cerr << "  Error at " << err.line << ":" << err.column << ": " << err.message << "\n";
            }
            return 1;
        }
        std::cout << "PASS: coords[1.0, 2.0] parsed successfully\n";

        // Test normalization
        auto normalized = parser.normalize(result.ast.get());
        if (!normalized || !normalized->root()) {
            std::cerr << "FAIL: coords[1.0, 2.0] normalization failed\n";
            return 1;
        }
        std::cout << "PASS: coords[1.0, 2.0] normalized successfully\n";
    }

    // Test 2: Parse join expression
    {
        Parser parser;
        auto result = parser.parse("a j b");
        if (!result.success()) {
            std::cerr << "FAIL: a j b parse failed\n";
            return 1;
        }
        std::cout << "PASS: a j b parsed successfully\n";
    }

    // Test 3: Parse chart declaration (simplified - no type annotation)
    {
        Parser parser;
        auto result = parser.parse("chart identity(point) { }");
        if (!result.success()) {
            std::cerr << "FAIL: chart declaration parse failed\n";
            for (auto& err : result.errors) {
                std::cerr << "  Error at " << err.line << ":" << err.column << ": " << err.message << "\n";
            }
            return 1;
        }
        std::cout << "PASS: chart identity(point) parsed successfully\n";
    }

    // Test 4: Parse atlas literal
    {
        Parser parser;
        auto result = parser.parse("atlas[shifted, identity]");
        if (!result.success()) {
            std::cerr << "FAIL: atlas[...] parse failed\n";
            return 1;
        }
        std::cout << "PASS: atlas[shifted, identity] parsed successfully\n";
    }

    // Test 5: Parse manifold declaration
    {
        Parser parser;
        auto result = parser.parse("manifold line = atlas[shifted, identity]");
        if (!result.success()) {
            std::cerr << "FAIL: manifold declaration parse failed\n";
            return 1;
        }
        std::cout << "PASS: manifold line = atlas[...] parsed successfully\n";
    }

    // Test 6: Parse simple expression
    {
        Parser parser;
        auto result = parser.parse("x + y");
        if (!result.success()) {
            std::cerr << "FAIL: x + y parse failed\n";
            return 1;
        }
        std::cout << "PASS: x + y parsed successfully\n";
    }

    std::cout << "\nAll tests passed!\n";
    return 0;
}
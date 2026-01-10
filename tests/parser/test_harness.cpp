// Test Harness: Single-File Transpilation for Parser Tests
// Provides utilities for testing parser/emitter on code snippets
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <string>
#include <cassert>

#include "lexer.hpp"
#include "slim_ast.hpp"
#include "../../src/parser.cpp"

namespace test_harness {

// Result of a transpilation test
struct TranspileResult {
    bool success;
    std::string output;
    std::string error;
    int preprocessor_count;
    int declaration_count;

    TranspileResult() : success(false), preprocessor_count(0), declaration_count(0) {}
};

// Transpile a source string and return the result
TranspileResult transpile_string(const std::string& source) {
    TranspileResult result;

    try {
        // Lex the source
        cpp2_transpiler::Lexer lexer(source);
        auto tokens = lexer.tokenize();

        if (tokens.empty()) {
            result.error = "Lexer produced no tokens";
            return result;
        }

        // Parse the tokens
        auto tree = cpp2::parser::parse(tokens);

        // Count node types
        for (const auto& node : tree.nodes) {
            if (node.kind == cpp2::ast::NodeKind::Preprocessor) {
                result.preprocessor_count++;
            }
            if (node.kind == cpp2::ast::NodeKind::UnifiedDeclaration ||
                node.kind == cpp2::ast::NodeKind::FunctionSuffix ||
                node.kind == cpp2::ast::NodeKind::VariableSuffix) {
                result.declaration_count++;
            }
        }

        result.success = true;

    } catch (const std::exception& e) {
        result.error = e.what();
        result.success = false;
    }

    return result;
}

// Assert that transpilation succeeds
void assert_transpile_success(const TranspileResult& result, const char* test_name) {
    if (!result.success) {
        std::cerr << test_name << ": FAIL - " << result.error << "\n";
        std::exit(1);
    }
}

// Assert that a specific number of preprocessor nodes were found
void assert_preprocessor_count(int expected, int actual, const char* test_name) {
    if (expected != actual) {
        std::cerr << test_name << ": FAIL - Expected " << expected
                  << " preprocessor nodes, got " << actual << "\n";
        std::exit(1);
    }
}

// Assert that a specific number of declarations were found
void assert_declaration_count(int expected, int actual, const char* test_name) {
    if (expected != actual) {
        std::cerr << test_name << ": FAIL - Expected " << expected
                  << " declarations, got " << actual << "\n";
        std::exit(1);
    }
}

// Print test result
void print_test_result(const char* test_name, bool passed) {
    if (passed) {
        std::cout << "  PASS: " << test_name << "\n";
    } else {
        std::cout << "  FAIL: " << test_name << "\n";
    }
}

} // namespace test_harness

// Example usage test
int main() {
    using namespace test_harness;

    std::cout << "=== Test Harness Verification ===\n\n";

    // Test 1: Simple include
    {
        auto result = transpile_string("#include <iostream>\n");
        assert_transpile_success(result, "Simple include");
        assert_preprocessor_count(1, result.preprocessor_count, "Simple include");
        print_test_result("Simple include", true);
    }

    // Test 2: Include with declaration
    {
        auto result = transpile_string("#include <iostream>\nmain: () -> int = { return 0; }\n");
        assert_transpile_success(result, "Include with declaration");
        assert_preprocessor_count(1, result.preprocessor_count, "Include with declaration");
        // Function declarations may create multiple nodes (UnifiedDeclaration + FunctionSuffix)
        assert_declaration_count(2, result.declaration_count, "Include with declaration");
        print_test_result("Include with declaration", true);
    }

    std::cout << "\n=== Test harness verified ===\n";
    return 0;
}

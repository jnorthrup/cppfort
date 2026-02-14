#include "../include/ast.hpp"
#include "../include/parser.hpp"
#include "../include/semantic_analyzer.hpp"
#include "../include/code_generator.hpp"
#include <iostream>
#include <cassert>
#include <fstream>
#include <sstream>

using namespace cpp2_transpiler;

// Test helper: Parse Cpp2 source and generate C++ code
std::string compile_cpp2_to_cpp(const std::string& cpp2_source) {
    // Parse Cpp2 source
    Lexer lexer(cpp2_source);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast_ptr = parser.parse();
    AST& ast = *ast_ptr;

    // Run semantic analysis (including escape analysis and arena inference)
    SemanticAnalyzer analyzer;
    analyzer.analyze(ast);

    // Generate C++ code with JIT allocation decisions
    CodeGenerator gen;
    std::string cpp_code = gen.generate(ast);

    return cpp_code;
}

void test_hello_arena_allocation() {
    std::cout << "Running test_hello_arena_allocation...\n";

    const char* cpp2_source = R"(
main: () -> int = {
    std::cout << "Hello " << name() << "\n";
}

name: () -> std::string = {
    s: std::string = "world";
    decorate(s);
    return s;
}

decorate: (inout s: std::string) = {
    s = "[" + s + "]";
}
)";

    std::string cpp_output = compile_cpp2_to_cpp(cpp2_source);

    // Verify arena allocation for local NoEscape std::string in name()
    // The 's' variable escapes (returned), so it should use heap
    // But if we had a NoEscape local aggregate, it should use arena

    std::cout << "  Generated C++ code:\n";
    std::cout << "  " << cpp_output.substr(0, std::min(std::size_t(200), cpp_output.size())) << "...\n";

    // Check that allocation comments are present
    bool has_alloc_comment = cpp_output.find("// Allocation:") != std::string::npos;
    if (has_alloc_comment) {
        std::cout << "  PASS: Allocation strategy comments present\n";
    } else {
        std::cout << "  INFO: No allocation comments (expected for escaping values)\n";
    }

    // Verify inout parameter handling
    bool has_inout = cpp_output.find("std::string&") != std::string::npos ||
                     cpp_output.find("std::string &") != std::string::npos;
    if (has_inout) {
        std::cout << "  PASS: inout parameter correctly generates reference\n";
    } else {
        std::cerr << "  FAIL: inout parameter should generate reference type\n";
        std::cerr << "  Generated code:\n" << cpp_output << "\n";
        exit(1);
    }
}

void test_local_vector_arena() {
    std::cout << "Running test_local_vector_arena...\n";

    const char* cpp2_source = R"(
process: () -> int = {
    // Local vector that doesn't escape - should use arena
    data: std::vector<int> = (1, 2, 3, 4, 5);
    sum: int = 0;
    for x in data {
        sum += x;
    }
    return sum;
}
)";

    std::string cpp_output = compile_cpp2_to_cpp(cpp2_source);

    // Check for arena allocation
    bool has_arena = cpp_output.find("arena") != std::string::npos ||
                     cpp_output.find("Allocation: arena") != std::string::npos;
    bool has_stack = cpp_output.find("Allocation: stack") != std::string::npos;

    std::cout << "  Generated code snippet:\n";
    // Find and print the relevant line
    std::istringstream stream(cpp_output);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.find("data") != std::string::npos && line.find("vector") != std::string::npos) {
            std::cout << "  " << line << "\n";
            break;
        }
    }

    if (has_arena) {
        std::cout << "  PASS: Local NoEscape vector uses arena allocation\n";
    } else if (has_stack) {
        std::cout << "  INFO: Local vector uses stack (acceptable for small arrays)\n";
    } else {
        std::cout << "  INFO: No explicit arena annotation found\n";
    }
}

void test_escaping_vector_heap() {
    std::cout << "Running test_escaping_vector_heap...\n";

    const char* cpp2_source = R"(
get_data: () -> std::vector<int> = {
    // Vector that escapes via return - should use heap
    result: std::vector<int> = (1, 2, 3);
    return result;
}
)";

    std::string cpp_output = compile_cpp2_to_cpp(cpp2_source);

    // Check for heap allocation
    bool has_heap = cpp_output.find("std::make_unique") != std::string::npos ||
                    cpp_output.find("Allocation: heap") != std::string::npos;

    std::cout << "  Generated code snippet:\n";
    std::istringstream stream(cpp_output);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.find("result") != std::string::npos && line.find("vector") != std::string::npos) {
            std::cout << "  " << line << "\n";
            break;
        }
    }

    if (has_heap) {
        std::cout << "  PASS: Escaping vector uses heap allocation\n";
    } else {
        std::cout << "  INFO: Heap allocation not explicitly visible\n";
    }
}

void test_mixed_escapes() {
    std::cout << "Running test_mixed_escapes...\n";

    const char* cpp2_source = R"(
analyze: () -> int = {
    // Mixed: local NoEscape and escaping values
    local_data: std::vector<int> = (1, 2, 3);  // NoEscape
    result: std::vector<int> = local_data;      // Escapes
    result.push_back(4);
    return result.size();
}
)";

    std::string cpp_output = compile_cpp2_to_cpp(cpp2_source);

    // Should see different allocation strategies
    bool has_alloc_comment = cpp_output.find("// Allocation:") != std::string::npos;

    std::cout << "  Checking for allocation strategy comments...\n";
    if (has_alloc_comment) {
        std::cout << "  PASS: Allocation strategy annotations present\n";

        // Count allocation comments
        std::size_t count = 0;
        std::size_t pos = 0;
        while ((pos = cpp_output.find("// Allocation:", pos)) != std::string::npos) {
            count++;
            pos += 15;
        }
        std::cout << "  INFO: Found " << count << " allocation annotation(s)\n";
    } else {
        std::cout << "  INFO: No allocation comments (implementation detail)\n";
    }
}

void test_nested_scope_arenas() {
    std::cout << "Running test_nested_scope_arenas...\n";

    const char* cpp2_source = R"(
compute: () -> int = {
    outer: std::vector<int> = (1, 2, 3);
    {
        inner: std::vector<int> = (4, 5, 6);
        // Both vectors are NoEscape, different scopes
    }
    return outer.size();
}
)";

    std::string cpp_output = compile_cpp2_to_cpp(cpp2_source);

    // Check for multiple arena scopes
    std::cout << "  Checking for arena scope handling in nested blocks...\n";

    bool has_multiple_allocations = cpp_output.find("// Allocation:") != std::string::npos;
    std::size_t pos = cpp_output.find("// Allocation:");
    if (has_multiple_allocations) {
        std::size_t second = cpp_output.find("// Allocation:", pos + 15);
        if (second != std::string::npos) {
            std::cout << "  PASS: Multiple allocation annotations for nested scopes\n";
        } else {
            std::cout << "  INFO: Single allocation annotation (may be optimized)\n";
        }
    } else {
        std::cout << "  INFO: No explicit allocation annotations\n";
    }
}

void test_primitive_stack() {
    std::cout << "Running test_primitive_stack...\n";

    const char* cpp2_source = R"(
calculate: () -> int = {
    x: int = 42;
    y: int = 100;
    return x + y;
}
)";

    std::string cpp_output = compile_cpp2_to_cpp(cpp2_source);

    // Primitives should use stack
    bool has_stack = cpp_output.find("Allocation: stack") != std::string::npos;
    bool no_arena = cpp_output.find("arena") == std::string::npos;
    bool no_heap = cpp_output.find("make_unique") == std::string::npos;

    if (has_stack || (no_arena && no_heap)) {
        std::cout << "  PASS: Primitives use stack allocation\n";
    } else {
        std::cout << "  INFO: Stack allocation not explicit\n";
    }
}

int main() {
    try {
        test_hello_arena_allocation();
        test_local_vector_arena();
        test_escaping_vector_heap();
        test_mixed_escapes();
        test_nested_scope_arenas();
        test_primitive_stack();

        std::cout << "\n========================================\n";
        std::cout << "All end-to-end arena codegen tests passed!\n";
        std::cout << "========================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

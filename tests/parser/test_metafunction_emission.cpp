// Test: Metafunction Emission
// Verifies that metafunctions (@value, @interface, @ordered, etc.) are correctly emitted.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <sstream>

#include "lexer.hpp"
#include "slim_ast.hpp"
#include "emitter.hpp"

// Include implementations
#include "../../src/lexer.cpp"
#include "../../src/parser.cpp"
#include "../../src/emitter.cpp"

namespace test_metafunction_emission {

// Helper to check if output contains substring
bool contains(const std::string& str, const std::string& substr) {
    return str.find(substr) != std::string::npos;
}

// Test @value metafunction generates comparison operator
void test_value_metafunction_comparison() {
    std::cout << "Running test_value_metafunction_comparison..." << std::endl;
    std::string code = "widget: @value type = { val: int = 0; }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output:\n" << output << "\n";
    
    // @value should generate operator<=> for comparison
    assert(contains(output, "operator<=>") && "@value should generate operator<=>");
    assert(contains(output, "std::strong_ordering") && "@value should use strong_ordering");
    
    std::cout << "  PASS\n";
}

// Test @value metafunction generates copy operations
void test_value_metafunction_copy() {
    std::cout << "Running test_value_metafunction_copy..." << std::endl;
    std::string code = "widget: @value type = { val: int = 0; }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output:\n" << output << "\n";
    
    // @value should generate copy constructor and copy assignment (with or without public:)
    assert((contains(output, "widget(widget const&)") || contains(output, "widget(widget const& that)")) 
           && "@value should generate copy constructor");
    assert((contains(output, "operator=(widget const&)") || contains(output, "operator=(widget const& that)"))
           && "@value should generate copy assignment");
    
    std::cout << "  PASS\n";
}

// Test @value metafunction generates move operations
void test_value_metafunction_move() {
    std::cout << "Running test_value_metafunction_move..." << std::endl;
    std::string code = "widget: @value type = { val: int = 0; }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output:\n" << output << "\n";
    
    // @value should generate move constructor and move assignment (with or without public:)
    assert((contains(output, "widget(widget&&)") || contains(output, "widget(widget&& that)"))
           && "@value should generate move constructor");
    assert((contains(output, "operator=(widget&&)") || contains(output, "operator=(widget&& that)"))
           && "@value should generate move assignment");
    
    std::cout << "  PASS\n";
}

// Test @value metafunction generates default constructor
void test_value_metafunction_default_ctor() {
    std::cout << "Running test_value_metafunction_default_ctor..." << std::endl;
    std::string code = "widget: @value type = { val: int = 0; }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output:\n" << output << "\n";
    
    // @value should generate explicit default constructor
    assert(contains(output, "explicit widget()") && "@value should generate explicit default constructor");
    
    std::cout << "  PASS\n";
}

// Test @interface metafunction generates pure virtual methods
void test_interface_metafunction() {
    std::cout << "Running test_interface_metafunction..." << std::endl;
    std::string code = "Drawable: @interface type = { draw: (this); }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output:\n" << output << "\n";
    
    // @interface should generate pure virtual methods
    assert(contains(output, "virtual") && "@interface should generate virtual method");
    assert(contains(output, "= 0") && "@interface should generate pure virtual (= 0)");
    // @interface should generate virtual destructor
    assert(contains(output, "virtual ~Drawable()") && "@interface should generate virtual destructor");
    
    std::cout << "  PASS\n";
}

// Test @ordered metafunction generates comparison with weak_ordering
void test_ordered_metafunction() {
    std::cout << "Running test_ordered_metafunction..." << std::endl;
    std::string code = "widget: @ordered type = { val: int = 0; }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output:\n" << output << "\n";
    
    // @ordered should generate operator<=> with weak_ordering
    assert(contains(output, "operator<=>") && "@ordered should generate operator<=>");
    
    std::cout << "  PASS\n";
}

// Test @polymorphic_base metafunction
void test_polymorphic_base_metafunction() {
    std::cout << "Running test_polymorphic_base_metafunction..." << std::endl;
    std::string code = "Machine: @polymorphic_base type = { work: (virtual this); }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    std::string output = generate_from_tree(tree, tokens);
    std::cout << "Output:\n" << output << "\n";
    
    // @polymorphic_base should generate virtual destructor
    assert(contains(output, "virtual ~Machine()") && "@polymorphic_base should generate virtual destructor");
    // @polymorphic_base should delete copy operations
    assert((contains(output, "= delete") || contains(output, "delete")) && 
           "@polymorphic_base should delete copy/move to prevent slicing");
    
    std::cout << "  PASS\n";
}

} // namespace test_metafunction_emission

int main() {
    using namespace test_metafunction_emission;
    
    std::cout << "============================================\n";
    std::cout << "Metafunction Emission Tests\n";
    std::cout << "============================================\n\n";
    
    try {
        test_value_metafunction_comparison();
        test_value_metafunction_copy();
        test_value_metafunction_move();
        test_value_metafunction_default_ctor();
        test_interface_metafunction();
        test_ordered_metafunction();
        test_polymorphic_base_metafunction();
        
        std::cout << "\n============================================\n";
        std::cout << "All metafunction emission tests PASSED!\n";
        std::cout << "============================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAILED: " << e.what() << std::endl;
        return 1;
    }
}

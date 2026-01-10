// Test: Parameter Qualifier Parsing
// Verifies that parameter qualifiers (inout, out, move, forward) are parsed correctly
// and represented in the AST.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <optional>

#include "lexer.hpp"
#include "slim_ast.hpp"
#include "../../src/lexer.cpp"
// Include parser implementation directly to access internal helpers if needed, 
// and because the build system might not link the object file for standalone tests easily.
#include "../../src/parser.cpp"

namespace test_qualifiers {

using namespace cpp2::ast;

// Helper to find a specific node kind in children
std::optional<Node> find_child(const ParseTree& tree, const Node& parent, NodeKind kind) {
    for (const auto& child : tree.children(parent)) {
        if (child.kind == kind) {
            return child;
        }
    }
    return std::nullopt;
}

// Helper to get text content of a node
std::string get_text(const ParseTree& tree, const Node& node) {
    auto tokens = tree.node_tokens(node);
    if (tokens.empty()) return "";
    
    // Simple reconstruction (might miss whitespace)
    std::string result;
    for (const auto& token : tokens) {
        result += token.lexeme;
    }
    return result;
}

// Helper to find the first Parameter node in the tree
std::optional<Node> find_first_parameter(const ParseTree& tree) {
    for (const auto& node : tree.nodes) {
        if (node.kind == NodeKind::Parameter) {
            return node;
        }
    }
    return std::nullopt;
}

// Helper to print tree
void print_tree(const ParseTree& tree) {
    std::cout << "AST:\n";
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        const auto& node = tree.nodes[i];
        std::cout << "  Node " << i << ": " << cpp2::ast::meta::name(node.kind)
                  << " [" << node.token_start << "-" << node.token_end << "] "
                  << "Children: " << node.child_count;
        if (node.has_children()) {
            std::cout << " First: " << node.first_child;
        }
        std::cout << " Text: '" << get_text(tree, node) << "'\n";
    }
}

void test_inout_qualifier() {
    std::cout << "Running test_inout_qualifier..." << std::endl;
    std::string code = "my_func: (inout s: std::string) -> void = { }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();

    std::cout << "Tokens: ";
    for (const auto& t : tokens) {
        std::cout << "'" << t.lexeme << "'(" << (int)t.type << ") ";
    }
    std::cout << "\n";

    auto tree = cpp2::parser::parse(tokens);
    print_tree(tree);
    
    auto param_opt = find_first_parameter(tree);
    assert(param_opt.has_value() && "Should find a parameter");
    
    auto qualifier_opt = find_child(tree, *param_opt, NodeKind::ParamQualifier);
    assert(qualifier_opt.has_value() && "Parameter should have a qualifier");
    
    std::string qual_text = get_text(tree, *qualifier_opt);
    assert(qual_text == "inout" && "Qualifier should be 'inout'");
    
    std::cout << "  PASS\n";
}

void test_move_qualifier() {
    std::cout << "Running test_move_qualifier..." << std::endl;
    std::string code = "my_func: (move x: Widget) -> void = { }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    auto param_opt = find_first_parameter(tree);
    assert(param_opt.has_value() && "Should find a parameter");
    
    auto qualifier_opt = find_child(tree, *param_opt, NodeKind::ParamQualifier);
    assert(qualifier_opt.has_value() && "Parameter should have a qualifier");
    
    std::string qual_text = get_text(tree, *qualifier_opt);
    assert(qual_text == "move" && "Qualifier should be 'move'");
    
    std::cout << "  PASS\n";
}

void test_out_qualifier() {
    std::cout << "Running test_out_qualifier..." << std::endl;
    std::string code = "my_func: (out result: int) -> void = { }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    auto param_opt = find_first_parameter(tree);
    assert(param_opt.has_value() && "Should find a parameter");
    
    auto qualifier_opt = find_child(tree, *param_opt, NodeKind::ParamQualifier);
    assert(qualifier_opt.has_value() && "Parameter should have a qualifier");
    
    std::string qual_text = get_text(tree, *qualifier_opt);
    assert(qual_text == "out" && "Qualifier should be 'out'");
    
    std::cout << "  PASS\n";
}

void test_forward_qualifier() {
    std::cout << "Running test_forward_qualifier..." << std::endl;
    std::string code = "my_func: (forward x: T) -> void = { }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    auto param_opt = find_first_parameter(tree);
    assert(param_opt.has_value() && "Should find a parameter");
    
    auto qualifier_opt = find_child(tree, *param_opt, NodeKind::ParamQualifier);
    assert(qualifier_opt.has_value() && "Parameter should have a qualifier");
    
    std::string qual_text = get_text(tree, *qualifier_opt);
    assert(qual_text == "forward" && "Qualifier should be 'forward'");
    
    std::cout << "  PASS\n";
}



} // namespace test_qualifiers

int main() {
    std::cout << "=== Parameter Qualifier Tests ===\n";
    try {
        test_qualifiers::test_inout_qualifier();
        test_qualifiers::test_move_qualifier();
        test_qualifiers::test_out_qualifier();
        test_qualifiers::test_forward_qualifier();

    } catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << "=== All Tests Passed ===\n";
    return 0;
}

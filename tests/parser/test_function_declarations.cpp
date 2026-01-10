// Test: Function Declaration Parsing
// Verifies that function declarations are parsed correctly and represented in the AST.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <optional>

#include "lexer.hpp"
#include "slim_ast.hpp"

// Include implementations
#include "../../src/lexer.cpp"
#include "../../src/parser.cpp"

namespace test_functions {

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

void test_basic_function() {
    std::cout << "Running test_basic_function..." << std::endl;
    std::string code = "my_func: () -> int = { return 42; }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    // Check root has TranslationUnit
    assert(tree.nodes[0].kind == NodeKind::TranslationUnit);
    
    // Find Declaration -> UnifiedDeclaration -> FunctionSuffix
    auto decl = find_child(tree, tree.nodes[0], NodeKind::Declaration);
    assert(decl.has_value());
    
    auto unified = find_child(tree, *decl, NodeKind::UnifiedDeclaration);
    assert(unified.has_value());
    
    auto suffix = find_child(tree, *unified, NodeKind::FunctionSuffix);
    assert(suffix.has_value() && "Should find FunctionSuffix");
    
    // Check ParamList
    auto params = find_child(tree, *suffix, NodeKind::ParamList);
    assert(params.has_value());
    
    // Check ReturnSpec
    auto ret = find_child(tree, *suffix, NodeKind::ReturnSpec);
    assert(ret.has_value());
    assert(get_text(tree, *ret) == "->int");
    
    // Check FunctionBody
    auto body = find_child(tree, *suffix, NodeKind::FunctionBody);
    assert(body.has_value());
    
    std::cout << "  PASS\n";
}

void test_void_function() {
    std::cout << "Running test_void_function..." << std::endl;
    std::string code = "my_void_func: (x: int) -> void = { }";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    auto decl = find_child(tree, tree.nodes[0], NodeKind::Declaration);
    auto unified = find_child(tree, *decl, NodeKind::UnifiedDeclaration);
    auto suffix = find_child(tree, *unified, NodeKind::FunctionSuffix);
    assert(suffix.has_value());
    
    // Check ParamList has parameter
    auto params = find_child(tree, *suffix, NodeKind::ParamList);
    assert(params.has_value());
    auto param = find_child(tree, *params, NodeKind::Parameter);
    assert(param.has_value());
    
    // Check ReturnSpec
    auto ret = find_child(tree, *suffix, NodeKind::ReturnSpec);
    assert(ret.has_value());
    assert(get_text(tree, *ret) == "->void");
    
    std::cout << "  PASS\n";
}

void test_expression_body() {
    std::cout << "Running test_expression_body..." << std::endl;
    std::string code = "expr_func: () = 123;";
    
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    auto decl = find_child(tree, tree.nodes[0], NodeKind::Declaration);
    auto unified = find_child(tree, *decl, NodeKind::UnifiedDeclaration);
    auto suffix = find_child(tree, *unified, NodeKind::FunctionSuffix);
    assert(suffix.has_value());
    
    // Check FunctionBody
    auto body = find_child(tree, *suffix, NodeKind::FunctionBody);
    assert(body.has_value());
    
    // Body should contain expression (Literal 123)
    // Note: FunctionBody structure for expression body is "= expr ;"
    // We expect an Expression child inside FunctionBody
    bool found_expr = false;
    for (const auto& child : tree.children(*body)) {
        if (meta::is_expression(child.kind) || child.kind == NodeKind::Literal) {
            found_expr = true;
            assert(get_text(tree, child) == "123");
            break;
        }
    }
    assert(found_expr && "Should find expression in function body");
    
    std::cout << "  PASS\n";
}

} // namespace test_functions

int main() {
    std::cout << "=== Function Declaration Tests ===\n";
    try {
        test_functions::test_basic_function();
        test_functions::test_void_function();
        test_functions::test_expression_body();
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << "=== All Tests Passed ===\n";
    return 0;
}

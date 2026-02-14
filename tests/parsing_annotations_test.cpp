
#include "../include/combinators/spirit.hpp"
#include "../include/slim_ast.hpp"
#include <iostream>
#include <vector>
#include <cassert>

using namespace cpp2::parser::spirit;
using namespace cpp2::ast;

void test_simple_node() {
    // Reset builder
    g_builder = TreeBuilder{};
    
    std::vector<cpp2_transpiler::Token> tokens = {
        {cpp2_transpiler::TokenType::Identifier, "id", 1, 1, 0}
    };
    TokenStream input{tokens};
    
    auto p = tok(cpp2_transpiler::TokenType::Identifier) % with_node(NodeKind::Identifier);
    
    auto res = p.parse(input);
    if (!res.success()) {
        std::cerr << "Parse failed\n";
        exit(1);
    }
    
    ParseTree tree = g_builder.finish(tokens);
    if (tree.nodes.size() != 1) {
        std::cerr << "Expected 1 node, got " << tree.nodes.size() << "\n";
        exit(1);
    }
    if (tree.nodes[0].kind != NodeKind::Identifier) {
        std::cerr << "Expected Identifier node\n";
        exit(1);
    }
    
    std::cout << "test_simple_node PASS\n";
}

void test_nested_nodes() {
    // Reset builder
    g_builder = TreeBuilder{};
    
    // ( id )
    std::vector<cpp2_transpiler::Token> tokens = {
        {cpp2_transpiler::TokenType::Unknown, "(", 1, 1, 0},
        {cpp2_transpiler::TokenType::Identifier, "id", 1, 3, 0},
        {cpp2_transpiler::TokenType::Unknown, ")", 1, 6, 0}
    };
    TokenStream input{tokens};
    
    auto id_parser = tok(cpp2_transpiler::TokenType::Identifier) % with_node(NodeKind::Identifier);
    auto group_parser = (lit("(") >> id_parser >> ")") % with_node(NodeKind::GroupedExpression);
    
    auto res = group_parser.parse(input);
    assert(res.success());
    
    ParseTree tree = g_builder.finish(tokens);
    // Tree should be: GroupedExpression -> [ Identifier ]
    // nodes storage: [ Identifier, GroupedExpression ] (children stored first? No, builder pushes.
    // begin(Group) -> push Group.
    // (
    // begin(Id) -> push Id.
    // end(Id) -> pop Id. Group adopts Id.
    // )
    // end(Group) -> pop Group.
    // Vector order: Identifier, GroupedExpression.
    
    assert(tree.nodes.size() == 2);
    assert(tree.nodes[0].kind == NodeKind::GroupedExpression);
    assert(tree.nodes[1].kind == NodeKind::Identifier);
    
    assert(tree.nodes[0].child_count == 1);
    assert(tree.nodes[0].child_start == 1);
    
    std::cout << "test_nested_nodes PASS\n";
}

void test_binary_node() {
    // Reset builder
    g_builder = TreeBuilder{};
    
    // a + b
    std::vector<cpp2_transpiler::Token> tokens = {
        {cpp2_transpiler::TokenType::Identifier, "a", 1, 1, 0},
        {cpp2_transpiler::TokenType::Plus, "+", 1, 3, 0},
        {cpp2_transpiler::TokenType::Identifier, "b", 1, 5, 0}
    };
    TokenStream input{tokens};
    
    // Simple definitions
    auto id = tok(cpp2_transpiler::TokenType::Identifier) % with_node(NodeKind::Identifier);
    auto plus = tok(cpp2_transpiler::TokenType::Plus);
    
    auto binary_part = (plus >> id) % with_binary(NodeKind::AdditiveExpression);
    // Wrap in a parent node to allow infix adoption
    auto expr = (id >> binary_part) % with_node(NodeKind::Expression);
    
    auto res = expr.parse(input);
    assert(res.success());
    
    ParseTree tree = g_builder.finish(tokens);
    
    // Tree:
    // 0: Expression
    // Children of Expression: [AdditiveExpression]
    // 1: AdditiveExpression (was Id slot, overwritten)
    // Children of Additive: [Id(a), Id(b)]
    // 2: Id(a) (relocated)
    // 3: Id(b) (new RHS)
    
    assert(tree.nodes.size() == 4);
    assert(tree.nodes[0].kind == NodeKind::Expression);
    
    // Check Expression children
    assert(tree.nodes[0].child_count == 1);
    uint32_t child1_idx = tree.nodes[0].child_start;
    assert(tree.nodes[child1_idx].kind == NodeKind::AdditiveExpression);
    
    // Check AdditiveExpression children
    assert(tree.nodes[child1_idx].child_count == 2);
    uint32_t lhs_idx = tree.nodes[child1_idx].child_start;
    
    assert(tree.nodes[lhs_idx].kind == NodeKind::Identifier);
    assert(tree.nodes[lhs_idx].token_start == 0); // 'a'
    
    // RHS is next sibling of LHS in vector?
    // start_infix logic: "nodes_.push_back(lhs_copy); nodes_[infix].child_start = new_lhs_idx;"
    // RHS is added via standard 'begin' -> 'push_back'.
    // So LHS is at new_lhs_idx (2). RHS is at 3.
    // They are contiguous.
    assert(tree.nodes[lhs_idx + 1].kind == NodeKind::Identifier);
    assert(tree.nodes[lhs_idx + 1].token_start == 2); // 'b'
    
    std::cout << "test_binary_node PASS\n";
}

struct MyClangType {};

void test_type_hint() {
    g_builder = TreeBuilder{};
    std::vector<cpp2_transpiler::Token> tokens = {
        {cpp2_transpiler::TokenType::Identifier, "id", 1, 1, 0}
    };
    TokenStream input{tokens};
    
    // id % ast_node<MyClangType>() % with_node(Identifier)
    // Order of %: (id % ast_node) % with_node.
    // ast_node is strict operator% overload returning P.
    auto p = tok(cpp2_transpiler::TokenType::Identifier) % ast_node<MyClangType>() % with_node(NodeKind::Identifier);
    
    auto res = p.parse(input);
    assert(res.success());
    
    ParseTree tree = g_builder.finish(tokens);
    assert(tree.nodes.size() == 1);
    assert(tree.nodes[0].kind == NodeKind::Identifier);
    
    std::cout << "test_type_hint PASS\n";
}

int main() {
    test_simple_node();
    test_nested_nodes();
    test_binary_node();
    test_type_hint();
    return 0;
}

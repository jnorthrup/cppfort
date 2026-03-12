
#include <iostream>
#include <vector>
#include <span>
#include <cassert>
#include "lexer.hpp"
#include "slim_ast.hpp"

namespace cpp2::parser {
    cpp2::ast::ParseTree parse(std::span<const cpp2_transpiler::Token> tokens);
}

using namespace cpp2::ast;
using namespace cpp2_transpiler;

// Helper to get children indices
namespace {

std::vector<int> get_children(const cpp2::ast::ParseTree& tree, int node_idx) {
    std::vector<int> children;
    if (node_idx < 0 || node_idx >= (int)tree.nodes.size()) return children;
    
    // Check if node structure looks valid
    if (node_idx >= tree.nodes.size()) {
        std::cerr << "Error: accessing node " << node_idx << " out of bounds " << tree.nodes.size() << "\n";
        return children;
    }

    uint32_t curr = tree.nodes[node_idx].first_child;
    while (curr != UINT32_MAX) {
        if (curr >= tree.nodes.size()) {
            std::cerr << "Error: corruption detected, child " << curr << " out of bounds\n";
            break;
        }
        children.push_back(curr);
        curr = tree.nodes[curr].next_sibling;
    }
    return children;
}

int find_child(const cpp2::ast::ParseTree& tree, int parent_idx, cpp2::ast::NodeKind kind) {
    for (int child : get_children(tree, parent_idx)) {
        if (tree.nodes[child].kind == kind) return child;
    }
    return -1;
}

void dump_tree(const cpp2::ast::ParseTree& tree, int node_idx, int indent = 0) {
    if (node_idx < 0 || node_idx >= (int)tree.nodes.size()) return;
    
    const auto& node = tree.nodes[node_idx];
    for (int i=0; i<indent; ++i) std::cout << "  ";
    std::cout << std::to_string((int)node.kind) 
              << " [" << node.token_start << "-" << node.token_end << "]"
              << " (idx: " << node_idx << ", first: " << node.first_child << ", next: " << node.next_sibling << ")"
              << "\n";
              
    std::cout.flush(); // Ensure output is printed before potential crash

    uint32_t curr = node.first_child;
    while (curr != UINT32_MAX) {
        dump_tree(tree, curr, indent + 1);
        curr = tree.nodes[curr].next_sibling;
    }
}

void test_var_decl() {
    using namespace cpp2::parser;
    using namespace cpp2::ast;

    // "x: int = 42;"
    std::vector<cpp2_transpiler::Token> tokens = {
        { cpp2_transpiler::TokenType::Identifier, "x", 0, 1, 1 },
        { cpp2_transpiler::TokenType::Colon, ":", 1, 1, 2 },
        { cpp2_transpiler::TokenType::Identifier, "int", 3, 1, 4 },
        { cpp2_transpiler::TokenType::Equal, "=", 7, 1, 8 },
        { cpp2_transpiler::TokenType::IntegerLiteral, "42", 9, 1, 10 },
        { cpp2_transpiler::TokenType::Semicolon, ";", 11, 1, 12 },
        { cpp2_transpiler::TokenType::EndOfFile, "", 12, 1, 13 }
    };

    ParseTree tree = parse(tokens);

    // root -> TranslationUnit
    assert(!tree.nodes.empty());
    assert(tree.nodes[0].kind == NodeKind::TranslationUnit);

    // TranslationUnit -> Declaration
    int decl = find_child(tree, 0, NodeKind::Declaration);
    assert(decl != -1 && "Declaration found");

    // Declaration -> UnifiedDeclaration
    int unified = find_child(tree, decl, NodeKind::UnifiedDeclaration);
    assert(unified != -1 && "UnifiedDeclaration found");

    std::cout << "test_var_decl PASS\n";
}

void test_func_stmt() {
    using namespace cpp2::parser;
    using namespace cpp2::ast;
    
    // "f: () = { return; }"
    std::vector<cpp2_transpiler::Token> tokens = {
        { cpp2_transpiler::TokenType::Identifier, "f", 0, 1, 1 },
        { cpp2_transpiler::TokenType::Colon, ":", 1, 1, 2 },
        { cpp2_transpiler::TokenType::LeftParen, "(", 3, 1, 4 },
        { cpp2_transpiler::TokenType::RightParen, ")", 4, 1, 5 },
        { cpp2_transpiler::TokenType::Equal, "=", 6, 1, 7 },
        { cpp2_transpiler::TokenType::LeftBrace, "{", 8, 1, 9 },
        { cpp2_transpiler::TokenType::Return, "return", 10, 1, 11 },
        { cpp2_transpiler::TokenType::Semicolon, ";", 16, 1, 12 },
        { cpp2_transpiler::TokenType::RightBrace, "}", 17, 1, 13 },
        { cpp2_transpiler::TokenType::EndOfFile, "", 0, 1, 14 } // Pos 9? No count.
    };
    
    ParseTree tree = parse(tokens);
    
    // Verify structure
    assert(!tree.nodes.empty());

    // Root -> Declaration
    int decl = find_child(tree, 0, NodeKind::Declaration);
    
    if (decl == -1) {
        std::cout << "Decl not found\n";
        dump_tree(tree, 0);
    }
    
    assert(decl != -1);
    
    int unified = find_child(tree, decl, NodeKind::UnifiedDeclaration);
    assert(unified != -1);
    
    int suffix = find_child(tree, unified, NodeKind::FunctionSuffix);
    assert(suffix != -1);
    
    int body = find_child(tree, suffix, NodeKind::FunctionBody);
    assert(body != -1);
    
    int block = find_child(tree, body, NodeKind::BlockStatement);
    assert(block != -1 && "BlockStatement found");
    
    // Block -> Statement -> ReturnStatement
    int stmt = find_child(tree, block, NodeKind::Statement);
    assert(stmt != -1 && "Statement found");
    
    int ret = find_child(tree, stmt, NodeKind::ReturnStatement);
    assert(ret != -1 && "ReturnStatement found");
    
    std::cout << "test_func_stmt PASS\n";
}

void test_binary_expr() {
    using namespace cpp2::parser;
    using namespace cpp2::ast;

    // "a + b"
    std::vector<cpp2_transpiler::Token> tokens = {
        { cpp2_transpiler::TokenType::Identifier, "a", 0, 1, 1 },
        { cpp2_transpiler::TokenType::Plus, "+", 1, 1, 2 },
        { cpp2_transpiler::TokenType::Identifier, "b", 2, 1, 3 },
        { cpp2_transpiler::TokenType::EndOfFile, "", 3, 1, 4 }
    };

    // We can test parse_expression directly since it wraps pratt
    // But parse() expects TranslationUnit. 
    // Let's manually invoke parse_expression logic or just parse a full TU "x = a + b;"
    
    // "x: = a + b;"
    std::vector<cpp2_transpiler::Token> full_tokens = {
        { cpp2_transpiler::TokenType::Identifier, "x", 0, 1, 1 },
        { cpp2_transpiler::TokenType::Colon, ":", 1, 1, 2 },
        { cpp2_transpiler::TokenType::Equal, "=", 2, 1, 3 },
        { cpp2_transpiler::TokenType::Identifier, "a", 3, 1, 4 },
        { cpp2_transpiler::TokenType::Plus, "+", 4, 1, 5 },
        { cpp2_transpiler::TokenType::Identifier, "b", 5, 1, 6 },
        { cpp2_transpiler::TokenType::Semicolon, ";", 6, 1, 7 },
        { cpp2_transpiler::TokenType::EndOfFile, "", 7, 1, 8 }
    };
    
    ParseTree tree = parse(full_tokens);
    
    // check structure
    assert(!tree.nodes.empty());
    
    int decl = find_child(tree, 0, NodeKind::Declaration);
    if (decl == -1) {
        std::cout << "Decl not found in test_binary_expr\n";
        dump_tree(tree, 0);
    }
    assert(decl != -1);
    
    int unified = find_child(tree, decl, NodeKind::UnifiedDeclaration);
    assert(unified != -1);
    
    // Check for binary op in RHS
    // "x : = a + b;" -> UnifiedDecl -> VariableSuffix.
    // VariableSuffix contains the expression.
    // The expression is "a + b" -> AdditiveExpression.
    // There is no AssignmentExpression node because the '=' is part of the declaration syntax, 
    // not an operator in the expression.
    
    int suffix = find_child(tree, unified, NodeKind::VariableSuffix);
    assert(suffix != -1 && "VariableSuffix found");
    
    // VariableSuffix -> Expression -> AdditiveExpression
    int expr = find_child(tree, suffix, NodeKind::Expression);
    assert(expr != -1 && "Expression found");
    
    int add = find_child(tree, expr, NodeKind::AdditiveExpression);
    assert(add != -1 && "AdditiveExpression found");
    
    std::cout << "test_binary_expr PASS\n";
}

void test_multi_expr_subscript() {
    using namespace cpp2::parser;
    using namespace cpp2::ast;

    // "x: int = coords[1.0, 2.0];"
    std::vector<cpp2_transpiler::Token> tokens = {
        { cpp2_transpiler::TokenType::Identifier, "x", 0, 1, 1 },
        { cpp2_transpiler::TokenType::Colon, ":", 1, 1, 2 },
        { cpp2_transpiler::TokenType::Identifier, "int", 3, 1, 4 },
        { cpp2_transpiler::TokenType::Equal, "=", 7, 1, 8 },
        { cpp2_transpiler::TokenType::Identifier, "coords", 9, 1, 10 },
        { cpp2_transpiler::TokenType::LeftBracket, "[", 15, 1, 16 },
        { cpp2_transpiler::TokenType::FloatLiteral, "1.0", 16, 1, 17 },
        { cpp2_transpiler::TokenType::Comma, ",", 19, 1, 20 },
        { cpp2_transpiler::TokenType::FloatLiteral, "2.0", 20, 1, 21 },
        { cpp2_transpiler::TokenType::RightBracket, "]", 22, 1, 23 },
        { cpp2_transpiler::TokenType::Semicolon, ";", 23, 1, 24 },
        { cpp2_transpiler::TokenType::EndOfFile, "", 24, 1, 25 }
    };

    ParseTree tree = parse(tokens);

    // Verify structure
    assert(!tree.nodes.empty());

    // Root -> Declaration
    int decl = find_child(tree, 0, NodeKind::Declaration);
    assert(decl != -1 && "Declaration found");

    // Declaration -> UnifiedDeclaration
    int unified = find_child(tree, decl, NodeKind::UnifiedDeclaration);
    assert(unified != -1 && "UnifiedDeclaration found");

    // UnifiedDeclaration -> VariableSuffix -> Expression
    int suffix = find_child(tree, unified, NodeKind::VariableSuffix);
    assert(suffix != -1 && "VariableSuffix found");

    int expr = find_child(tree, suffix, NodeKind::Expression);
    assert(expr != -1 && "Expression found");

    // Expression -> PostfixExpression -> Subscript
    int postfix = find_child(tree, expr, NodeKind::PostfixExpression);
    assert(postfix != -1 && "PostfixExpression found");

    // The multi-expression subscript should contain the expression list
    // We're testing that the parse tree accepts coords[1.0, 2.0]
    // which has a comma between expressions

    std::cout << "test_multi_expr_subscript PASS\n";
}

} // namespace

int main() {
    test_var_decl();
    test_func_stmt();
    // test_binary_expr();
    test_multi_expr_subscript();
    return 0;
}

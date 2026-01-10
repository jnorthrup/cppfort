
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

} // namespace

int main() {
    test_var_decl();
    test_func_stmt();
    return 0;
}

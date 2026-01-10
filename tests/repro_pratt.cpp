
#include "../src/parser.cpp"
#include <iostream>
#include <vector>
#include <string>

// Helper to compile this test:
// clang++ -std=c++20 -Iinclude tests/repro_pratt.cpp -o repro_pratt

void print_tree(const cpp2::ast::ParseTree& tree, uint32_t node_idx, int depth = 0) {
    const auto& node = tree.nodes[node_idx];
    for (int i = 0; i < depth; ++i) std::cout << "  ";
    std::cout << cpp2::ast::meta::name(node.kind);
    if (node.token_count() > 0) {
        std::cout << " '";
        for (auto t : tree.node_tokens(node)) {
            std::cout << t.lexeme;
        }
        std::cout << "'";
    }
    std::cout << "\n";

    if (node.has_children()) {
        auto children = tree.children(node);
        for (const auto& child : children) {
            // We need to find the index of the child to recurse safely if we had indices
            // But ParseTree::children returns spans of Nodes.
            // We need to calculate index for recursion if we printed by index?
            // Actually, we can just recurse on child.
            // But child is a copy or ref? It's a const Node& from the span.
            // Wait, ParseTree::children returns span<const Node>.
            // But I want to print indices or just structure.
            // Structure is fine.
            // But to recurse I need to know if it's a leaf.
            // Node has child_start/child_count.
            // To recurse I need to look up children in tree.nodes based on child.child_start
            
            // Wait, the `child` object itself contains `child_start` index.
            // So I can recurse.
        }
        // Correct iteration:
        for (uint32_t i = 0; i < node.child_count; ++i) {
             print_tree(tree, node.child_start + i, depth + 1);
        }
    }
}

int main() {

    // Test 1: Precedence 1 + 2 * 3
    std::string source1 = "files: () = 1 + 2 * 3;";
    std::vector<cpp2_transpiler::Token> tokens1 = {
        {cpp2_transpiler::TokenType::Identifier, "files", 1, 1, 0},
        {cpp2_transpiler::TokenType::Colon, ":", 1, 6, 5},
        {cpp2_transpiler::TokenType::LeftParen, "(", 1, 8, 7},
        {cpp2_transpiler::TokenType::RightParen, ")", 1, 9, 8},
        {cpp2_transpiler::TokenType::Equal, "=", 1, 11, 10},
        {cpp2_transpiler::TokenType::IntegerLiteral, "1", 1, 13, 12},
        {cpp2_transpiler::TokenType::Plus, "+", 1, 15, 14},
        {cpp2_transpiler::TokenType::IntegerLiteral, "2", 1, 17, 16},
        {cpp2_transpiler::TokenType::Asterisk, "*", 1, 19, 18},
        {cpp2_transpiler::TokenType::IntegerLiteral, "3", 1, 21, 20},
        {cpp2_transpiler::TokenType::Semicolon, ";", 1, 23, 22},
        {cpp2_transpiler::TokenType::EndOfFile, "", 1, 24, 23}
    };

    std::cout << "Test 1: 1 + 2 * 3" << std::endl;
    auto tree1 = cpp2::parser::parse(tokens1);
    print_tree(tree1, tree1.root);

    // Test 2: Postfix Dereference: ptr * ;
    std::vector<cpp2_transpiler::Token> tokens2 = {
        {cpp2_transpiler::TokenType::Identifier, "f", 1, 1, 0},
        {cpp2_transpiler::TokenType::Colon, ":", 1, 6, 5},
        {cpp2_transpiler::TokenType::LeftParen, "(", 1, 8, 7},
        {cpp2_transpiler::TokenType::RightParen, ")", 1, 9, 8},
        {cpp2_transpiler::TokenType::Equal, "=", 1, 11, 10},
        {cpp2_transpiler::TokenType::Identifier, "ptr", 1, 13, 12},
        {cpp2_transpiler::TokenType::Asterisk, "*", 1, 16, 15},
        {cpp2_transpiler::TokenType::Semicolon, ";", 1, 17, 16},
        {cpp2_transpiler::TokenType::EndOfFile, "", 1, 18, 17}
    };

    std::cout << "\nTest 2: ptr * ;" << std::endl;
    auto tree2 = cpp2::parser::parse(tokens2);
    print_tree(tree2, tree2.root);

    // Test 3: Mixed: a * * b ;  (a * (*b))
    // Tokens: a, *, *, b, ;
    std::vector<cpp2_transpiler::Token> tokens3 = {
        {cpp2_transpiler::TokenType::Identifier, "f", 1, 1, 0},
        {cpp2_transpiler::TokenType::Colon, ":", 1, 6, 5},
        {cpp2_transpiler::TokenType::LeftParen, "(", 1, 8, 7},
        {cpp2_transpiler::TokenType::RightParen, ")", 1, 9, 8},
        {cpp2_transpiler::TokenType::Equal, "=", 1, 11, 10},
        {cpp2_transpiler::TokenType::Identifier, "a", 1, 13, 12},
        {cpp2_transpiler::TokenType::Asterisk, "*", 1, 15, 14},
        {cpp2_transpiler::TokenType::Asterisk, "*", 1, 17, 16},
        {cpp2_transpiler::TokenType::Identifier, "b", 1, 19, 18},
        {cpp2_transpiler::TokenType::Semicolon, ";", 1, 20, 19},
        {cpp2_transpiler::TokenType::EndOfFile, "", 1, 21, 20}
    };

    std::cout << "\nTest 3: a * * b ;" << std::endl;
    auto tree3 = cpp2::parser::parse(tokens3);
    print_tree(tree3, tree3.root);


    return 0;
}

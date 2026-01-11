#include "combinator_parser.hpp"
#include "lexer.hpp"
#include "slim_ast.hpp"
#include <iostream>

using namespace cpp2::ast;

void debug_for_tree() {
    std::string code = R"(
main: () -> void = {
    for items do (x) {
        return x;
    }
}
)";

    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);

    std::cout << "=== Parse Tree Debug ===\n";
    std::cout << "Total nodes: " << tree.nodes.size() << "\n";
    std::cout << "Root index: " << tree.root << "\n\n";

    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        const auto& node = tree.nodes[i];
        std::cout << "Node " << i << ": ";
        std::cout << "kind=" << static_cast<int>(node.kind);
        std::cout << ", child_count=" << node.child_count;
        std::cout << ", token_start=" << node.token_start;
        std::cout << ", token_end=" << node.token_end;

        // Print token text for this node
        if (node.token_start < tokens.size() && node.token_end <= tokens.size()) {
            std::cout << ", text=\"";
            for (size_t t = node.token_start; t < node.token_end; ++t) {
                std::cout << tokens[t].lexeme;
            }
            std::cout << "\"";
        }
        std::cout << "\n";
    }

    // Find ForStatement node and print its children
    std::cout << "\n=== ForStatement Children ===\n";
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        if (tree.nodes[i].kind == NodeKind::ForStatement) {
            std::cout << "ForStatement at index " << i << ":\n";
            for (const auto& child : tree.children(tree.nodes[i])) {
                std::cout << "  - kind=" << static_cast<int>(child.kind);
                std::cout << ", text=\"";
                if (child.token_start < tokens.size() && child.token_end <= tokens.size()) {
                    for (size_t t = child.token_start; t < child.token_end; ++t) {
                        std::cout << tokens[t].lexeme;
                    }
                }
                std::cout << "\"\n";
            }
        }
    }
}

int main() {
    debug_for_tree();
    return 0;
}

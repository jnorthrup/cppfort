// Test: Pratt Parser Expressions
// Verifies that the Pratt parser handles all operator precedence levels correctly.
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

namespace test_pratt {

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
    std::string result;
    for (const auto& token : tokens) {
        result += token.lexeme;
    }
    return result;
}

// Check if a node text matches expected
void check_node_text(const ParseTree& tree, const Node& node, const std::string& expected) {
    std::string text = get_text(tree, node);
    if (text != expected) {
        std::cerr << "Node text mismatch: expected '" << expected << "', got '" << text << "'\n";
        std::exit(1);
    }
}

// Helper to verify binary expression structure
// Expected: (LHS) (OP) (RHS)
void verify_binary(const ParseTree& tree, const Node& expr_node, NodeKind expr_kind, 
                   const std::string& op_text) {
    assert(expr_node.kind == expr_kind);
    
    // Infix structure: LHS (child 0), Op (child 1), RHS (child 2)
    // Actually, `start_infix` pushes Infix node which has LHS as first child.
    // Then `parse_pratt` consumes op and RHS, adding them as children.
    // So children are: LHS, Op, RHS.
    
    int count = 0;
    for (const auto& child : tree.children(expr_node)) {
        if (count == 1) { // Op
             // Op is wrapped in BinaryOp/AssignmentOp/etc node?
             // parser.cpp: begin(op_kind); input.next(); end();
             // So child 1 is the Op node.
             assert(get_text(tree, child) == op_text);
        }
        count++;
    }
    assert(count == 3 && "Binary expression should have 3 children");
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

void test_binary_operators() {
    std::cout << "Running test_binary_operators..." << std::endl;
    // Simple expression: 1 + 2 * 3
    // Precedence: 1 + (2 * 3)
    // std::string code = "x = 1 + 2 * 3;";
    // cpp2_transpiler::Lexer lexer(code);
    // auto tokens = lexer.tokenize();
    // auto tree = cpp2::parser::parse(tokens);
    
    // Structure:
    // AssignmentExpression (=)
    //   Identifier (x)
    //   AssignmentOp (=)
    //   AdditiveExpression (+)
    //     Literal (1)
    //     BinaryOp (+)
    //     MultiplicativeExpression (*)
    //       Literal (2)
    //       BinaryOp (*)
    //       Literal (3)

    // Find the assignment expression
    // It's inside a Statement -> ExpressionStatement -> AssignmentExpression (or similar)
    // Actually, `x = ...;` is parsed as AssignmentExpression inside ExpressionStatement if top level?
    // Wait, top level is TranslationUnit -> Declaration -> UnifiedDeclaration?
    // "x = ..." is likely "x: auto = ..." deduction if top level, OR ExpressionStatement if inside block.
    // The parser handles top-level declarations.
    // `x = 1 + 2 * 3;` at top level is NOT a valid declaration in Cpp2 unless it's `x := ...` or inside function.
    // Let's use a variable declaration: `x: int = 1 + 2 * 3;`
    
    std::string code = "x: int = 1 + 2 * 3;";
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);

    print_tree(tree);
    
    // Find VariableSuffix -> init expression
    // Declaration -> UnifiedDeclaration -> VariableSuffix -> (children)
    // One child is TypeSpecifier, next is "=" (maybe), next is Expression.
    
    // Let's dig for the Expression node
    const Node* expr_root = nullptr;
    for (const auto& node : tree.nodes) {
        if (node.kind == NodeKind::AdditiveExpression) {
            expr_root = &node;
            break;
        }
    }
    assert(expr_root != nullptr && "Should find AdditiveExpression");
    
    // Verify 1 + (2 * 3) structure
    // AdditiveExpression (+)
    //   LHS: Literal (1)
    //   Op: BinaryOp (+)
    //   RHS: MultiplicativeExpression (*)
    
    int child_idx = 0;
    const Node* rhs_node = nullptr;
    for (const auto& child : tree.children(*expr_root)) {
        std::cout << "Child " << child_idx << ": Kind=" << cpp2::ast::meta::name(child.kind) << " Text='" << get_text(tree, child) << "'\n";
        if (child_idx == 0) {
            assert(child.kind == NodeKind::Literal);
            assert(get_text(tree, child) == "1");
        } else if (child_idx == 1) {
            // Op is a node (BinaryOp) wrapping the token
            assert(child.kind == NodeKind::BinaryOp);
            assert(get_text(tree, child) == "+");
        } else if (child_idx == 2) {
            assert(child.kind == NodeKind::MultiplicativeExpression);
            rhs_node = &child;
        }
        child_idx++;
    }
    
    assert(rhs_node != nullptr);
    
    // Verify 2 * 3
    child_idx = 0;
    for (const auto& child : tree.children(*rhs_node)) {
        if (child_idx == 0) {
            assert(child.kind == NodeKind::Literal);
            assert(get_text(tree, child) == "2");
        } else if (child_idx == 1) {
            assert(child.kind == NodeKind::BinaryOp);
            assert(get_text(tree, child) == "*");
        } else if (child_idx == 2) {
            assert(child.kind == NodeKind::Literal);
            assert(get_text(tree, child) == "3");
        }
        child_idx++;
    }

    std::cout << "  PASS\n";
}

void test_comparison_logic() {
    std::cout << "Running test_comparison_logic..." << std::endl;
    // x: bool = a < b && c == d;
    // Precedence: (a < b) && (c == d)
    // && is LogicalAndExpression
    // < is ComparisonExpression
    // == is EqualityExpression
    // CMP > EQ ? No, CMP (10) > EQ (9)? 
    // Wait, let's check parser.cpp precedence.
    // EQ=9, CMP=10. So CMP binds tighter than EQ? 
    // Usually < binds tighter than == in C++.
    // 1 < 2 == true  -> (1 < 2) == true. Correct.
    
    std::string code = "x: bool = a < b && c == d;";
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    // Root should be LogicalAndExpression (&&)
    const Node* root = nullptr;
    for (const auto& node : tree.nodes) {
        if (node.kind == NodeKind::LogicalAndExpression) {
            root = &node;
            break;
        }
    }
    assert(root != nullptr && "Should find LogicalAndExpression");
    
    // Check children: Comparison (lhs), &&, Equality (rhs)
    // Wait, && (LogicAnd) binds looser than Equality.
    // So LHS = (a < b), RHS = (c == d) ? 
    // No, Equality (==) binds looser than Comparison (<).
    // So (c == d) is EqualityExpression.
    
    // LHS of && should be ComparisonExpression? 
    // Actually, Pratt parser builds up.
    // Input: a < b && c == d
    // 1. parse a.
    // 2. see < (prec 10). recurse.
    //    parse b.
    //    see && (prec 5). < has higher prec. return (a < b).
    // 3. see && (prec 5).
    //    LHS = (a < b).
    //    recurse for RHS.
    //    parse c.
    //    see == (prec 9). recurse.
    //    parse d.
    //    see ; (prec 0). return (c == d).
    //    return (c == d).
    // 4. build (a < b) && (c == d).
    
    // So LHS is ComparisonExpression (a < b).
    // RHS is EqualityExpression (c == d).
    
    int child_idx = 0;
    for (const auto& child : tree.children(*root)) {
        if (child_idx == 0) {
            assert(child.kind == NodeKind::ComparisonExpression);
            assert(get_text(tree, child) == "a<b");
        } else if (child_idx == 1) {
            assert(child.kind == NodeKind::BinaryOp);
            assert(get_text(tree, child) == "&&");
        } else if (child_idx == 2) {
            assert(child.kind == NodeKind::EqualityExpression);
            assert(get_text(tree, child) == "c==d");
        }
        child_idx++;
    }
    
    std::cout << "  PASS\n";
}

void test_assignment() {
    std::cout << "Running test_assignment..." << std::endl;
    // x += y = z
    // Assignment is right-associative.
    // x += (y = z)
    // This requires parsing statement "x += y = z;"
    // We can wrap it in a function body to ensure it parses as statement.
    
    std::string code = "f: () = { x += y = z; }";
    cpp2_transpiler::Lexer lexer(code);
    auto tokens = lexer.tokenize();
    auto tree = cpp2::parser::parse(tokens);
    
    // Find AssignmentExpression
    // Outer: x += ...
    const Node* outer = nullptr;
    // We need to be careful to find the *first* (outer) assignment.
    // In AST nodes vector, children usually come after parent? Or before?
    // TreeBuilder pushes children then parent.
    // So parent is later in the vector.
    // We should search from back or find the one that spans the whole statement.
    
    for (auto it = tree.nodes.rbegin(); it != tree.nodes.rend(); ++it) {
        if (it->kind == NodeKind::AssignmentExpression) {
            outer = &(*it);
            break;
        }
    }
    assert(outer != nullptr);
    
    // Verify outer is +=
    // LHS: x
    // RHS: AssignmentExpression (=)
    
    int child_idx = 0;
    const Node* rhs = nullptr;
    for (const auto& child : tree.children(*outer)) {
        if (child_idx == 0) {
            assert(get_text(tree, child) == "x");
        } else if (child_idx == 1) {
            assert(child.kind == NodeKind::AssignmentOp);
            assert(get_text(tree, child) == "+=");
        } else if (child_idx == 2) {
            assert(child.kind == NodeKind::AssignmentExpression);
            rhs = &child;
        }
        child_idx++;
    }
    assert(rhs != nullptr);
    
    // Verify inner is =
    child_idx = 0;
    for (const auto& child : tree.children(*rhs)) {
        if (child_idx == 0) {
            assert(get_text(tree, child) == "y");
        } else if (child_idx == 1) {
            assert(child.kind == NodeKind::AssignmentOp);
            assert(get_text(tree, child) == "=");
        } else if (child_idx == 2) {
            assert(get_text(tree, child) == "z");
        }
        child_idx++;
    }
    
    std::cout << "  PASS\n";
}

} // namespace test_pratt

int main() {
    std::cout << "=== Pratt Parser Expression Tests ===\n";
    try {
        test_pratt::test_binary_operators();
        test_pratt::test_comparison_logic();
        test_pratt::test_assignment();
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << "=== All Tests Passed ===\n";
    return 0;
}

// Semantic analysis pass for Cpp2 AST
// Performs escape analysis, lifetime tracking, and other semantic checks

#include "../include/ast.hpp"
#include <unordered_map>
#include <string>

namespace cpp2_transpiler {

// Helper class to track variable usage during escape analysis
class EscapeAnalyzer {
public:
    std::unordered_map<std::string, EscapeInfo> variable_escapes;

    // Analyze a single statement for variable escapes
    void analyze_statement(Statement* stmt) {
        if (!stmt) return;

        // Handle different statement types
        if (auto* expr_stmt = dynamic_cast<ExpressionStatement*>(stmt)) {
            analyze_expression(expr_stmt->expression.get());
        }
        else if (auto* return_stmt = dynamic_cast<ReturnStatement*>(stmt)) {
            if (return_stmt->value) {
                // Variable returned from function escapes to return
                if (auto* id_expr = dynamic_cast<IdentifierExpression*>(return_stmt->value.get())) {
                    auto it = variable_escapes.find(id_expr->name);
                    if (it != variable_escapes.end()) {
                        it->second.kind = EscapeKind::EscapeToReturn;
                        it->second.needs_lifetime_extension = true;
                    }
                }
            }
        }
        else if (auto* block_stmt = dynamic_cast<BlockStatement*>(stmt)) {
            for (auto& s : block_stmt->statements) {
                analyze_statement(s.get());
            }
        }
        else if (auto* decl_stmt = dynamic_cast<DeclarationStatement*>(stmt)) {
            if (auto* var_decl = dynamic_cast<VariableDeclaration*>(decl_stmt->declaration.get())) {
                // Register variable with default NoEscape
                EscapeInfo info;
                info.kind = EscapeKind::NoEscape;
                info.needs_lifetime_extension = false;
                variable_escapes[var_decl->name] = info;
            }
        }
        else if (auto* if_stmt = dynamic_cast<IfStatement*>(stmt)) {
            analyze_statement(if_stmt->then_branch.get());
            if (if_stmt->else_branch) {
                analyze_statement(if_stmt->else_branch.get());
            }
        }
        else if (auto* while_stmt = dynamic_cast<WhileStatement*>(stmt)) {
            analyze_statement(while_stmt->body.get());
        }
        else if (auto* for_stmt = dynamic_cast<ForStatement*>(stmt)) {
            analyze_statement(for_stmt->body.get());
        }
        else if (auto* for_range = dynamic_cast<ForRangeStatement*>(stmt)) {
            analyze_statement(for_range->body.get());
        }
    }

    // Analyze expression for variable usage
    void analyze_expression(Expression* expr) {
        if (!expr) return;

        if (auto* id_expr = dynamic_cast<IdentifierExpression*>(expr)) {
            // Variable referenced - no special escaping (unless it's in a return, handled above)
        }
        else if (auto* call_expr = dynamic_cast<CallExpression*>(expr)) {
            analyze_expression(call_expr->callee.get());
            for (auto& arg : call_expr->arguments) {
                analyze_expression(arg.get());
            }
        }
        else if (auto* binary_expr = dynamic_cast<BinaryExpression*>(expr)) {
            analyze_expression(binary_expr->left.get());
            analyze_expression(binary_expr->right.get());
        }
        else if (auto* unary_expr = dynamic_cast<UnaryExpression*>(expr)) {
            analyze_expression(unary_expr->operand.get());
        }
    }

    // Analyze a function declaration
    void analyze_function(FunctionDeclaration* func) {
        if (!func || !func->body) return;

        // Clear previous analysis
        variable_escapes.clear();

        // Analyze function body
        analyze_statement(func->body.get());

        // At this point, variable_escapes contains escape info for all variables
        // In a full implementation, we would attach this info back to the AST nodes
        // For now, the analysis is just computed but not yet attached
    }
};

} // namespace cpp2_transpiler

// Public API for escape analysis
void analyze_escape(cpp2_transpiler::AST& ast) {
    cpp2_transpiler::EscapeAnalyzer analyzer;

    // Analyze each declaration in the AST
    for (auto& decl : ast.declarations) {
        if (auto* func = dynamic_cast<cpp2_transpiler::FunctionDeclaration*>(decl.get())) {
            analyzer.analyze_function(func);
        }
    }
}

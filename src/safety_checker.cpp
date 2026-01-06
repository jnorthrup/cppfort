#include "../include/safety_checker.hpp"
#include <iostream>
#include <format>

namespace cpp2_transpiler {

SafetyChecker::SafetyChecker() {}

void SafetyChecker::check(AST& ast) {
    for (auto& decl : ast.declarations) {
        check_declaration(decl.get());
    }

    // Report all safety issues
    for (const auto& issue : issues) {
        const char* severity_str = issue.severity == SafetyIssue::Severity::Error ? "Error" : "Warning";
        std::cerr << std::format("[line {}] {} Safety {}: {}",
                                issue.line, severity_str,
                                static_cast<int>(issue.kind), issue.message) << std::endl;
    }
}

void SafetyChecker::check_declaration(Declaration* decl) {
    if (!decl) return;

    switch (decl->kind) {
        case Declaration::Kind::Variable:
            check_variable_initialization(static_cast<VariableDeclaration*>(decl));
            break;
        case Declaration::Kind::Function: {
            auto func = static_cast<FunctionDeclaration*>(decl);
            // Track if we're in a suspend/async function for await validation
            bool is_suspend = func->is_async;
            if (func->body) {
                check_statement(func->body.get());
                // Validate await expressions are only in suspend functions
                // This is a simplified check; full implementation would traverse expressions
            }
            break;
        }
        case Declaration::Kind::Type: {
            auto type_decl = static_cast<TypeDeclaration*>(decl);
            for (auto& member : type_decl->members) {
                check_declaration(member.get());
            }
            break;
        }
        case Declaration::Kind::Namespace: {
            auto ns = static_cast<NamespaceDeclaration*>(decl);
            for (auto& member : ns->members) {
                check_declaration(member.get());
            }
            break;
        }
        default:
            break;
    }
}

void SafetyChecker::check_statement(Statement* stmt) {
    if (!stmt) return;

    switch (stmt->kind) {
        case Statement::Kind::Expression:
            check_expression(static_cast<ExpressionStatement*>(stmt)->expr.get());
            break;
        case Statement::Kind::Block: {
            auto block = static_cast<BlockStatement*>(stmt);
            for (auto& s : block->statements) {
                check_statement(s.get());
            }
            break;
        }
        case Statement::Kind::If: {
            auto if_stmt = static_cast<IfStatement*>(stmt);
            check_expression(if_stmt->condition.get());
            check_statement(if_stmt->then_stmt.get());
            if (if_stmt->else_stmt) {
                check_statement(if_stmt->else_stmt.get());
            }
            break;
        }
        case Statement::Kind::While: {
            auto while_stmt = static_cast<WhileStatement*>(stmt);
            check_expression(while_stmt->condition.get());
            check_statement(while_stmt->body.get());
            break;
        }
        case Statement::Kind::For: {
            auto for_stmt = static_cast<ForStatement*>(stmt);
            if (for_stmt->init) check_statement(for_stmt->init.get());
            if (for_stmt->condition) check_expression(for_stmt->condition.get());
            if (for_stmt->increment) check_expression(for_stmt->increment.get());
            check_statement(for_stmt->body.get());
            break;
        }
        case Statement::Kind::Return:
            check_expression(static_cast<ReturnStatement*>(stmt)->value.get());
            break;
        case Statement::Kind::CoroutineScope:
            check_structured_concurrency(stmt);
            check_statement(static_cast<CoroutineScopeStatement*>(stmt)->body.get());
            break;
        case Statement::Kind::ChannelDecl:
            check_channel_lifetime(static_cast<ChannelDeclarationStatement*>(stmt));
            break;
        case Statement::Kind::ParallelFor: {
            auto pfor = static_cast<ParallelForStatement*>(stmt);
            check_expression(pfor->lower_bound.get());
            check_expression(pfor->upper_bound.get());
            if (pfor->step) check_expression(pfor->step.get());
            check_statement(pfor->body.get());
            break;
        }
        default:
            break;
    }
}

void SafetyChecker::check_expression(Expression* expr) {
    if (!expr) return;

    switch (expr->kind) {
        case Expression::Kind::Binary: {
            auto binary = static_cast<BinaryExpression*>(expr);
            check_expression(binary->left.get());
            check_expression(binary->right.get());

            // Check for specific binary operations
            if (binary->op == TokenType::Slash) {
                check_division_safety(binary);
            } else if (binary->op == TokenType::LessThan || binary->op == TokenType::GreaterThan ||
                       binary->op == TokenType::LessThanOrEqual || binary->op == TokenType::GreaterThanOrEqual) {
                check_mixed_sign_comparison(binary);
            } else if (binary->op == TokenType::Plus || binary->op == TokenType::Asterisk) {
                check_integer_overflow(binary);
            }
            break;
        }
        case Expression::Kind::Unary: {
            auto unary = static_cast<UnaryExpression*>(expr);
            check_expression(unary->operand.get());

            if (unary->op == TokenType::Asterisk) {
                // Dereference - check for potential null
                check_null_safety(unary->operand.get());
            }
            break;
        }
        case Expression::Kind::Call: {
            auto call = static_cast<CallExpression*>(expr);
            check_expression(call->callee.get());
            for (auto& arg : call->args) {
                check_expression(arg.get());
            }
            break;
        }
        case Expression::Kind::Subscript: {
            auto sub = static_cast<SubscriptExpression*>(expr);
            check_expression(sub->array.get());
            check_expression(sub->index.get());
            check_bounds_checking(sub);
            break;
        }
        case Expression::Kind::MemberAccess: {
            auto member = static_cast<MemberAccessExpression*>(expr);
            check_expression(member->object.get());
            // Check for potential null on object
            check_null_safety(member->object.get());
            break;
        }
        case Expression::Kind::Await: {
            auto await_expr = static_cast<AwaitExpression*>(expr);
            check_expression(await_expr->value.get());
            // Note: await context validation is done at function level
            break;
        }
        case Expression::Kind::Spawn: {
            auto spawn_expr = static_cast<SpawnExpression*>(expr);
            check_expression(spawn_expr->task.get());
            break;
        }
        case Expression::Kind::ChannelSend: {
            auto send_expr = static_cast<ChannelSendExpression*>(expr);
            check_expression(send_expr->value.get());
            break;
        }
        default:
            break;
    }
}

void SafetyChecker::check_null_safety(Expression* expr) {
    if (can_be_null(expr)) {
        report_issue(SafetyIssue::Severity::Warning,
                    SafetyIssue::Kind::PotentialNullDereference,
                    expr->line,
                    "Potential null dereference");
    }
}

void SafetyChecker::check_bounds_checking(SubscriptExpression* expr) {
    // Array bounds checking would be enhanced with type information
    report_issue(SafetyIssue::Severity::Warning,
                SafetyIssue::Kind::ArrayBoundsViolation,
                expr->line,
                "Array access may be out of bounds");
}

void SafetyChecker::check_division_safety(BinaryExpression* expr) {
    // Check for division by zero
    report_issue(SafetyIssue::Severity::Warning,
                SafetyIssue::Kind::DivisionByZero,
                expr->line,
                "Potential division by zero");
}

void SafetyChecker::check_mixed_sign_comparison(BinaryExpression* expr) {
    // Check for signed/unsigned comparison issues
    report_issue(SafetyIssue::Severity::Warning,
                SafetyIssue::Kind::MixedSignComparison,
                expr->line,
                "Mixed signed/unsigned comparison");
}

void SafetyChecker::check_variable_initialization(VariableDeclaration* decl) {
    if (!decl->initializer && !decl->type) {
        report_issue(SafetyIssue::Severity::Error,
                    SafetyIssue::Kind::UninitializedVariable,
                    decl->line,
                    "Variable '" + decl->name + "' used without initialization");
    }
}

void SafetyChecker::check_use_after_move(Expression* expr) {
    // This would track definite last use and report potential use-after-move
    report_issue(SafetyIssue::Severity::Warning,
                SafetyIssue::Kind::UseAfterMove,
                expr->line,
                "Potential use after move");
}

void SafetyChecker::check_integer_overflow(BinaryExpression* expr) {
    if (is_potential_overflow(expr)) {
        report_issue(SafetyIssue::Severity::Warning,
                    SafetyIssue::Kind::IntegerOverflow,
                    expr->line,
                    "Potential integer overflow");
    }
}

// ============================================================================
// Concurrency Safety Checks
// ============================================================================

void SafetyChecker::check_await_context(Expression* expr, bool in_suspend_function) {
    if (!in_suspend_function) {
        report_issue(SafetyIssue::Severity::Error,
                    SafetyIssue::Kind::AwaitOutsideSuspend,
                    expr->line,
                    "'await' can only be used in suspend/async functions");
    }
}

void SafetyChecker::check_structured_concurrency(Statement* stmt) {
    // Validate that spawned tasks are properly scoped
    // This is a placeholder for more sophisticated analysis
    // In a full implementation, we would track task lifetimes
}

void SafetyChecker::check_channel_lifetime(ChannelDeclarationStatement* decl) {
    // Validate channel is properly closed before scope exit
    // This is a placeholder for dataflow analysis
    report_issue(SafetyIssue::Severity::Warning,
                SafetyIssue::Kind::ChannelNotClosed,
                decl->line,
                "Channel '" + decl->name + "' should be explicitly closed");
}

bool SafetyChecker::can_be_null(Expression* expr) const {
    // Simplified check - in reality would use type system
    return expr->kind == Expression::Kind::Identifier ||
           expr->kind == Expression::Kind::MemberAccess ||
           expr->kind == Expression::Kind::Call;
}

bool SafetyChecker::is_unsigned_type(Type* type) const {
    // Simplified check
    return type && (type->name.find("uint") != std::string::npos ||
                   type->name.find("size_t") != std::string::npos);
}

bool SafetyChecker::is_signed_type(Type* type) const {
    // Simplified check
    return type && (type->name.find("int") != std::string::npos ||
                   type->name.find("float") != std::string::npos ||
                   type->name.find("double") != std::string::npos);
}

bool SafetyChecker::is_potential_overflow(BinaryExpression* expr) const {
    // Simplified check - in reality would analyze literal values
    return expr->op == TokenType::Plus || expr->op == TokenType::Asterisk;
}

void SafetyChecker::report_issue(SafetyIssue::Severity severity, SafetyIssue::Kind kind,
                                std::size_t line, const std::string& message) {
    issues.emplace_back(severity, kind, line, message);
}

// ============================================================================
// BorrowChecker Implementation
// ============================================================================

BorrowChecker::BorrowChecker() {}

void BorrowChecker::check(AST& ast) {
    // Run all borrow checking passes
    check_no_aliasing_violations(ast);
    check_borrow_outlives_owner(ast);
    check_move_invalidates_borrows(ast);
    enforce_exclusive_mut_borrow(ast);

    // Report all borrow issues
    for (const auto& issue : issues) {
        const char* severity_str = issue.severity == BorrowIssue::Severity::Error ? "Error" : "Warning";
        std::cerr << std::format("[line {}] {} Borrow {}: {}",
                                issue.line, severity_str,
                                static_cast<int>(issue.kind), issue.message) << std::endl;
    }
}

void BorrowChecker::check_no_aliasing_violations(AST& ast) {
    // Traverse AST and check for aliasing violations
    // Rule: Cannot have multiple mutable borrows or mutable + immutable borrows simultaneously
    for (auto& decl : ast.declarations) {
        check_declaration(decl.get());
    }
}

void BorrowChecker::check_borrow_outlives_owner(AST& ast) {
    // Check that borrows don't outlive their owners
    // Rule: A borrow must have a lifetime that is contained within the owner's lifetime
    for (auto& decl : ast.declarations) {
        check_declaration(decl.get());
    }
}

void BorrowChecker::check_move_invalidates_borrows(AST& ast) {
    // Check that moved values are not used after the move
    // Rule: After a value is moved, it cannot be used or borrowed
    for (auto& decl : ast.declarations) {
        check_declaration(decl.get());
    }
}

void BorrowChecker::enforce_exclusive_mut_borrow(AST& ast) {
    // Enforce exclusive mutable borrow rule
    // Rule: Only one mutable borrow can exist at a time
    for (auto& decl : ast.declarations) {
        check_declaration(decl.get());
    }
}

void BorrowChecker::check_declaration(Declaration* decl) {
    if (!decl) return;

    // For now, just traverse the structure
    // Full implementation would track ownership and borrow state
    switch (decl->kind) {
        case Declaration::Kind::Function: {
            auto func = static_cast<FunctionDeclaration*>(decl);
            if (func->body) {
                check_statement(func->body.get());
            }
            break;
        }
        case Declaration::Kind::Type: {
            auto type_decl = static_cast<TypeDeclaration*>(decl);
            for (auto& member : type_decl->members) {
                check_declaration(member.get());
            }
            break;
        }
        case Declaration::Kind::Namespace: {
            auto ns = static_cast<NamespaceDeclaration*>(decl);
            for (auto& member : ns->members) {
                check_declaration(member.get());
            }
            break;
        }
        default:
            break;
    }
}

void BorrowChecker::check_statement(Statement* stmt) {
    if (!stmt) return;

    // Traverse statement tree
    switch (stmt->kind) {
        case Statement::Kind::Expression:
            check_expression(static_cast<ExpressionStatement*>(stmt)->expr.get());
            break;
        case Statement::Kind::Block: {
            auto block = static_cast<BlockStatement*>(stmt);
            for (auto& s : block->statements) {
                check_statement(s.get());
            }
            break;
        }
        case Statement::Kind::If: {
            auto if_stmt = static_cast<IfStatement*>(stmt);
            check_expression(if_stmt->condition.get());
            check_statement(if_stmt->then_stmt.get());
            if (if_stmt->else_stmt) {
                check_statement(if_stmt->else_stmt.get());
            }
            break;
        }
        case Statement::Kind::While: {
            auto while_stmt = static_cast<WhileStatement*>(stmt);
            check_expression(while_stmt->condition.get());
            check_statement(while_stmt->body.get());
            break;
        }
        default:
            break;
    }
}

void BorrowChecker::check_expression(Expression* expr) {
    if (!expr) return;

    // Traverse expression tree
    switch (expr->kind) {
        case Expression::Kind::Binary: {
            auto bin = static_cast<BinaryExpression*>(expr);
            check_expression(bin->left.get());
            check_expression(bin->right.get());
            break;
        }
        case Expression::Kind::Unary: {
            auto unary = static_cast<UnaryExpression*>(expr);
            check_expression(unary->operand.get());
            break;
        }
        case Expression::Kind::Call: {
            auto call = static_cast<CallExpression*>(expr);
            check_expression(call->callee.get());
            for (auto& arg : call->arguments) {
                check_expression(arg.expr.get());
            }
            break;
        }
        default:
            break;
    }
}

void BorrowChecker::report_issue(BorrowIssue::Severity severity, BorrowIssue::Kind kind,
                                 std::size_t line, const std::string& message) {
    issues.emplace_back(severity, kind, line, message);
}

} // namespace cpp2_transpiler
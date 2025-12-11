#pragma once

#include "ast.hpp"
#include <vector>
#include <memory>

namespace cpp2_transpiler {

class SafetyChecker {
public:
    SafetyChecker();
    void check(AST& ast);

private:
    struct SafetyIssue {
        enum class Severity {
            Warning,
            Error
        };

        enum class Kind {
            PotentialNullDereference,
            ArrayBoundsViolation,
            DivisionByZero,
            MixedSignComparison,
            UninitializedVariable,
            UnsafePointerCast,
            IntegerOverflow,
            UseAfterMove
        };

        Severity severity;
        Kind kind;
        std::size_t line;
        std::string message;

        SafetyIssue(Severity s, Kind k, std::size_t l, std::string m)
            : severity(s), kind(k), line(l), message(std::move(m)) {}
    };

    std::vector<SafetyIssue> issues;

    // Check methods
    void check_declaration(Declaration* decl);
    void check_statement(Statement* stmt);
    void check_expression(Expression* expr);

    // Specific safety checks
    void check_null_safety(Expression* expr);
    void check_bounds_checking(SubscriptExpression* expr);
    void check_division_safety(BinaryExpression* expr);
    void check_mixed_sign_comparison(BinaryExpression* expr);
    void check_variable_initialization(VariableDeclaration* decl);
    void check_use_after_move(Expression* expr);
    void check_integer_overflow(BinaryExpression* expr);

    // Helper methods
    bool can_be_null(Expression* expr) const;
    bool is_unsigned_type(Type* type) const;
    bool is_signed_type(Type* type) const;
    bool is_potential_overflow(BinaryExpression* expr) const;

    // Report methods
    void report_issue(SafetyIssue::Severity severity, SafetyIssue::Kind kind,
                     std::size_t line, const std::string& message);
};

} // namespace cpp2_transpiler
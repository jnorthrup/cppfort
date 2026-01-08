// include/codegen/expression_emitter.hpp - Expression generation for C++ code
// Part of Phase 2: Code Generator Extraction
#pragma once

#include "codegen/emitter_context.hpp"
#include "codegen/type_emitter.hpp"
#include "ast.hpp"
#include <string>

namespace cpp2_transpiler {

// Forward declarations
class StatementEmitter;

/// Generates C++ expression code from Cpp2 AST expressions
class ExpressionEmitter : public EmitterBase {
public:
    ExpressionEmitter(EmitterContext& ctx, TypeEmitter& type_emitter)
        : EmitterBase(ctx), type_emitter_(type_emitter) {}

    /// Generate expression to output stream
    void generate_expression(Expression* expr);
    
    /// Generate expression to string (for embedding in larger constructs)
    std::string generate_expression_to_string(Expression* expr);

    /// Generate statement to string (for lambda bodies)
    std::string generate_statement_to_string(Statement* stmt);

private:
    void generate_literal_expression(LiteralExpression* expr);
    void generate_identifier_expression(IdentifierExpression* expr);
    void generate_binary_expression(BinaryExpression* expr);
    void generate_unary_expression(UnaryExpression* expr);
    void generate_call_expression(CallExpression* expr);
    void generate_member_access_expression(MemberAccessExpression* expr);
    void generate_subscript_expression(SubscriptExpression* expr);
    void generate_ternary_expression(TernaryExpression* expr);
    void generate_lambda_expression(LambdaExpression* expr);
    void generate_is_expression(IsExpression* expr);
    void generate_as_expression(AsExpression* expr);
    void generate_string_interpolation(StringInterpolationExpression* expr);
    void generate_range_expression(RangeExpression* expr);
    void generate_list_expression(ListExpression* expr);
    void generate_struct_initializer_expression(StructInitializerExpression* expr);
    void generate_metafunction_call_expression(MetafunctionCallExpression* expr);

    // UFCS handling
    std::string resolve_ufcs_call(CallExpression* call);

    TypeEmitter& type_emitter_;
};

} // namespace cpp2_transpiler

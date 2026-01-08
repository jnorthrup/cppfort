// include/codegen/statement_emitter.hpp - Statement generation for C++ code
// Part of Phase 2: Code Generator Extraction
#pragma once

#include "codegen/emitter_context.hpp"
#include "codegen/expression_emitter.hpp"
#include "ast.hpp"

namespace cpp2_transpiler {

/// Generates C++ statement code from Cpp2 AST statements
class StatementEmitter : public EmitterBase {
public:
    StatementEmitter(EmitterContext& ctx, ExpressionEmitter& expr_emitter, TypeEmitter& type_emitter)
        : EmitterBase(ctx), expr_emitter_(expr_emitter), type_emitter_(type_emitter) {}

    /// Generate statement to output stream
    void generate_statement(Statement* stmt);

private:
    void generate_block_statement(BlockStatement* stmt);
    void generate_scope_block_statement(ScopeBlockStatement* stmt);
    void generate_expression_statement(ExpressionStatement* stmt);
    void generate_if_statement(IfStatement* stmt);
    void generate_while_statement(WhileStatement* stmt);
    void generate_do_while_statement(DoWhileStatement* stmt);
    void generate_for_statement(ForStatement* stmt);
    void generate_for_range_statement(ForRangeStatement* stmt);
    void generate_switch_statement(SwitchStatement* stmt);
    void generate_inspect_statement(InspectStatement* stmt);
    void generate_return_statement(ReturnStatement* stmt);
    void generate_break_statement(BreakStatement* stmt);
    void generate_continue_statement(ContinueStatement* stmt);
    void generate_try_statement(TryStatement* stmt);
    void generate_throw_statement(ThrowStatement* stmt);
    void generate_contract_statement(ContractStatement* stmt);

    // Concurrency statements (Kotlin-style structured concurrency)
    void generate_coroutine_scope_statement(CoroutineScopeStatement* stmt);
    void generate_parallel_for_statement(ParallelForStatement* stmt);
    void generate_channel_declaration(ChannelDeclarationStatement* stmt);

    ExpressionEmitter& expr_emitter_;
    TypeEmitter& type_emitter_;
};

} // namespace cpp2_transpiler

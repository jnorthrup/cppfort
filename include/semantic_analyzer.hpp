#pragma once

#include "ast.hpp"
#include "utils.hpp"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <vector>
#include <memory>
#include <optional>

namespace cpp2_transpiler {

struct Symbol {
    enum class Kind {
        Variable,
        Function,
        Type,
        Namespace,
        Parameter,
        Member,
        EnumMember
    };

    Kind kind;
    std::string name;
    Type* type;
    Declaration* declaration;
    bool is_const = false;
    bool is_mut = false;

    Symbol(Kind k, std::string n, Type* t, Declaration* d)
        : kind(k), name(std::move(n)), type(t), declaration(d) {}
};

class Scope {
public:
    Scope(std::shared_ptr<Scope> parent = nullptr) : parent_scope(parent) {}

    void add_symbol(const std::string& name, std::unique_ptr<Symbol> symbol) {
        symbols[name] = std::move(symbol);
    }

    Symbol* lookup(const std::string& name) const {
        auto it = symbols.find(name);
        if (it != symbols.end()) {
            return it->second.get();
        }
        return parent_scope ? parent_scope->lookup(name) : nullptr;
    }

    Symbol* lookup_local(const std::string& name) const {
        auto it = symbols.find(name);
        return it != symbols.end() ? it->second.get() : nullptr;
    }

    std::shared_ptr<Scope> parent() const { return parent_scope; }

    void for_each_local_symbol(const std::function<void(Symbol*)>& callback) {
        for (const auto& [name, symbol] : symbols) {
            callback(symbol.get());
        }
    }

private:
    std::unordered_map<std::string, std::unique_ptr<Symbol>> symbols;
    std::shared_ptr<Scope> parent_scope;
};

// ============================================================================
// Parameter Qualifier to Ownership Mapping
// ============================================================================

// Maps Cpp2 parameter qualifiers to ownership semantics for borrow checking
inline OwnershipKind map_qualifier_to_ownership(ParameterQualifier qualifier) {
    switch (qualifier) {
        case ParameterQualifier::In:
            return OwnershipKind::Borrowed;      // Immutable borrow (const T&)
        case ParameterQualifier::Out:
            return OwnershipKind::MutBorrowed;   // Mutable borrow, write-before-return (T&)
        case ParameterQualifier::InOut:
            return OwnershipKind::MutBorrowed;   // Mutable borrow with read/write (T&)
        case ParameterQualifier::Move:
            return OwnershipKind::Moved;         // Ownership transfer (T&&)
        case ParameterQualifier::Forward:
            // Forward is conditional: Moved for rvalues, Borrowed for lvalues
            // At the AST level, we default to Moved; actual semantics resolved during codegen
            return OwnershipKind::Moved;
        default:
            return OwnershipKind::Owned;         // Default: owned value
    }
}

inline ParameterQualifier canonicalize_parameter_qualifier_for_semantics(
    const std::vector<ParameterQualifier>& qualifiers) {
    if (qualifiers.empty()) {
        return ParameterQualifier::None;
    }
    // TODO: If unqualified params become implicit 'in', change effective qualifier here.
    return qualifiers.front();
}

inline bool qualifier_is_explicit(const std::vector<ParameterQualifier>& qualifiers) {
    return !qualifiers.empty();
}

class SemanticAnalyzer {
public:
    SemanticAnalyzer();
    void analyze(AST& ast);

private:
    std::shared_ptr<Scope> current_scope;
    AST* current_ast = nullptr;

    // Type checking
    std::unique_ptr<Type> check_type(std::unique_ptr<Type> type);
    void check_type_ptr(const Type* type);  // Check type without ownership transfer
    bool is_type_compatible(const Type* lhs, const Type* rhs) const;
    std::unique_ptr<Type> deduce_type(Expression* expr);
    void check_expression(Expression* expr);
    void check_statement(Statement* stmt);
    void check_declaration(Declaration* decl);

    // Symbol table management
    void push_scope();
    void pop_scope();
    void add_symbol(const std::string& name, std::unique_ptr<Symbol> symbol);
    Symbol* lookup_symbol(const std::string& name) const;

    // Type resolution
    Type* resolve_type(const Type* type);
    void resolve_type_declaration(TypeDeclaration* type_decl);
    void resolve_enum_declaration(TypeDeclaration* enum_decl);

    // Function checking
    void check_function(FunctionDeclaration* func);
    void check_function_body(FunctionDeclaration* func);
    void check_parameter_types(FunctionDeclaration* func);
    void check_return_type(FunctionDeclaration* func, Type* actual_type);

    // Expression type checking
    void check_literal_expression(LiteralExpression* expr);
    void check_identifier_expression(IdentifierExpression* expr);
    void check_binary_expression(BinaryExpression* expr);
    void check_unary_expression(UnaryExpression* expr);
    void check_call_expression(CallExpression* expr);
    void check_member_access_expression(MemberAccessExpression* expr);
    void check_subscript_expression(SubscriptExpression* expr);
    void check_ternary_expression(TernaryExpression* expr);
    void check_lambda_expression(LambdaExpression* expr);
    void check_is_expression(IsExpression* expr);
    void check_as_expression(AsExpression* expr);
    void check_range_expression(RangeExpression* expr);

    // Safety checks
    void check_variable_usage(VariableDeclaration* var);
    void check_unsafe_operations(Expression* expr);
    void check_null_safety(Expression* expr);
    void check_bounds_checking(SubscriptExpression* expr);
    void check_mixed_sign_arithmetic(BinaryExpression* expr);

    // Concurrency expression checking (Kotlin-style structured concurrency)
    void check_await_expression(AwaitExpression* expr);
    void check_spawn_expression(SpawnExpression* expr);
    void check_channel_send_expression(ChannelSendExpression* expr);
    void check_channel_recv_expression(ChannelRecvExpression* expr);
    void check_channel_select_expression(ChannelSelectExpression* expr);

    // Contract checking
    void check_contract(ContractExpression* contract);
    void check_function_contracts(FunctionDeclaration* func);

    // Phase 7: Arena allocation
    void analyze_scope_for_arena(Scope* scope, BlockStatement* block);
    std::size_t next_arena_id = 1;

    // Phase 3/4 skeleton pass: explicit non-invasive traversal hooks
    void analyze_escape_and_borrow(AST& ast);
    void analyze_escape_and_borrow_declaration(Declaration* decl);
    void analyze_escape_and_borrow_statement(Statement* stmt);
    void analyze_escape_and_borrow_expression(Expression* expr);

    // Template handling
    void check_template_parameters(TemplateStatement* tmpl);
    void instantiate_template(Declaration* decl, const std::vector<std::string>& args);

    // UFCS resolution
    void resolve_ufcs(CallExpression* call);
    bool is_member_call(Expression* callee) const;

    // Helpers
    bool is_builtin_type(const std::string& name) const;
    std::string get_mangled_name(FunctionDeclaration* func) const;
    void report_error(std::size_t line, const std::string& message);
    void report_warning(std::size_t line, const std::string& message);

    // State tracking
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::unordered_set<std::string, SimpleStringHash> undeclared_variables;
    std::unordered_map<Declaration*, bool> checked_declarations;
    bool has_cpp1_passthrough = false;  // Track if AST contains C++1 passthrough declarations

    // Concurrency state tracking
    bool in_suspend_function = false;         // True when inside a suspend function
    int in_coroutine_scope_depth = 0;         // Depth of nested coroutineScope blocks

    // Built-in types
    void register_builtin_types();
    std::unordered_map<std::string, std::unique_ptr<Type>, SimpleStringHash> builtin_types;

    // Definite last use tracking
    struct VariableUsage {
        bool used = false;
        bool definitely_last_use = false;
        std::size_t last_use_line = 0;
    };
    std::unordered_map<std::string, VariableUsage, SimpleStringHash> variable_usage;
    void track_variable_usage(const std::string& name, std::size_t line);
    void check_unused_variables();
};

} // namespace cpp2_transpiler

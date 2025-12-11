#pragma once

#include "ast.hpp"
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <unordered_set>

namespace cpp2_transpiler {

class CodeGenerator {
public:
    CodeGenerator();
    std::string generate(AST& ast);

private:
    std::ostringstream output;
    int indent_level = 0;
    bool needs_semicolon = true;

    // State tracking
    std::unordered_set<std::string> generated_functions;
    std::unordered_set<std::string> generated_types;
    std::vector<std::string> includes;

    // Helper methods
    void write_line(const std::string& line);
    void write(const std::string& text);
    void indent();
    void dedent();
    std::string get_indent() const;

    // Generation methods
    void generate_declaration(Declaration* decl);
    void generate_variable_declaration(VariableDeclaration* decl);
    void generate_function_declaration(FunctionDeclaration* decl);
    void generate_type_declaration(TypeDeclaration* decl);
    void generate_namespace_declaration(NamespaceDeclaration* decl);
    void generate_operator_declaration(OperatorDeclaration* decl);
    void generate_using_declaration(UsingDeclaration* decl);
    void generate_import_declaration(ImportDeclaration* decl);

    void generate_statement(Statement* stmt);
    void generate_block_statement(BlockStatement* stmt);
    void generate_expression_statement(ExpressionStatement* stmt);
    void generate_if_statement(IfStatement* stmt);
    void generate_while_statement(WhileStatement* stmt);
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

    void generate_expression(Expression* expr);
    std::string generate_expression_to_string(Expression* expr);
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

    // Type generation
    std::string generate_type(Type* type);
    std::string generate_builtin_type(Type* type);
    std::string generate_user_type(Type* type);
    std::string generate_pointer_type(Type* type);
    std::string generate_reference_type(Type* type);
    std::string generate_array_type(Type* type);
    std::string generate_function_type(Type* type);
    std::string generate_template_type(Type* type);

    // Safety check generation
    void generate_bounds_check(SubscriptExpression* expr);
    void generate_null_check(Expression* expr);
    void generate_division_check(BinaryExpression* expr);
    void generate_mixed_sign_check(BinaryExpression* expr);

    // Contract generation
    void generate_contract_pre(const std::string& condition, const std::optional<std::string>& message);
    void generate_contract_post(const std::string& condition, const std::optional<std::string>& message);
    void generate_contract_assert(const std::string& condition, const std::optional<std::string>& message);
    std::string generate_contract_group_name(const std::string& function_name);

    // Pattern matching generation
    void generate_pattern(const InspectStatement::Pattern& pattern);
    std::string generate_pattern_match(const InspectStatement::Pattern& pattern, const std::string& value_var);

    // UFCS handling
    std::string resolve_ufcs_call(CallExpression* call);

    // Definite last use handling
    void insert_move_semantics(Expression* expr);
    bool is_definite_last_use(const std::string& variable_name);

    // Metafunction expansion
    void expand_metafunction(TypeDeclaration* type_decl);
    void expand_value_metafunction(TypeDeclaration* type_decl);
    void expand_ordered_metafunction(TypeDeclaration* type_decl);
    void expand_copyable_metafunction(TypeDeclaration* type_decl);
    void expand_interface_metafunction(TypeDeclaration* type_decl);
    void expand_polymorphic_base_metafunction(TypeDeclaration* type_decl);
    void expand_enum_metafunction(TypeDeclaration* type_decl);
    void expand_flag_enum_metafunction(TypeDeclaration* type_decl);
    void expand_union_metafunction(TypeDeclaration* type_decl);
    void expand_struct_metafunction(TypeDeclaration* type_decl);
    void expand_hashable_metafunction(TypeDeclaration* type_decl);

    // String formatting
    std::string format_string_literal(const std::string& str);
    std::string escape_string(const std::string& str);

    // Name mangling
    std::string mangle_function_name(FunctionDeclaration* func);
    std::string mangle_template_name(const std::string& base_name, const std::vector<std::string>& params);

    // Include management
    void add_include(const std::string& header);
    void write_includes();

    // Utility
    std::string generate_unique_name(const std::string& base);
    std::string get_cpp_keyword(const std::string& cpp2_keyword);
    bool needs_nodiscard(FunctionDeclaration* func);
};

} // namespace cpp2_transpiler
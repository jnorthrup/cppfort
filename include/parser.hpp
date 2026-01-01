#pragma once

#include "ast.hpp"
#include "lexer.hpp"
#include <vector>
#include <memory>
#include <span>
#include <optional>
#include <limits>

namespace cpp2_transpiler {

class Parser {
public:
    explicit Parser(std::span<Token> tokens);

    std::unique_ptr<AST> parse();

private:
    std::span<Token> tokens;
    std::size_t current = 0;

    // Pending markdown blocks to attach to next declaration
    std::vector<MarkdownBlockAttr> pending_markdown_blocks;

    // Track pending '>' from '>>' split for nested templates
    bool pending_gt = false;

    // Parsing utilities
    const Token& peek() const;
    const Token& advance();
    const Token& previous() const;
    bool is_at_end() const;
    bool check(TokenType type) const;
    bool is_identifier_like() const;  // Check if current token can be used as identifier
    std::string_view get_identifier_lexeme() const;  // Get lexeme for identifier-like tokens
    bool match(TokenType type);
    bool match(std::initializer_list<TokenType> types);
    const Token& consume(TokenType type, const char* message);
    bool consume_if(TokenType type);
    bool consume_template_close();  // Consume '>' or first half of '>>' for nested templates

    template<typename F>
    auto synchronize_on_error(F&& func) -> decltype(func());

    // Entry point
    std::unique_ptr<Declaration> declaration();
    std::unique_ptr<Statement> statement();
    std::unique_ptr<Expression> expression();

    // Declarations
    std::unique_ptr<Declaration> variable_declaration();
    std::unique_ptr<Declaration> function_declaration();
    std::unique_ptr<Declaration> type_declaration(std::vector<std::string> decorators = {});
    std::unique_ptr<Declaration> namespace_declaration();
    std::unique_ptr<Declaration> operator_declaration();
    std::unique_ptr<Declaration> using_declaration();
    std::unique_ptr<Declaration> import_declaration();
    std::unique_ptr<Declaration> template_declaration();
    std::unique_ptr<Declaration> cpp1_passthrough_declaration(bool is_struct_type = false);  // C++1 passthrough

    // C++1 detection
    bool check_cpp1_function_syntax();
    bool check_cpp1_struct_syntax();
    bool check_cpp1_constexpr_syntax();

    // Types
    std::unique_ptr<Type> type();
    std::unique_ptr<Type> basic_type();
    std::unique_ptr<Type> qualified_type();
    std::unique_ptr<Type> template_type();
    std::unique_ptr<Type> function_type();

    // Statements
    std::unique_ptr<Statement> block_statement();
    std::unique_ptr<Statement> if_statement();
    std::unique_ptr<Statement> while_statement();
    std::unique_ptr<Statement> do_while_statement();
    std::unique_ptr<Statement> for_statement();
    std::unique_ptr<Statement> for_range_statement();
    std::unique_ptr<Statement> labeled_loop_statement(std::string label);
    std::unique_ptr<Statement> try_loop_with_initializer();  // Cpp2 (copy i:=0) while/for/do
    std::unique_ptr<Statement> switch_statement();
    std::unique_ptr<Statement> inspect_statement();
    std::unique_ptr<Expression> inspect_expression();
    std::unique_ptr<Statement> return_statement();
    std::unique_ptr<Statement> break_statement();
    std::unique_ptr<Statement> continue_statement();
    std::unique_ptr<Statement> try_statement();
    std::unique_ptr<Statement> throw_statement();
    std::unique_ptr<Statement> contract_statement();
    std::unique_ptr<Statement> static_assert_statement();

    // Concurrency statements (Kotlin-style)
    std::unique_ptr<Statement> coroutine_scope_statement();
    std::unique_ptr<Statement> channel_declaration_statement();
    std::unique_ptr<Statement> parallel_for_statement();

    // Expressions (precedence climbing)
    std::unique_ptr<Expression> assignment_expression();
    std::unique_ptr<Expression> ternary_expression();
    std::unique_ptr<Expression> logical_or_expression();
    std::unique_ptr<Expression> logical_and_expression();
    std::unique_ptr<Expression> equality_expression();
    std::unique_ptr<Expression> bitwise_or_expression();
    std::unique_ptr<Expression> bitwise_xor_expression();
    std::unique_ptr<Expression> bitwise_and_expression();
    std::unique_ptr<Expression> comparison_expression();
    std::unique_ptr<Expression> range_expression();
    std::unique_ptr<Expression> shift_expression();
    std::unique_ptr<Expression> addition_expression();
    std::unique_ptr<Expression> multiplication_expression();
    std::unique_ptr<Expression> prefix_expression();
    std::unique_ptr<Expression> postfix_expression();
    std::unique_ptr<Expression> primary_expression();

    // Expression helpers
    std::unique_ptr<Expression> call_expression(std::unique_ptr<Expression> callee);
    std::unique_ptr<Expression> member_access_expression(std::unique_ptr<Expression> object);
    std::unique_ptr<Expression> subscript_expression(std::unique_ptr<Expression> array);
    std::unique_ptr<Expression> string_interpolation();
    std::unique_ptr<Expression> list_literal();
    std::unique_ptr<Expression> struct_initializer();
    std::unique_ptr<Expression> lambda_expression();
    std::unique_ptr<Expression> cpp1_lambda_expression();  // C++1 [capture](params) { body }
    std::unique_ptr<Expression> function_expression();  // Cpp2 :(params) -> type = { body }
    std::unique_ptr<Expression> metafunction_call();
    std::unique_ptr<Expression> is_expression();
    std::unique_ptr<Expression> as_expression();

    // Concurrency expressions (Kotlin-style)
    std::unique_ptr<Expression> await_expression();
    std::unique_ptr<Expression> spawn_expression();
    std::unique_ptr<Expression> channel_send_expression();
    std::unique_ptr<Expression> channel_recv_expression();
    std::unique_ptr<Expression> select_expression();

    // Pattern matching for inspect
    std::pair<InspectStatement::Pattern, std::unique_ptr<Statement>> inspect_arm();
    InspectStatement::Pattern pattern();

    // Contract handling
    std::vector<std::unique_ptr<ContractExpression>> parse_contracts();

    // Markdown block handling
    void collect_markdown_blocks();
    void attach_markdown_blocks(Declaration* decl);

    // Parameter qualifier parsing (Cpp2-specific)
    std::vector<ParameterQualifier> parse_parameter_qualifiers();

    // Template handling
    std::vector<std::string> template_parameters();
    bool is_template_start();
    bool is_type_qualifier();  // Check for type qualifiers (final, virtual, etc.)
    bool is_cpp1_template_start();  // Detect C++1 template syntax for passthrough

    // Error handling
    void error(const Token& token, const char* message);
    void error_at(const Token& token, const char* message);
    void error_at_current(const char* message);

    bool panic_mode = false;
    bool suppress_errors = false;  // For speculative parsing
    std::size_t last_error_position = std::numeric_limits<std::size_t>::max();
    std::string last_error_text;
    
public:
    // Error count for checking if parsing succeeded
    std::size_t error_count = 0;
    bool had_errors() const { return error_count > 0; }
};

} // namespace cpp2_transpiler
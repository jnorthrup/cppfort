#pragma once

#include "ast.hpp"
#include "lexer.hpp"
#include <vector>
#include <memory>
#include <span>
#include <optional>

namespace cpp2_transpiler {

class Parser {
public:
    explicit Parser(std::span<Token> tokens);

    std::unique_ptr<AST> parse();

private:
    std::span<Token> tokens;
    std::size_t current = 0;

    // Parsing utilities
    const Token& peek() const;
    const Token& advance();
    const Token& previous() const;
    bool is_at_end() const;
    bool check(TokenType type) const;
    bool match(TokenType type);
    bool match(std::initializer_list<TokenType> types);
    const Token& consume(TokenType type, const char* message);
    bool consume_if(TokenType type);

    template<typename F>
    auto synchronize_on_error(F&& func) -> decltype(func());

    // Entry point
    std::unique_ptr<Declaration> declaration();
    std::unique_ptr<Statement> statement();
    std::unique_ptr<Expression> expression();

    // Declarations
    std::unique_ptr<Declaration> variable_declaration();
    std::unique_ptr<Declaration> function_declaration();
    std::unique_ptr<Declaration> type_declaration();
    std::unique_ptr<Declaration> namespace_declaration();
    std::unique_ptr<Declaration> operator_declaration();
    std::unique_ptr<Declaration> using_declaration();
    std::unique_ptr<Declaration> import_declaration();
    std::unique_ptr<Declaration> template_declaration();

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
    std::unique_ptr<Statement> for_statement();
    std::unique_ptr<Statement> for_range_statement();
    std::unique_ptr<Statement> switch_statement();
    std::unique_ptr<Statement> inspect_statement();
    std::unique_ptr<Statement> return_statement();
    std::unique_ptr<Statement> break_statement();
    std::unique_ptr<Statement> continue_statement();
    std::unique_ptr<Statement> try_statement();
    std::unique_ptr<Statement> throw_statement();
    std::unique_ptr<Statement> contract_statement();
    std::unique_ptr<Statement> static_assert_statement();

    // Expressions (precedence climbing)
    std::unique_ptr<Expression> assignment_expression();
    std::unique_ptr<Expression> ternary_expression();
    std::unique_ptr<Expression> logical_or_expression();
    std::unique_ptr<Expression> logical_and_expression();
    std::unique_ptr<Expression> equality_expression();
    std::unique_ptr<Expression> comparison_expression();
    std::unique_ptr<Expression> range_expression();
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
    std::unique_ptr<Expression> metafunction_call();
    std::unique_ptr<Expression> is_expression();
    std::unique_ptr<Expression> as_expression();

    // Pattern matching for inspect
    std::pair<InspectStatement::Pattern, std::unique_ptr<Statement>> inspect_arm();
    InspectStatement::Pattern pattern();

    // Contract handling
    std::vector<std::unique_ptr<ContractExpression>> parse_contracts();

    // Template handling
    std::vector<std::string> template_parameters();
    bool is_template_start();

    // Error handling
    void error(const Token& token, const char* message);
    void error_at(const Token& token, const char* message);
    void error_at_current(const char* message);

    bool panic_mode = false;
};

} // namespace cpp2_transpiler
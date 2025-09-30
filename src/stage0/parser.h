#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ast.h"
#include "token.h"

namespace cppfort::stage0 {

struct ParseError : std::runtime_error {
    explicit ParseError(const std::string& message)
        : std::runtime_error(message) {}
};

class Parser {
  public:
    Parser(std::vector<Token> tokens, std::string source);

    [[nodiscard]] TranslationUnit parse();

  private:
    [[nodiscard]] bool is_at_end() const;
    [[nodiscard]] const Token& peek() const;
    [[nodiscard]] const Token& previous() const;
    [[nodiscard]] const Token& peek_next() const;
    [[nodiscard]] const Token& advance();
    [[nodiscard]] bool check(TokenType type) const;
    [[nodiscard]] bool match(TokenType type);
    [[nodiscard]] const Token& consume(TokenType type, const std::string& message);

    [[nodiscard]] TranslationUnit parse_translation_unit();
    [[nodiscard]] std::variant<FunctionDecl, TypeDecl> parse_declaration();
    [[nodiscard]] FunctionDecl parse_function();
    [[nodiscard]] FunctionDecl parse_function_after_name(const Token& name);
    [[nodiscard]] std::vector<Parameter> parse_parameter_list();
    [[nodiscard]] Parameter parse_loop_parameter();
    [[nodiscard]] IncludeDecl parse_include_directive(const Token& directive);
    [[nodiscard]] RawDecl parse_raw_declaration();
    [[nodiscard]] bool is_cpp_raw_declaration() const;
    [[nodiscard]] Block parse_block();
    [[nodiscard]] Statement parse_statement();
    [[nodiscard]] ForChainStmt parse_for_chain_statement(const Token& keyword);
    [[nodiscard]] VariableDecl parse_variable_decl(const Token& name);
    [[nodiscard]] ReturnStmt parse_return_statement(const Token& keyword);
    [[nodiscard]] AssertStmt parse_assert_statement(const Token& keyword);
    [[nodiscard]] ExpressionStmt parse_expression_statement(const Token& start_token);

    [[nodiscard]] std::string collect_text_until(const std::vector<TokenType>& end_types, SourceLocation* span_location = nullptr);
    [[nodiscard]] std::string collect_text_until_semicolon();
  [[nodiscard]] std::string collect_raw_statement_text();
    [[nodiscard]] std::string slice(const Token& first, const Token& last) const;

    static std::string trim_copy(std::string_view text);

  private:
    std::vector<Token> m_tokens;
    std::string m_source;
    std::size_t m_current {0};
};

} // namespace cppfort::stage0

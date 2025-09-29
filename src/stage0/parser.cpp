#include "parser.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace cppfort::stage0 {

namespace {
[[nodiscard]] bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
}

Parser::Parser(std::vector<Token> tokens, std::string source)
    : m_tokens(std::move(tokens)), m_source(std::move(source)) {}

TranslationUnit Parser::parse() {
    return parse_translation_unit();
}

TranslationUnit Parser::parse_translation_unit() {
    TranslationUnit unit;
    while (!is_at_end()) {
        auto decl = parse_declaration();
        if (std::holds_alternative<FunctionDecl>(decl)) {
            unit.functions.push_back(std::move(std::get<FunctionDecl>(decl)));
        } else {
            unit.types.push_back(std::move(std::get<TypeDecl>(decl)));
        }
    }
    return unit;
}

std::variant<FunctionDecl, TypeDecl> Parser::parse_declaration() {
    const Token& name = consume(TokenType::Identifier, "Expected identifier at start of declaration");
    static_cast<void>(consume(TokenType::Colon, "Expected ':' after identifier"));

    // Check if it's a type declaration
    if (check(TokenType::KeywordType)) {
        advance(); // consume 'type'
        static_cast<void>(consume(TokenType::Equals, "Expected '=' after 'type'"));
        static_cast<void>(consume(TokenType::LBrace, "Expected '{' to start type body"));
        auto body = collect_text_until({TokenType::RBrace});
        static_cast<void>(consume(TokenType::RBrace, "Expected '}' to close type body"));
        return TypeDecl {name.lexeme, trim_copy(body), name.location};
    } else {
        // Function declaration
        return parse_function_after_name(name);
    }
}

FunctionDecl Parser::parse_function() {
    const Token& name = consume(TokenType::Identifier, "Expected identifier at start of declaration");
    static_cast<void>(consume(TokenType::Colon, "Expected ':' after identifier"));
    return parse_function_after_name(name);
}

FunctionDecl Parser::parse_function_after_name(const Token& name) {
    // 'name' and the ':' have already been consumed by the caller.
    auto parameters = parse_parameter_list();

    std::optional<std::string> return_type;
    if (match(TokenType::Arrow)) {
        SourceLocation span_location;
        auto text = collect_text_until({TokenType::Equals}, &span_location);
        return_type = trim_copy(text);
        if (return_type->empty()) {
            throw ParseError("Return type body may not be empty");
        }
    }

    static_cast<void>(consume(TokenType::Equals, "Expected '=' before function body"));

    FunctionBody body;
    if (match(TokenType::LBrace)) {
        body = parse_block();
    } else {
        auto expression_text = collect_text_until_semicolon();
        SourceLocation location = previous().location;
        body = ExpressionBody {trim_copy(expression_text), location};
    }

    FunctionDecl decl;
    decl.name = name.lexeme;
    decl.parameters = std::move(parameters);
    decl.return_type = std::move(return_type);
    decl.body = std::move(body);
    decl.location = name.location;
    return decl;
}

std::vector<Parameter> Parser::parse_parameter_list() {
    std::vector<Parameter> parameters;
    static_cast<void>(consume(TokenType::LParen, "Expected '(' to start parameter list"));

    if (check(TokenType::RParen)) {
        static_cast<void>(advance());
        return parameters;
    }

    while (true) {
        // Check for optional parameter kind
        std::string kind;
        if (check(TokenType::KeywordIn)) {
            advance();
            kind = "in";
        } else if (check(TokenType::KeywordInout)) {
            advance();
            kind = "inout";
        } else if (check(TokenType::KeywordOut)) {
            advance();
            kind = "out";
        } else if (check(TokenType::KeywordCopy)) {
            advance();
            kind = "copy";
        } else if (check(TokenType::KeywordMove)) {
            advance();
            kind = "move";
        } else if (check(TokenType::KeywordForward)) {
            advance();
            kind = "forward";
        }

        const Token& param_name = consume(TokenType::Identifier, "Expected parameter name");
        static_cast<void>(consume(TokenType::Colon, "Expected ':' between parameter name and type"));

        SourceLocation span_location;
        std::string type_text = collect_text_until({TokenType::Comma, TokenType::RParen}, &span_location);
        type_text = trim_copy(type_text);
        // Allow empty type for type deduction

        Parameter param;
        param.name = param_name.lexeme;
        param.type = std::move(type_text);
        param.location = param_name.location;
        parameters.push_back(std::move(param));

        if (match(TokenType::Comma)) {
            continue;
        }

        static_cast<void>(consume(TokenType::RParen, "Expected ')' after parameter list"));
        break;
    }

    return parameters;
}

Block Parser::parse_block() {
    Block block;
    block.location = previous().location;

    while (!check(TokenType::RBrace) && !is_at_end()) {
        block.statements.push_back(parse_statement());
    }

    static_cast<void>(consume(TokenType::RBrace, "Expected '}' to close block"));
    return block;
}

Statement Parser::parse_statement() {
    if (check(TokenType::KeywordReturn)) {
        const Token& keyword = advance();
        return parse_return_statement(keyword);
    }

    if (check(TokenType::KeywordAssert)) {
        const Token& keyword = advance();
        return parse_assert_statement(keyword);
    }

    if (check(TokenType::Identifier) && peek_next().type == TokenType::Colon) {
        const Token& name = advance();
        static_cast<void>(consume(TokenType::Colon, "Expected ':' after identifier in declaration"));
        return parse_variable_decl(name);
    }

    const Token& start = peek();
    return parse_expression_statement(start);
}

VariableDecl Parser::parse_variable_decl(const Token& name) {
    SourceLocation span_location;
    std::string type_text = collect_text_until({TokenType::Equals, TokenType::Semicolon}, &span_location);
    type_text = trim_copy(type_text);
    // Allow empty type for type deduction

    std::optional<std::string> initializer;
    if (match(TokenType::Equals)) {
        auto init = collect_text_until_semicolon();
        initializer = trim_copy(init);
    }

    static_cast<void>(consume(TokenType::Semicolon, "Expected ';' after declaration"));

    VariableDecl decl;
    decl.name = name.lexeme;
    decl.type = std::move(type_text);
    decl.initializer = std::move(initializer);
    decl.location = name.location;
    return decl;
}

ReturnStmt Parser::parse_return_statement(const Token& keyword) {
    if (check(TokenType::Semicolon)) {
        static_cast<void>(advance());
        return ReturnStmt {std::nullopt, keyword.location};
    }

    auto expression = collect_text_until_semicolon();
    static_cast<void>(consume(TokenType::Semicolon, "Expected ';' after return statement"));
    return ReturnStmt {trim_copy(expression), keyword.location};
}

AssertStmt Parser::parse_assert_statement(const Token& keyword) {
    static_cast<void>(consume(TokenType::LParen, "Expected '(' after assert"));
    auto condition = collect_text_until({TokenType::RParen});
    static_cast<void>(consume(TokenType::RParen, "Expected ')' after assert condition"));
    static_cast<void>(consume(TokenType::Semicolon, "Expected ';' after assert"));
    return AssertStmt {trim_copy(condition), keyword.location};
}

ExpressionStmt Parser::parse_expression_statement(const Token& start_token) {
    auto expression = collect_text_until_semicolon();
    static_cast<void>(consume(TokenType::Semicolon, "Expected ';' after expression"));
    ExpressionStmt stmt;
    stmt.expression = trim_copy(expression);
    stmt.location = start_token.location;
    return stmt;
}

std::string Parser::collect_text_until(const std::vector<TokenType>& end_types, SourceLocation* span_location) {
    if (std::find(end_types.begin(), end_types.end(), peek().type) != end_types.end()) {
        return {};
    }

    const Token& first = peek();
    const Token* last = nullptr;

    auto is_end_type = [&](TokenType t) {
        return std::find(end_types.begin(), end_types.end(), t) != end_types.end();
    };

    while (!is_at_end() && !is_end_type(peek().type)) {
        last = &advance();
    }

    if (!last) {
        return {};
    }

    if (span_location) {
        *span_location = first.location;
    }

    return slice(first, *last);
}

std::string Parser::collect_text_until_semicolon() {
    return collect_text_until({TokenType::Semicolon});
}

std::string Parser::slice(const Token& first, const Token& last) const {
    auto start = first.offset;
    auto end = last.end_offset();
    if (start > end || end > m_source.size()) {
        throw ParseError("Invalid token span");
    }
    std::string text = m_source.substr(start, end - start);

    auto view = std::string_view(text);
    while (!view.empty() && is_whitespace(view.front())) {
        view.remove_prefix(1);
    }
    while (!view.empty() && is_whitespace(view.back())) {
        view.remove_suffix(1);
    }
    return std::string(view);
}

std::string Parser::trim_copy(std::string_view text) {
    while (!text.empty() && is_whitespace(text.front())) {
        text.remove_prefix(1);
    }
    while (!text.empty() && is_whitespace(text.back())) {
        text.remove_suffix(1);
    }
    return std::string(text);
}

bool Parser::is_at_end() const {
    return peek().type == TokenType::EndOfFile;
}

const Token& Parser::peek() const {
    if (m_current >= m_tokens.size()) {
        return m_tokens.back();
    }
    return m_tokens[m_current];
}

const Token& Parser::previous() const {
    return m_tokens[m_current - 1];
}

const Token& Parser::peek_next() const {
    if (m_current + 1 >= m_tokens.size()) {
        return m_tokens.back();
    }
    return m_tokens[m_current + 1];
}

bool Parser::check(TokenType type) const {
    if (is_at_end()) {
        return false;
    }
    return peek().type == type;
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        static_cast<void>(advance());
        return true;
    }
    return false;
}

const Token& Parser::advance() {
    if (!is_at_end()) {
        ++m_current;
    }
    return previous();
}

const Token& Parser::consume(TokenType type, const std::string& message) {
    if (check(type)) {
        return advance();
    }
    std::ostringstream oss;
    const auto& token = peek();
    oss << message << " at " << token.location.line << ':' << token.location.column;
    throw ParseError(oss.str());
}

} // namespace cppfort::stage0

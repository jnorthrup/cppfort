#include "parser.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <utility>

namespace cppfort::stage0 {

namespace {
[[nodiscard]] bool is_whitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
}

Parser::Parser(::std::vector<Token> tokens, ::std::string source)
    : m_tokens(::std::move(tokens)), m_source(::std::move(source)) {}

TranslationUnit Parser::parse() {
    return parse_translation_unit();
}

TranslationUnit Parser::parse_translation_unit() {
    TranslationUnit unit;
    while (!is_at_end()) {
        if (check(TokenType::EndOfFile)) {
            break;
        }

        if (check(TokenType::Preprocessor)) {
            const Token& directive = advance();
            auto text = trim_copy(directive.lexeme);
            if (text.rfind("#include", 0) == 0) {
                unit.includes.push_back(parse_include_directive(directive));
            } else {
                unit.raw_declarations.push_back({::std::move(text), directive.location});
            }
            continue;
        }

        if (is_cpp_raw_declaration()) {
            unit.raw_declarations.push_back(parse_raw_declaration());
            continue;
        }

        auto decl = parse_declaration();
        if (::std::holds_alternative<FunctionDecl>(decl)) {
            unit.functions.push_back(::std::move(::std::get<FunctionDecl>(decl)));
        } else {
            unit.types.push_back(::std::move(::std::get<TypeDecl>(decl)));
        }
    }
    return unit;
}

::std::variant<FunctionDecl, TypeDecl> Parser::parse_declaration() {
    // Support two styles:
    //  1) traditional: name: () -> type = { ... }
    //  2) auto-style: auto name() -> type { ... }
    const Token* name_tok = nullptr;
    if (check(TokenType::KeywordAuto)) {
        // consume 'auto' and then expect an identifier
        static_cast<void>(advance());
        name_tok = &consume(TokenType::Identifier, "Expected identifier after 'auto'");
    } else {
        name_tok = &consume(TokenType::Identifier, "Expected identifier at start of declaration");
        static_cast<void>(consume(TokenType::Colon, "Expected ':' after identifier"));
    }
    const Token& name = *name_tok;

    // Check if it's a type declaration
    if (check(TokenType::KeywordType)) {
        static_cast<void>(advance()); // consume 'type'
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
    const Token* name_tok = nullptr;
    if (check(TokenType::KeywordAuto)) {
        static_cast<void>(advance());
        name_tok = &consume(TokenType::Identifier, "Expected identifier after 'auto'");
    } else {
        name_tok = &consume(TokenType::Identifier, "Expected identifier at start of declaration");
        static_cast<void>(consume(TokenType::Colon, "Expected ':' after identifier"));
    }
    return parse_function_after_name(*name_tok);
}

FunctionDecl Parser::parse_function_after_name(const Token& name) {
    // 'name' and the ':' have already been consumed by the caller.
    auto parameters = parse_parameter_list();

    ::std::optional<::std::string> return_type;
    if (match(TokenType::Arrow)) {
        SourceLocation span_location;
        // Stop collecting return-type text if we hit either '=' (traditional
        // colon-style function body) or '{' (auto-style function body).
        auto text = collect_text_until({TokenType::Equals, TokenType::LBrace}, &span_location);
        return_type = trim_copy(text);
        if (return_type->empty()) {
            throw ParseError("Return type body may not be empty");
        }
    }

    // Accept either an '=' before the function body (traditional syntax)
    // or a '{' directly (auto-style syntax). If '=' is present then the
    // body may be a block or an expression; if '{' is present it's a block.
    FunctionBody body;
    if (match(TokenType::Equals)) {
        if (match(TokenType::LBrace)) {
            body.emplace<Block>(parse_block());
        } else {
            auto expression_text = collect_text_until_semicolon();
            SourceLocation location = previous().location;
            body = ExpressionBody {trim_copy(expression_text), location};
        }
    } else if (match(TokenType::LBrace)) {
        body.emplace<Block>(parse_block());
    } else {
        throw ParseError("Expected '=' or '{' before function body");
    }

    FunctionDecl decl;
    decl.name = name.lexeme;
    decl.parameters = ::std::move(parameters);
    decl.return_type = ::std::move(return_type);
    decl.body = ::std::move(body);
    decl.location = name.location;
    return decl;
}

::std::vector<Parameter> Parser::parse_parameter_list() {
    ::std::vector<Parameter> parameters;
    static_cast<void>(consume(TokenType::LParen, "Expected '(' to start parameter list"));

    if (check(TokenType::RParen)) {
        static_cast<void>(advance());
        return parameters;
    }

    while (true) {
        // Check for optional parameter kind
        cppfort::stage0::ParameterKind pkind = cppfort::stage0::ParameterKind::Default;
        if (check(TokenType::KeywordIn)) {
            static_cast<void>(advance());
            pkind = cppfort::stage0::ParameterKind::In;
        } else if (check(TokenType::KeywordInout)) {
            static_cast<void>(advance());
            pkind = cppfort::stage0::ParameterKind::InOut;
        } else if (check(TokenType::KeywordOut)) {
            static_cast<void>(advance());
            pkind = cppfort::stage0::ParameterKind::Out;
        } else if (check(TokenType::KeywordCopy)) {
            static_cast<void>(advance());
            pkind = cppfort::stage0::ParameterKind::Copy;
        } else if (check(TokenType::KeywordMove)) {
            static_cast<void>(advance());
            pkind = cppfort::stage0::ParameterKind::Move;
        } else if (check(TokenType::KeywordForward)) {
            static_cast<void>(advance());
            pkind = cppfort::stage0::ParameterKind::Forward;
        }

        const Token& param_name = consume(TokenType::Identifier, "Expected parameter name");

        SourceLocation span_location;
        ::std::string type_text;
        // Allow omitted ':' for type deduction (e.g., unnamed/auto-typed params)
        if (check(TokenType::Colon)) {
            static_cast<void>(advance());
            type_text = collect_text_until({TokenType::Comma, TokenType::RParen}, &span_location);
            type_text = trim_copy(type_text);
        } else {
            // empty type -> deduction
            type_text = ::std::string();
        }

    Parameter param;
    param.name = param_name.lexeme;
    param.type = ::std::move(type_text);
    param.location = param_name.location;
    param.kind = pkind;
        parameters.push_back(::std::move(param));

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

Parameter Parser::parse_loop_parameter() {
    Parameter param;
    param.kind = ParameterKind::Default;

    if (check(TokenType::KeywordIn)) {
        static_cast<void>(advance());
        param.kind = ParameterKind::In;
    } else if (check(TokenType::KeywordInout)) {
        static_cast<void>(advance());
        param.kind = ParameterKind::InOut;
    } else if (check(TokenType::KeywordOut)) {
        static_cast<void>(advance());
        param.kind = ParameterKind::Out;
    } else if (check(TokenType::KeywordCopy)) {
        static_cast<void>(advance());
        param.kind = ParameterKind::Copy;
    } else if (check(TokenType::KeywordMove)) {
        static_cast<void>(advance());
        param.kind = ParameterKind::Move;
    } else if (check(TokenType::KeywordForward)) {
        static_cast<void>(advance());
        param.kind = ParameterKind::Forward;
    }

    const Token& name = consume(TokenType::Identifier, "Expected loop variable name");
    param.name = name.lexeme;
    param.location = name.location;

    if (match(TokenType::Colon)) {
        auto type_text = collect_text_until({TokenType::RParen});
        param.type = trim_copy(type_text);
    }

    static_cast<void>(consume(TokenType::RParen, "Expected ')' after loop variable"));

    return param;
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

    if (check(TokenType::KeywordFor)) {
        const Token& keyword = advance();
        return parse_for_chain_statement(keyword);
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
    ::std::string type_text = collect_text_until({TokenType::Equals, TokenType::Semicolon}, &span_location);
    type_text = trim_copy(type_text);
    // Allow empty type for type deduction

    ::std::optional<::std::string> initializer;
    if (match(TokenType::Equals)) {
        auto init = collect_text_until_semicolon();
        initializer = trim_copy(init);
    }

    static_cast<void>(consume(TokenType::Semicolon, "Expected ';' after declaration"));

    VariableDecl decl;
    decl.name = name.lexeme;
    decl.type = ::std::move(type_text);
    decl.initializer = ::std::move(initializer);
    decl.location = name.location;
    return decl;
}

ReturnStmt Parser::parse_return_statement(const Token& keyword) {
    if (check(TokenType::Semicolon)) {
        static_cast<void>(advance());
        return ReturnStmt {::std::nullopt, keyword.location};
    }

    auto expression = collect_text_until_semicolon();
    static_cast<void>(consume(TokenType::Semicolon, "Expected ';' after return statement"));
    return ReturnStmt {trim_copy(expression), keyword.location};
}

AssertStmt Parser::parse_assert_statement(const Token& keyword) {
    ::std::optional<::std::string> category;

    if (match(TokenType::Less)) {
        const Token& less_token = previous();
        ::std::size_t content_start = less_token.end_offset();
        ::std::size_t content_end = content_start;
        ::std::size_t depth = 1;
        while (!is_at_end() && depth > 0) {
            const Token& tok = advance();
            if (tok.type == TokenType::Less) {
                ++depth;
            } else if (tok.type == TokenType::Greater) {
                --depth;
                if (depth == 0) {
                    content_end = tok.offset;
                    break;
                }
            }
        }
        if (depth != 0) {
            throw ParseError("Unterminated assert annotation");
        }
        if (content_end > content_start) {
            category = trim_copy(::std::string_view(m_source).substr(content_start, content_end - content_start));
        } else {
            category = ::std::string{};
        }
    }

    static_cast<void>(consume(TokenType::LParen, "Expected '(' after assert"));
    auto condition = collect_text_until({TokenType::RParen});
    static_cast<void>(consume(TokenType::RParen, "Expected ')' after assert condition"));
    static_cast<void>(consume(TokenType::Semicolon, "Expected ';' after assert"));
    return AssertStmt {trim_copy(condition), ::std::move(category), keyword.location};
}

ForChainStmt Parser::parse_for_chain_statement(const Token& keyword) {
    ForChainStmt stmt;
    stmt.location = keyword.location;

    auto collect_until = [&](const ::std::vector<TokenType>& terms) {
        return trim_copy(collect_text_until(terms));
    };

    stmt.range_expression = collect_until({TokenType::KeywordNext, TokenType::KeywordDo});
    if (stmt.range_expression.empty()) {
        throw ParseError("Expected range expression after 'for'");
    }

    if (check(TokenType::KeywordNext)) {
        static_cast<void>(advance());
        auto expr = collect_until({TokenType::KeywordDo});
        if (expr.empty()) {
            throw ParseError("Expected expression after 'next'");
        }
        stmt.next_expression = ::std::move(expr);
    }

    static_cast<void>(consume(TokenType::KeywordDo, "Expected 'do' in for-chain"));
    static_cast<void>(consume(TokenType::LParen, "Expected '(' after 'do'"));
    stmt.loop_parameter = parse_loop_parameter();

    if (match(TokenType::LBrace)) {
        auto block = parse_block();
        stmt.body = ::std::move(block);
    } else {
        auto block = ::std::make_unique<Block>();
        block->location = stmt.loop_parameter.location;
        block->statements.push_back(parse_statement());
        stmt.body = ::std::move(*block);
    }
    return stmt;
}

ExpressionStmt Parser::parse_expression_statement(const Token& start_token) {
    auto expression = collect_text_until_semicolon();
    static_cast<void>(consume(TokenType::Semicolon, "Expected ';' after expression"));
    ExpressionStmt stmt;
    stmt.expression = trim_copy(expression);
    stmt.location = start_token.location;
    return stmt;
}

::std::string Parser::collect_text_until(const ::std::vector<TokenType>& end_types, SourceLocation* span_location) {
    auto is_end_type = [&](TokenType t) {
        return ::std::find(end_types.begin(), end_types.end(), t) != end_types.end();
    };

    if (is_end_type(peek().type)) {
        return {};
    }

    const Token& first = peek();
    const Token* last = nullptr;

    ::std::size_t paren_depth = 0;
    ::std::size_t brace_depth = 0;
    ::std::size_t bracket_depth = 0;
    ::std::size_t angle_depth = 0;

    while (!is_at_end()) {
        if (is_end_type(peek().type) && paren_depth == 0 && brace_depth == 0 && bracket_depth == 0 && angle_depth == 0) {
            break;
        }

        last = &advance();
        switch (last->type) {
            case TokenType::LParen: ++paren_depth; break;
            case TokenType::RParen:
                if (paren_depth > 0) {
                    --paren_depth;
                }
                break;
            case TokenType::LBrace: ++brace_depth; break;
            case TokenType::RBrace:
                if (brace_depth > 0) {
                    --brace_depth;
                }
                break;
            case TokenType::LBracket: ++bracket_depth; break;
            case TokenType::RBracket:
                if (bracket_depth > 0) {
                    --bracket_depth;
                }
                break;
            case TokenType::Less: ++angle_depth; break;
            case TokenType::Greater:
                if (angle_depth > 0) {
                    --angle_depth;
                }
                break;
            default:
                break;
        }
    }

    if (!last) {
        return {};
    }

    if (span_location) {
        *span_location = first.location;
    }

    return slice(first, *last);
}

::std::string Parser::collect_text_until_semicolon() {
    return collect_text_until({TokenType::Semicolon});
}

IncludeDecl Parser::parse_include_directive(const Token& directive) {
    auto text = trim_copy(directive.lexeme);
    IncludeDecl include;
    include.location = directive.location;

    auto remainder = trim_copy(text.substr(::std::string("#include").size()));
    if (!remainder.empty() && remainder.front() == '<' && remainder.back() == '>') {
        include.is_system = true;
        include.path = remainder.substr(1, remainder.size() - 2);
    } else if (!remainder.empty() && remainder.front() == '"' && remainder.back() == '"') {
        include.path = remainder.substr(1, remainder.size() - 2);
    } else {
        include.path = ::std::move(remainder);
    }
    return include;
}

bool Parser::is_cpp_raw_declaration() const {
    if (check(TokenType::KeywordNamespace) || check(TokenType::KeywordUsing)) {
        return true;
    }
    if (!check(TokenType::Identifier)) {
        return false;
    }

    const ::std::string& lex = peek().lexeme;
    return lex == "template" || lex == "struct" || lex == "class" || lex == "enum" || lex == "extern";
}

RawDecl Parser::parse_raw_declaration() {
    const Token& first = advance();
    const Token* last = &first;

    ::std::size_t brace_depth = 0;
    ::std::size_t paren_depth = 0;
    ::std::size_t bracket_depth = 0;
    ::std::size_t angle_depth = 0;

    auto update_depth = [&](const Token& token) {
        switch (token.type) {
            case TokenType::LBrace: ++brace_depth; break;
            case TokenType::RBrace:
                if (brace_depth > 0) {
                    --brace_depth;
                }
                break;
            case TokenType::LParen: ++paren_depth; break;
            case TokenType::RParen:
                if (paren_depth > 0) {
                    --paren_depth;
                }
                break;
            case TokenType::LBracket: ++bracket_depth; break;
            case TokenType::RBracket:
                if (bracket_depth > 0) {
                    --bracket_depth;
                }
                break;
            case TokenType::Less:
                ++angle_depth;
                break;
            case TokenType::Greater:
                if (angle_depth > 0) {
                    --angle_depth;
                }
                break;
            default:
                break;
        }
    };

    while (!is_at_end()) {
        if (check(TokenType::Semicolon) && brace_depth == 0 && paren_depth == 0 && bracket_depth == 0 && angle_depth == 0) {
            last = &advance();
            break;
        }

        if (check(TokenType::EndOfFile)) {
            break;
        }

        update_depth(peek());
        last = &advance();
    }

    RawDecl raw;
    raw.location = first.location;
    raw.text = slice(first, *last);
    return raw;
}

::std::string Parser::slice(const Token& first, const Token& last) const {
    auto start = first.offset;
    auto end = last.end_offset();
    if (start > end || end > m_source.size()) {
        throw ParseError("Invalid token span");
    }
    ::std::string text = m_source.substr(start, end - start);

    auto view = ::std::string_view(text);
    while (!view.empty() && is_whitespace(view.front())) {
        view.remove_prefix(1);
    }
    while (!view.empty() && is_whitespace(view.back())) {
        view.remove_suffix(1);
    }
    return ::std::string(view);
}

::std::string Parser::trim_copy(::std::string_view text) {
    while (!text.empty() && is_whitespace(text.front())) {
        text.remove_prefix(1);
    }
    while (!text.empty() && is_whitespace(text.back())) {
        text.remove_suffix(1);
    }
    return ::std::string(text);
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

const Token& Parser::consume(TokenType type, const ::std::string& message) {
    if (check(type)) {
        return advance();
    }
    ::std::ostringstream oss;
    const auto& token = peek();
    oss << message << " at " << token.location.line << ':' << token.location.column;
    throw ParseError(oss.str());
}

} // namespace cppfort::stage0

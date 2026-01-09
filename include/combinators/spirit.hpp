#pragma once

// ============================================================================
// Spirit-Like Parser Combinators for Cpp2
// ============================================================================
//
// This file provides Spirit-like operator syntax sugar over the functional
// combinators in ebnf.hpp, specialized for Cpp2 token parsing.
//
// EBNF → C++ Operator Mapping:
//   EBNF sequence (a b)      → a >> b
//   EBNF alternation (a | b) → a | b  
//   EBNF repetition ({ a })  → *a
//   EBNF one-or-more (a+)    → +a
//   EBNF optional ([ a ])    → -a
//   EBNF list (a % sep)      → a % sep
//   Transform                → a[f]
//
// Usage:
//   using namespace cpp2::parser::spirit;
//   
//   // Terminal parsers
//   constexpr auto identifier = token(TokenType::Identifier);
//   constexpr auto lparen = token(TokenType::LeftParen);
//   constexpr auto rparen = token(TokenType::RightParen);
//   constexpr auto comma = token(TokenType::Comma);
//
//   // Combined rules
//   constexpr auto param_list = lparen >> (identifier % comma) >> rparen;
//   constexpr auto optional_params = -param_list;
//
// ============================================================================

#ifndef CPP2_COMBINATORS_SPIRIT_HPP
#define CPP2_COMBINATORS_SPIRIT_HPP

#include "ebnf.hpp"
#include "../lexer.hpp"
#include <string_view>

namespace cpp2::parser::spirit {

// Import ebnf namespace for underlying combinators
namespace ebnf = cpp2::combinators::ebnf;

// ============================================================================
// Token Stream Input Adapter
// ============================================================================
// Adapts std::span<Token> to work with ebnf combinators

struct TokenStream {
    std::span<const cpp2_transpiler::Token> tokens;
    std::size_t pos = 0;
    
    using Iterator = std::span<const cpp2_transpiler::Token>::iterator;
    
    constexpr TokenStream() = default;
    
    constexpr explicit TokenStream(std::span<const cpp2_transpiler::Token> toks) 
        : tokens(toks), pos(0) {}
    
    constexpr TokenStream(std::span<const cpp2_transpiler::Token> toks, std::size_t p)
        : tokens(toks), pos(p) {}
    
    [[nodiscard]] constexpr bool empty() const {
        return pos >= tokens.size() || 
               tokens[pos].type == cpp2_transpiler::TokenType::EndOfFile;
    }
    
    [[nodiscard]] constexpr const cpp2_transpiler::Token& peek() const {
        if (pos >= tokens.size()) {
            return tokens.back(); // EOF token
        }
        return tokens[pos];
    }
    
    constexpr cpp2_transpiler::Token advance() {
        if (pos < tokens.size()) {
            return tokens[pos++];
        }
        return tokens.back();
    }
    
    constexpr Iterator begin() const { return tokens.begin() + pos; }
    constexpr Iterator end() const { return tokens.end(); }
};

// ============================================================================
// Token Parser (Terminal)
// ============================================================================
// Matches a specific TokenType

struct TokenParser {
    cpp2_transpiler::TokenType expected;
    
    constexpr auto parse(TokenStream input) const 
        -> ebnf::Result<cpp2_transpiler::Token, TokenStream> 
    {
        if (input.empty() || input.peek().type != expected) {
            return ebnf::Result<cpp2_transpiler::Token, TokenStream>::fail(input);
        }
        auto tok = input.advance();
        return ebnf::Result<cpp2_transpiler::Token, TokenStream>::ok(tok, input);
    }
};

// Forward declare Proto for factory functions
template<typename P> struct Proto;
template<typename P> constexpr auto lift(P p);

// Factory for token parsers (returns Proto for operator compatibility)
constexpr auto token(cpp2_transpiler::TokenType type) {
    return lift(TokenParser{type});
}

// ============================================================================
// Lexeme Parser (Terminal with Specific Value)
// ============================================================================
// Matches a token type with a specific lexeme value

struct LexemeParser {
    cpp2_transpiler::TokenType expected_type;
    std::string_view expected_lexeme;
    
    constexpr auto parse(TokenStream input) const
        -> ebnf::Result<cpp2_transpiler::Token, TokenStream>
    {
        if (input.empty() || 
            input.peek().type != expected_type ||
            input.peek().lexeme != expected_lexeme) {
            return ebnf::Result<cpp2_transpiler::Token, TokenStream>::fail(input);
        }
        auto tok = input.advance();
        return ebnf::Result<cpp2_transpiler::Token, TokenStream>::ok(tok, input);
    }
};

// Factory for lexeme parsers (returns Proto for operator compatibility)
constexpr auto lexeme(cpp2_transpiler::TokenType type, std::string_view lex) {
    return lift(LexemeParser{type, lex});
}


// ============================================================================
// Keyword Parser
// ============================================================================
// Matches a keyword token

constexpr auto keyword(std::string_view kw) {
    // Keywords use Identifier type with specific lexeme
    return LexemeParser{cpp2_transpiler::TokenType::Identifier, kw};
}

// ============================================================================
// Proto Wrapper (from operators.hpp pattern)
// ============================================================================
// Wraps parsers to provide Spirit-like operator syntax

template<typename P>
struct Proto {
    P parser;
    
    template<typename Input>
    constexpr auto parse(Input input) const {
        return parser.parse(input);
    }
    
    // Transform: p[f] - semantic action
    template<typename F>
    constexpr auto operator[](F f) const {
        auto m = ebnf::map(parser, std::move(f));
        return Proto<decltype(m)>{std::move(m)};
    }
};

// Lift any parser into Proto wrapper
template<typename P>
constexpr auto lift(P p) {
    return Proto<P>{std::move(p)};
}

// ============================================================================
// Operator Overloads (Proto → Proto)
// ============================================================================

// Sequence: a >> b
template<typename L, typename R>
constexpr auto operator>>(Proto<L> l, Proto<R> r) {
    return lift(ebnf::seq(std::move(l.parser), std::move(r.parser)));
}

// Alternative: a | b
template<typename L, typename R>
constexpr auto operator|(Proto<L> l, Proto<R> r) {
    return lift(ebnf::alt(std::move(l.parser), std::move(r.parser)));
}

// List: a % b (sep_by)
template<typename L, typename R>
constexpr auto operator%(Proto<L> l, Proto<R> r) {
    return lift(ebnf::sep_by(std::move(l.parser), std::move(r.parser)));
}

// Zero-or-more: *p
template<typename P>
constexpr auto operator*(Proto<P> p) {
    return lift(ebnf::many(std::move(p.parser)));
}

// One-or-more: +p
template<typename P>
constexpr auto operator+(Proto<P> p) {
    return lift(ebnf::some(std::move(p.parser)));
}

// Optional: -p
template<typename P>
constexpr auto operator-(Proto<P> p) {
    return lift(ebnf::opt(std::move(p.parser)));
}

// Difference: a - b (matches a but not b)
template<typename L, typename R>
constexpr auto operator-(Proto<L> l, Proto<R> r) {
    return lift(ebnf::seq_right(
        ebnf::not_followed_by(std::move(r.parser)),
        std::move(l.parser)
    ));
}

// ============================================================================
// Lifted Token Factories
// ============================================================================

// Create lifted token parser
constexpr auto tok(cpp2_transpiler::TokenType type) {
    return lift(token(type));
}

// Create lifted lexeme parser
constexpr auto lex(cpp2_transpiler::TokenType type, std::string_view lexeme) {
    return lift(LexemeParser{type, lexeme});
}

// Create lifted keyword parser
constexpr auto kw(std::string_view keyword_name) {
    return lift(keyword(keyword_name));
}

// ============================================================================
// Common Token Shortcuts
// ============================================================================

namespace tokens {

using TT = cpp2_transpiler::TokenType;

// Identifiers and Literals
inline const auto identifier   = tok(TT::Identifier);
inline const auto integer_lit  = tok(TT::IntegerLiteral);
inline const auto float_lit    = tok(TT::FloatLiteral);
inline const auto string_lit   = tok(TT::StringLiteral);
inline const auto char_lit     = tok(TT::CharacterLiteral);

// Delimiters
inline const auto lparen       = tok(TT::LeftParen);
inline const auto rparen       = tok(TT::RightParen);
inline const auto lbrace       = tok(TT::LeftBrace);
inline const auto rbrace       = tok(TT::RightBrace);
inline const auto lbracket     = tok(TT::LeftBracket);
inline const auto rbracket     = tok(TT::RightBracket);

// Punctuation
inline const auto semicolon    = tok(TT::Semicolon);
inline const auto colon        = tok(TT::Colon);
inline const auto double_colon = tok(TT::DoubleColon);
inline const auto colon_equal  = tok(TT::ColonEqual);
inline const auto comma        = tok(TT::Comma);
inline const auto dot          = tok(TT::Dot);
inline const auto arrow        = tok(TT::Arrow);
inline const auto at           = tok(TT::At);
inline const auto underscore   = tok(TT::Underscore);

// Operators
inline const auto eq           = tok(TT::Equal);
inline const auto plus         = tok(TT::Plus);
inline const auto minus        = tok(TT::Minus);
inline const auto star         = tok(TT::Asterisk);
inline const auto slash        = tok(TT::Slash);
inline const auto ampersand    = tok(TT::Ampersand);
inline const auto pipe         = tok(TT::Pipe);
inline const auto pipeline     = tok(TT::Pipeline);
inline const auto lt           = tok(TT::LessThan);
inline const auto gt           = tok(TT::GreaterThan);

// Keywords
inline const auto kw_let       = tok(TT::Let);
inline const auto kw_const     = tok(TT::Const);
inline const auto kw_func      = tok(TT::Func);
inline const auto kw_type      = tok(TT::Type);
inline const auto kw_namespace = tok(TT::Namespace);
inline const auto kw_if        = tok(TT::If);
inline const auto kw_else      = tok(TT::Else);
inline const auto kw_while     = tok(TT::While);
inline const auto kw_for       = tok(TT::For);
inline const auto kw_return    = tok(TT::Return);
inline const auto kw_in        = tok(TT::In);
inline const auto kw_out       = tok(TT::Out);
inline const auto kw_inout     = tok(TT::Inout);
inline const auto kw_copy      = tok(TT::Copy);
inline const auto kw_move      = tok(TT::Move);
inline const auto kw_forward   = tok(TT::Forward);
inline const auto kw_inspect   = tok(TT::Inspect);
inline const auto kw_is        = tok(TT::Is);
inline const auto kw_as        = tok(TT::As);

} // namespace tokens

// ============================================================================
// Recursive Rule Support
// ============================================================================
// For defining recursive grammar rules

template<typename T>
struct RecursiveRule {
    std::function<ebnf::Result<T, TokenStream>(TokenStream)> parser_fn;
    
    auto parse(TokenStream input) const {
        return parser_fn(input);
    }
};

// Create a recursive rule by passing a lambda that receives the rule reference
template<typename T, typename F>
auto recursive(F factory) {
    RecursiveRule<T> rule;
    rule.parser_fn = [&rule, factory](TokenStream input) {
        return factory(lift(rule)).parser.parse(input);
    };
    return lift(rule);
}

// ============================================================================
// Utility Combinators
// ============================================================================

// Between: open >> content << close
template<typename Open, typename Content, typename Close>
constexpr auto between(Proto<Open> open, Proto<Content> content, Proto<Close> close) {
    return lift(ebnf::between(
        std::move(open.parser),
        std::move(content.parser),
        std::move(close.parser)
    ));
}

// Parenthesized: ( content )
template<typename Content>
constexpr auto parens(Proto<Content> content) {
    return between(tokens::lparen, std::move(content), tokens::rparen);
}

// Braced: { content }
template<typename Content>
constexpr auto braces(Proto<Content> content) {
    return between(tokens::lbrace, std::move(content), tokens::rbrace);
}

// Bracketed: [ content ]
template<typename Content>
constexpr auto brackets(Proto<Content> content) {
    return between(tokens::lbracket, std::move(content), tokens::rbracket);
}

// Separated by comma
template<typename Item>
constexpr auto comma_list(Proto<Item> item) {
    return item % tokens::comma;
}

} // namespace cpp2::parser::spirit

#endif // CPP2_COMBINATORS_SPIRIT_HPP

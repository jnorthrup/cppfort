#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace cpp2_transpiler {

enum class TokenType : uint8_t {
    // Literals
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    CharacterLiteral,
    BooleanLiteral,

    // Keywords
    Func,
    Type,
    Namespace,
    Operator,
    Struct,
    Class,
    Interface,
    Enum,
    Union,
    Const,
    Let,
    Mut,
    Return,
    If,
    Else,
    While,
    For,
    Do,
    Break,
    Continue,
    Switch,
    Case,
    Default,
    Inspect,
    When,
    Is,
    As,
    In,
    Where,
    Concept,
    Requires,
    Auto,
    This,
    Base,
    Super,
    Import,
    Module,
    Export,
    Public,
    Private,
    Protected,
    Virtual,
    Override,
    Final,
    Explicit,
    Implicit,
    Unsafe,
    Checked,
    Try,
    Catch,
    Throw,
    Noexcept,
    Decltype,
    Sizeof,
    Typeid,
    Alignof,
    Alignas,
    Static_assert,

    // Operators and punctuation
    Colon,           // :
    DoubleColon,     // ::
    Semicolon,       // ;
    Comma,           // ,
    Dot,             // .
    DoubleDot,       // ..
    TripleDot,       // ...
    Ellipsis,        // ...
    Arrow,           // ->
    FatArrow,        // =>
    DoubleArrow,     // =>
    Equal,           // =
    DoubleEqual,     // ==
    NotEqual,        // !=
    LessThan,        // <
    GreaterThan,     // >
    LessThanOrEqual, // <=
    GreaterThanOrEqual, // >=
    Plus,            // +
    Minus,           // -
    Asterisk,        // *
        Elvis,           // ?:
    Slash,           // /
    Percent,         // %
        SlashEqual,      // /=
    PlusPlus,        // ++
    MinusMinus,      // --
    PlusEqual,       // +=
    MinusEqual,      // -=
    AsteriskEqual,   // *=
    PercentEqual,     // %=
    Exponentiation,  // **
    Ampersand,       // &
    DoubleAmpersand, // &&
    AmpersandEqual,  // &=
    Pipe,            // |
    DoublePipe,      // ||
    PipeEqual,       // |=
    Caret,           // ^
    CaretEqual,      // ^=
    Tilde,           // ~
    Exclamation,     // !
    Question,        // ?
    Dollar,          // $
    At,              // @
    Hash,            // # (preprocessor directives)

    // Brackets
    LeftParen,       // (
    RightParen,      // )
    LeftBracket,     // [
    RightBracket,    // ]
    LeftBrace,       // {
    RightBrace,      // }

    LeftShift,       // <<
    LeftShiftEqual,  // <<=
    RightShift,      // >>
    RightShiftEqual, // >>=

    // Special
    EndOfFile,
    Unknown,

    // Cpp2 specific
    Underscore,      // _
    RangeInclusive,  // ..=
    RangeExclusive,  // ..<
    ContractPre,     // pre
    ContractPost,    // post
    ContractAssert,  // assert
    Meta,            // @meta
    TemplateBegin,   // <
    TemplateEnd,     // >
    True,
    False,
    Using,
    Template,

    // Parameter qualifiers (corpus-derived)
    Inout,           // inout
    Out,             // out
    Move,            // move
    Forward,         // forward

    // Concurrency keywords (Kotlin-style)
    Suspend,         // suspend - marks function as suspendable
    Async,           // async - marks function as async
    Await,           // await - suspend until value ready
    Launch,          // launch - spawn fire-and-forget coroutine
    CoroutineScope,  // coroutineScope - structured concurrency block
    Channel,         // channel - channel declaration
    Select,          // select - multi-channel select
    ParallelFor,     // parallel_for - parallel loop
};

struct Token {
    TokenType type;
    std::string_view lexeme;
    std::size_t line;
    std::size_t column;
    std::size_t position;

    Token(TokenType t, std::string_view lex, std::size_t l, std::size_t c, std::size_t p)
        : type(t), lexeme(lex), line(l), column(c), position(p) {}
};

class Lexer {
public:
    explicit Lexer(std::string_view source);
    explicit Lexer(const std::string& source);

    std::vector<Token> tokenize();

private:
    std::string_view source;
    std::size_t current;
    std::size_t line;
    std::size_t column;
    std::size_t start;

    std::vector<Token> tokens;

    void scan_token();

    char advance();
    char peek() const;
    char peek_next() const;
    bool match(char expected);
    bool is_at_end() const;

    void add_token(TokenType type);
    void add_token(TokenType type, std::string_view lexeme);

    void scan_identifier();
    void scan_number();
    void scan_string();
    void scan_character();
    void scan_line_comment();
    void scan_block_comment();
    void scan_preprocessor();

    bool is_digit(char c) const;
    bool is_identifier_start(char c) const;
    bool is_identifier_char(char c) const;

    TokenType check_keyword(std::size_t start, std::size_t length,
                           const char* rest, TokenType type);
    TokenType identifier_type();

    void skip_whitespace();
};

} // namespace cpp2_transpiler
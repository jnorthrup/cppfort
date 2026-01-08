// include/core/tokens.hpp - Token types and Token struct for cpp2_transpiler
// Extracted from lexer.hpp as part of modular build restructuring
#pragma once

#include <string_view>
#include <cstddef>
#include <cstdint>

namespace cpp2_transpiler {

enum class TokenType : uint8_t {
    // Literals
    Identifier,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    InterpolatedRawStringLiteral,  // Cpp2 $R"..." raw string with interpolation
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
    Next,
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
    That,   // Cpp2 'that' parameter for copy/move constructors
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
    Throws,   // Cpp2 function exception specifier
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
    ColonEqual,      // := (type-deduced declaration)
    EqualColon,      // =: (function named return type)
    DoubleEqual,     // ==
    NotEqual,        // !=
    LessThan,        // <
    GreaterThan,     // >
    LessThanOrEqual, // <=
    GreaterThanOrEqual, // >=
    Spaceship,       // <=>
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
    PercentEqual,    // %=
    Exponentiation,  // **
    Ampersand,       // &
    DoubleAmpersand, // &&
    AmpersandEqual,  // &=
    Pipe,            // |
    DoublePipe,      // ||
    PipeEqual,       // |=
    Pipeline,        // |> (pipeline operator)
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
    Copy,            // copy
    InRef,           // in_ref (reference parameter)
    ForwardRef,      // forward_ref (forwarding reference)

    // Concurrency keywords (Kotlin-style)
    Suspend,         // suspend - marks function as suspendable
    Async,           // async - marks function as async
    Await,           // await - suspend until value ready
    Launch,          // launch - spawn fire-and-forget coroutine
    CoroutineScope,  // coroutineScope - structured concurrency block
    Channel,         // channel - channel declaration
    Select,          // select - multi-channel select
    ParallelFor,     // parallel_for - parallel loop

    // Markdown blocks for CAS-linked modules
    MarkdownBlock,   // markdown block with embedded content

    // C++26 contract attributes
    DoubleLeftBracket,  // [[
    DoubleRightBracket, // ]]
    AttributeExpect,    // expects (in [[expects]])
    AttributeEnsure,    // ensures (in [[ensures]])
    AttributeAssert,    // assert (in [[assert]])
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

} // namespace cpp2_transpiler

#pragma once

#include <cstddef>
#include <string>
#include <utility>

namespace cppfort::stage0 {

struct SourceLocation {
    std::string file;
    std::size_t line {0};
    std::size_t column {0};

    SourceLocation() = default;
    SourceLocation(std::string file_, std::size_t line_, std::size_t column_)
        : file(std::move(file_)), line(line_), column(column_) {}
};

enum class TokenType {
    // structural
    Identifier,
    Number,
    String,
    Char,
    Colon,
    Semicolon,
    Comma,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Arrow,
    Equals,
    DoubleColon,
    Dot,
    Less,
    Greater,
    LShift,
    RShift,
    Ampersand,
    Star,
    Plus,
    Minus,
    Slash,
    Percent,
    Pipe,
    Caret,
    Exclamation,
    Question,
    Dollar,
    At,
    Preprocessor,

    // Cpp2 specific keywords
    KeywordReturn,
    KeywordIf,
    KeywordElse,
    KeywordFor,
    KeywordWhile,
    KeywordDo,
    KeywordNext,
    KeywordAssert,
    KeywordPre,
    KeywordPost,
    KeywordAuto,
    KeywordUsing,
    KeywordNamespace,
    KeywordType,
    KeywordPublic,
    KeywordProtected,
    KeywordPrivate,
    KeywordVirtual,
    KeywordOverride,
    KeywordFinal,
    KeywordIn,
    KeywordInout,
    KeywordOut,
    KeywordCopy,
    KeywordMove,
    KeywordForward,

    EndOfFile,
};

struct Token {
    TokenType type {TokenType::EndOfFile};
    std::string lexeme;
    SourceLocation location;
    std::size_t offset {0};
    std::size_t length {0};

    Token() = default;

    Token(TokenType type_, std::string lexeme_, SourceLocation loc_, std::size_t offset_, std::size_t length_)
        : type(type_), lexeme(std::move(lexeme_)), location(std::move(loc_)), offset(offset_), length(length_) {}

    std::size_t end_offset() const { return offset + length; }
};

} // namespace cppfort::stage0

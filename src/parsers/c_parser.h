#pragma once
// C11/C17 Parser with preprocessor support
// Converts C source code to unified IR

#include "../ir/ir.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <stack>

namespace parsers {

// Token types for C lexer
enum class CTokenType {
    // Literals
    IntLiteral,
    FloatLiteral,
    CharLiteral,
    StringLiteral,

    // Identifiers and keywords
    Identifier,
    Keyword,

    // Operators
    Plus, Minus, Star, Slash, Percent,
    Ampersand, Pipe, Caret, Tilde,
    LShift, RShift,
    Less, Greater, LessEq, GreaterEq,
    Equal, NotEqual,
    LogicalAnd, LogicalOr, LogicalNot,
    Assign,
    PlusAssign, MinusAssign, StarAssign, SlashAssign,
    PercentAssign, AmpAssign, PipeAssign, CaretAssign,
    LShiftAssign, RShiftAssign,
    Increment, Decrement,
    Arrow, Dot,
    Question, Colon,

    // Punctuation
    LParen, RParen,
    LBracket, RBracket,
    LBrace, RBrace,
    Semicolon, Comma,

    // Preprocessor
    Hash, HashHash,
    PreprocessorDirective,

    // Special
    Eof,
    Invalid
};

struct CToken {
    CTokenType type;
    std::string value;
    size_t line;
    size_t column;
};

// C Lexer
class CLexer {
public:
    explicit CLexer(const std::string& source);

    CToken nextToken();
    CToken peekToken();
    void advance();
    bool isEof() const;

private:
    std::string source;
    size_t pos = 0;
    size_t line = 1;
    size_t column = 1;
    CToken current_token;
    bool has_peeked = false;

    void skipWhitespace();
    void skipComment();
    CToken readNumber();
    CToken readIdentifier();
    CToken readString();
    CToken readChar();
    CToken readOperator();
    CToken readPreprocessor();

    bool isDigit(char c) const;
    bool isAlpha(char c) const;
    bool isAlnum(char c) const;
    char peekChar(size_t offset = 0) const;
    char nextChar();
};

// C Parser
class CParser {
public:
    explicit CParser(const std::string& source);

    std::shared_ptr<ir::Module> parse();
    std::vector<std::string> getErrors() const { return errors; }

private:
    CLexer lexer;
    ir::IRBuilder builder;
    std::vector<std::string> errors;

    // Symbol table for type checking
    struct SymbolInfo {
        std::string name;
        ir::TypePtr type;
        bool is_function = false;
        bool is_typedef = false;
    };
    std::unordered_map<std::string, SymbolInfo> symbols;
    std::stack<std::unordered_map<std::string, SymbolInfo>> scope_stack;

    // Parsing state
    std::shared_ptr<ir::Module> current_module;
    std::vector<std::string> typedefs;  // Track typedef names

    // Top-level parsing
    void parseTranslationUnit();
    void parseExternalDeclaration();

    // Declarations
    ir::DeclPtr parseDeclaration();
    std::vector<ir::DeclPtr> parseDeclarationSpecifiers();
    ir::TypePtr parseTypeSpecifier();
    ir::DeclPtr parseDeclarator(ir::TypePtr base_type);
    ir::TypePtr parsePointerType(ir::TypePtr base);
    ir::DeclPtr parseParameterDeclaration();
    std::vector<std::pair<std::string, ir::TypePtr>> parseParameterList();

    // Type parsing
    ir::TypePtr parseStructOrUnion();
    ir::TypePtr parseEnum();
    ir::TypePtr parseTypedefName();

    // Functions
    ir::FuncPtr parseFunction(const std::string& name, ir::TypePtr return_type,
                             std::vector<std::pair<std::string, ir::TypePtr>> params);

    // Statements
    ir::StmtPtr parseStatement();
    ir::StmtPtr parseCompoundStatement();
    ir::StmtPtr parseExpressionStatement();
    ir::StmtPtr parseSelectionStatement();
    ir::StmtPtr parseIterationStatement();
    ir::StmtPtr parseJumpStatement();
    ir::StmtPtr parseLabeledStatement();

    // Expressions
    ir::ExprPtr parseExpression();
    ir::ExprPtr parseAssignmentExpression();
    ir::ExprPtr parseConditionalExpression();
    ir::ExprPtr parseLogicalOrExpression();
    ir::ExprPtr parseLogicalAndExpression();
    ir::ExprPtr parseBitwiseOrExpression();
    ir::ExprPtr parseBitwiseXorExpression();
    ir::ExprPtr parseBitwiseAndExpression();
    ir::ExprPtr parseEqualityExpression();
    ir::ExprPtr parseRelationalExpression();
    ir::ExprPtr parseShiftExpression();
    ir::ExprPtr parseAdditiveExpression();
    ir::ExprPtr parseMultiplicativeExpression();
    ir::ExprPtr parseCastExpression();
    ir::ExprPtr parseUnaryExpression();
    ir::ExprPtr parsePostfixExpression();
    ir::ExprPtr parsePrimaryExpression();

    // Preprocessor handling
    void handlePreprocessorDirective(const std::string& directive);
    void handleInclude(const std::string& filename);
    void handleDefine(const std::string& macro, const std::string& value);
    void handleIfdef(const std::string& macro);

    // Helper functions
    bool expect(CTokenType type);
    bool match(CTokenType type);
    bool matchKeyword(const std::string& keyword);
    void consume(CTokenType type, const std::string& error_msg);
    void consumeKeyword(const std::string& keyword, const std::string& error_msg);

    void error(const std::string& msg);
    void enterScope();
    void exitScope();
    void addSymbol(const std::string& name, ir::TypePtr type, bool is_function = false);
    SymbolInfo* lookupSymbol(const std::string& name);

    // Type helpers
    ir::TypePtr makeBasicType(const std::string& type_name);
    bool isTypedefName(const std::string& name) const;
    std::string getTypeString(ir::TypePtr type) const;
};

// Preprocessor
class CPreprocessor {
public:
    explicit CPreprocessor(const std::string& source);

    std::string process();
    void addIncludePath(const std::string& path);
    void defineM macro(const std::string& name, const std::string& value = "");

private:
    std::string source;
    std::vector<std::string> include_paths;
    std::unordered_map<std::string, std::string> macros;
    std::unordered_map<std::string, std::vector<std::string>> function_macros;

    std::string expandMacros(const std::string& line);
    std::string processInclude(const std::string& filename);
    void processDefine(const std::string& line);
    void processIf(const std::string& condition);
    void processIfdef(const std::string& macro);
    void processIfndef(const std::string& macro);

    bool evaluateCondition(const std::string& condition);
    std::string findIncludeFile(const std::string& filename);
};

} // namespace parsers
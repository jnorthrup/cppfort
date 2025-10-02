#pragma once

#include "node.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace cppfort::c {

/**
 * C11/C17 Parser - Emits Sea of Nodes IR
 *
 * This parser handles full C syntax including:
 * - Preprocessor directives (macros, includes, conditionals)
 * - Pointer arithmetic and manual memory management
 * - struct/union types with flexible arrays
 * - Designated initializers
 * - Variadic functions (va_list, va_arg)
 * - Implicit conversions and undefined behavior preservation
 *
 * Output: Node graph with C-specific semantics preserved
 */

// Forward declarations
class CPreprocessor;
class CToken;
class CASTNode;

// ============================================================================
// C Token Types
// ============================================================================
enum class CTokenType {
    // Keywords
    AUTO, BREAK, CASE, CHAR, CONST, CONTINUE, DEFAULT, DO,
    DOUBLE, ELSE, ENUM, EXTERN, FLOAT, FOR, GOTO, IF,
    INLINE, INT, LONG, REGISTER, RESTRICT, RETURN, SHORT, SIGNED,
    SIZEOF, STATIC, STRUCT, SWITCH, TYPEDEF, UNION, UNSIGNED, VOID,
    VOLATILE, WHILE,
    _ALIGNAS, _ALIGNOF, _ATOMIC, _BOOL, _COMPLEX, _GENERIC,
    _IMAGINARY, _NORETURN, _STATIC_ASSERT, _THREAD_LOCAL,

    // Identifiers and literals
    IDENTIFIER, INTEGER_LITERAL, FLOAT_LITERAL, STRING_LITERAL, CHAR_LITERAL,

    // Operators and punctuation
    PLUS, MINUS, STAR, SLASH, PERCENT, AMPERSAND, PIPE, CARET, TILDE,
    EXCLAIM, QUESTION, COLON, SEMICOLON, COMMA, DOT, ARROW,
    LBRACE, RBRACE, LPAREN, RPAREN, LBRACKET, RBRACKET,
    LT, GT, LE, GE, EQ, NE, AND, OR,
    ASSIGN, PLUS_ASSIGN, MINUS_ASSIGN, STAR_ASSIGN, SLASH_ASSIGN,
    PERCENT_ASSIGN, AMPERSAND_ASSIGN, PIPE_ASSIGN, CARET_ASSIGN,
    LSHIFT, RSHIFT, LSHIFT_ASSIGN, RSHIFT_ASSIGN,
    INCREMENT, DECREMENT, SIZEOF_OP, ALIGNOF_OP,

    // Preprocessor
    HASH, HASH_HASH, DEFINED,

    // Special
    EOF_TOKEN, INVALID
};

struct CToken {
    CTokenType type;
    std::string text;
    std::string filename;
    int line;
    int column;

    CToken(CTokenType t, const std::string& txt, const std::string& file, int ln, int col)
        : type(t), text(txt), filename(file), line(ln), column(col) {}
};

// ============================================================================
// C Preprocessor
// ============================================================================
class CPreprocessor {
private:
    std::unordered_map<std::string, std::string> _macros;
    std::vector<std::string> _includePaths;
    std::vector<std::string> _conditionalStack;  // #if/#ifdef nesting

public:
    CPreprocessor(const std::vector<std::string>& includePaths = {})
        : _includePaths(includePaths) {}

    // Define a macro
    void defineMacro(const std::string& name, const std::string& value);

    // Undefine a macro
    void undefineMacro(const std::string& name);

    // Check if macro is defined
    bool isDefined(const std::string& name) const;

    // Expand macros in token stream
    std::vector<CToken> expand(const std::vector<CToken>& tokens);

    // Process #include directive
    std::vector<CToken> processInclude(const std::string& filename);

    // Process #if/#ifdef/#ifndef conditionals
    bool evaluateConditional(const std::string& condition);
};

// ============================================================================
// C AST Node (intermediate before Node graph emission)
// ============================================================================
enum class CASTNodeType {
    TRANSLATION_UNIT, FUNCTION_DECL, VAR_DECL, STRUCT_DECL, UNION_DECL,
    ENUM_DECL, TYPEDEF_DECL, COMPOUND_STMT, EXPR_STMT, RETURN_STMT,
    IF_STMT, WHILE_STMT, DO_STMT, FOR_STMT, SWITCH_STMT, CASE_STMT,
    DEFAULT_STMT, BREAK_STMT, CONTINUE_STMT, GOTO_STMT, LABEL_STMT,
    BINARY_EXPR, UNARY_EXPR, CALL_EXPR, CAST_EXPR, SIZEOF_EXPR,
    ALIGNOF_EXPR, ARRAY_SUBSCRIPT, MEMBER_ACCESS, POINTER_DEREF,
    ADDRESS_OF, COMMA_EXPR, CONDITIONAL_EXPR, COMPOUND_LITERAL,
    DESIGNATED_INIT, INIT_LIST, INTEGER_CONST, FLOAT_CONST, STRING_CONST,
    CHAR_CONST, IDENTIFIER_REF
};

class CASTNode {
public:
    CASTNodeType type;
    std::vector<std::unique_ptr<CASTNode>> children;
    std::string value;  // For identifiers, literals
    int line;
    int column;

    CASTNode(CASTNodeType t, int ln = 0, int col = 0)
        : type(t), line(ln), column(col) {}

    void addChild(std::unique_ptr<CASTNode> child) {
        children.push_back(std::move(child));
    }
};

// ============================================================================
// C Parser
// ============================================================================
class CParser {
private:
    std::vector<CToken> _tokens;
    size_t _pos;
    CPreprocessor* _preprocessor;

    // Parser state
    std::unordered_map<std::string, ir::Type*> _symbolTable;
    std::unordered_map<std::string, ir::Node*> _variables;

    // Current parsing context
    ir::Node* _currentFunction;
    ir::CFGNode* _currentBlock;

    // Helper methods
    const CToken& peek() const;
    const CToken& advance();
    bool match(CTokenType type);
    bool expect(CTokenType type);
    void error(const std::string& message);

    // Recursive descent parsing - declarations
    std::unique_ptr<CASTNode> parseTranslationUnit();
    std::unique_ptr<CASTNode> parseFunctionDecl();
    std::unique_ptr<CASTNode> parseVarDecl();
    std::unique_ptr<CASTNode> parseStructDecl();
    std::unique_ptr<CASTNode> parseUnionDecl();
    std::unique_ptr<CASTNode> parseEnumDecl();
    std::unique_ptr<CASTNode> parseTypedefDecl();

    // Statement parsing
    std::unique_ptr<CASTNode> parseStatement();
    std::unique_ptr<CASTNode> parseCompoundStatement();
    std::unique_ptr<CASTNode> parseIfStatement();
    std::unique_ptr<CASTNode> parseWhileStatement();
    std::unique_ptr<CASTNode> parseDoStatement();
    std::unique_ptr<CASTNode> parseForStatement();
    std::unique_ptr<CASTNode> parseSwitchStatement();

    // Type parsing
    ir::Type* parseTypeSpecifier();
    ir::Type* parseDeclarator(ir::Type* baseType);

public:
    CParser(const std::vector<CToken>& tokens, CPreprocessor* preprocessor = nullptr)
        : _tokens(tokens), _pos(0), _preprocessor(preprocessor),
          _currentFunction(nullptr), _currentBlock(nullptr) {}

    // Parse C source into AST
    std::unique_ptr<CASTNode> parse();

    // Expression parsing (public for testing)
    std::unique_ptr<CASTNode> parseExpression();
    std::unique_ptr<CASTNode> parseAssignmentExpression();
    std::unique_ptr<CASTNode> parseConditionalExpression();
    std::unique_ptr<CASTNode> parseLogicalOrExpression();
    std::unique_ptr<CASTNode> parseLogicalAndExpression();
    std::unique_ptr<CASTNode> parseBitwiseOrExpression();
    std::unique_ptr<CASTNode> parseBitwiseXorExpression();
    std::unique_ptr<CASTNode> parseBitwiseAndExpression();
    std::unique_ptr<CASTNode> parseEqualityExpression();
    std::unique_ptr<CASTNode> parseRelationalExpression();
    std::unique_ptr<CASTNode> parseShiftExpression();
    std::unique_ptr<CASTNode> parseAdditiveExpression();
    std::unique_ptr<CASTNode> parseMultiplicativeExpression();
    std::unique_ptr<CASTNode> parseCastExpression();
    std::unique_ptr<CASTNode> parseUnaryExpression();
    std::unique_ptr<CASTNode> parsePostfixExpression();
    std::unique_ptr<CASTNode> parsePrimaryExpression();

    // Convert AST to Sea of Nodes IR
    ir::Node* emit(const CASTNode* ast);

    // Full pipeline: source → tokens → AST → Node graph
    static ir::Node* parseSource(const std::string& source,
                                  const std::vector<std::string>& includePaths = {});
};

// ============================================================================
// C Lexer
// ============================================================================
class CLexer {
private:
    std::string _source;
    size_t _pos;
    std::string _filename;
    int _line;
    int _column;

    char peek() const;
    char advance();
    void skipWhitespace();
    void skipLineComment();
    void skipBlockComment();

    CToken lexIdentifierOrKeyword();
    CToken lexNumber();
    CToken lexString();
    CToken lexChar();
    CToken lexOperator();

public:
    CLexer(const std::string& source, const std::string& filename = "<input>")
        : _source(source), _pos(0), _filename(filename), _line(1), _column(1) {}

    std::vector<CToken> tokenize();
};

} // namespace cppfort::c

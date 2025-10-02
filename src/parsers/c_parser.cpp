#include "c_parser.h"
#include <cctype>
#include <sstream>
#include <fstream>
#include <algorithm>

namespace parsers {

// CLexer implementation

CLexer::CLexer(const std::string& src) : source(src) {}

CToken CLexer::nextToken() {
    if (has_peeked) {
        has_peeked = false;
        return current_token;
    }

    skipWhitespace();

    if (pos >= source.length()) {
        return {CTokenType::Eof, "", line, column};
    }

    size_t start_line = line;
    size_t start_column = column;

    char c = peekChar();

    // Comments
    if (c == '/' && peekChar(1) == '/') {
        skipComment();
        return nextToken();
    }
    if (c == '/' && peekChar(1) == '*') {
        skipComment();
        return nextToken();
    }

    // Preprocessor
    if (c == '#') {
        return readPreprocessor();
    }

    // Numbers
    if (isDigit(c)) {
        return readNumber();
    }

    // Identifiers and keywords
    if (isAlpha(c) || c == '_') {
        return readIdentifier();
    }

    // Strings
    if (c == '"') {
        return readString();
    }

    // Characters
    if (c == '\'') {
        return readChar();
    }

    // Operators and punctuation
    return readOperator();
}

CToken CLexer::peekToken() {
    if (!has_peeked) {
        current_token = nextToken();
        has_peeked = true;
    }
    return current_token;
}

void CLexer::advance() {
    nextToken();
}

bool CLexer::isEof() const {
    return pos >= source.length();
}

void CLexer::skipWhitespace() {
    while (pos < source.length() && std::isspace(source[pos])) {
        if (source[pos] == '\n') {
            line++;
            column = 1;
        } else {
            column++;
        }
        pos++;
    }
}

void CLexer::skipComment() {
    if (peekChar() == '/' && peekChar(1) == '/') {
        // Line comment
        pos += 2;
        column += 2;
        while (pos < source.length() && source[pos] != '\n') {
            pos++;
            column++;
        }
    } else if (peekChar() == '/' && peekChar(1) == '*') {
        // Block comment
        pos += 2;
        column += 2;
        while (pos < source.length() - 1) {
            if (source[pos] == '*' && source[pos + 1] == '/') {
                pos += 2;
                column += 2;
                break;
            }
            if (source[pos] == '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
            pos++;
        }
    }
}

CToken CLexer::readNumber() {
    size_t start = pos;
    size_t start_line = line;
    size_t start_col = column;
    bool is_float = false;

    // Handle hex numbers
    if (peekChar() == '0' && (peekChar(1) == 'x' || peekChar(1) == 'X')) {
        pos += 2;
        column += 2;
        while (pos < source.length() && std::isxdigit(source[pos])) {
            pos++;
            column++;
        }
    } else {
        // Decimal or octal
        while (pos < source.length() && isDigit(source[pos])) {
            pos++;
            column++;
        }

        // Check for float
        if (peekChar() == '.') {
            is_float = true;
            pos++;
            column++;
            while (pos < source.length() && isDigit(source[pos])) {
                pos++;
                column++;
            }
        }

        // Check for exponent
        if (peekChar() == 'e' || peekChar() == 'E') {
            is_float = true;
            pos++;
            column++;
            if (peekChar() == '+' || peekChar() == '-') {
                pos++;
                column++;
            }
            while (pos < source.length() && isDigit(source[pos])) {
                pos++;
                column++;
            }
        }

        // Suffixes
        while (pos < source.length() &&
               (source[pos] == 'u' || source[pos] == 'U' ||
                source[pos] == 'l' || source[pos] == 'L' ||
                source[pos] == 'f' || source[pos] == 'F')) {
            pos++;
            column++;
        }
    }

    std::string value = source.substr(start, pos - start);
    return {is_float ? CTokenType::FloatLiteral : CTokenType::IntLiteral,
            value, start_line, start_col};
}

CToken CLexer::readIdentifier() {
    size_t start = pos;
    size_t start_line = line;
    size_t start_col = column;

    while (pos < source.length() && (isAlnum(source[pos]) || source[pos] == '_')) {
        pos++;
        column++;
    }

    std::string value = source.substr(start, pos - start);

    // Check if it's a keyword
    static const std::unordered_set<std::string> keywords = {
        "auto", "break", "case", "char", "const", "continue", "default",
        "do", "double", "else", "enum", "extern", "float", "for", "goto",
        "if", "inline", "int", "long", "register", "restrict", "return",
        "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
        "union", "unsigned", "void", "volatile", "while",
        "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic",
        "_Imaginary", "_Noreturn", "_Static_assert", "_Thread_local"
    };

    CTokenType type = keywords.count(value) ? CTokenType::Keyword : CTokenType::Identifier;
    return {type, value, start_line, start_col};
}

CToken CLexer::readString() {
    size_t start_line = line;
    size_t start_col = column;
    pos++; // Skip opening quote
    column++;

    std::string value;
    while (pos < source.length() && source[pos] != '"') {
        if (source[pos] == '\\' && pos + 1 < source.length()) {
            pos++;
            column++;
            char escape = source[pos];
            switch (escape) {
                case 'n': value += '\n'; break;
                case 't': value += '\t'; break;
                case 'r': value += '\r'; break;
                case '\\': value += '\\'; break;
                case '"': value += '"'; break;
                case '\'': value += '\''; break;
                case '0': value += '\0'; break;
                default: value += escape; break;
            }
        } else {
            value += source[pos];
        }
        if (source[pos] == '\n') {
            line++;
            column = 1;
        } else {
            column++;
        }
        pos++;
    }

    if (pos < source.length()) {
        pos++; // Skip closing quote
        column++;
    }

    return {CTokenType::StringLiteral, value, start_line, start_col};
}

CToken CLexer::readChar() {
    size_t start_line = line;
    size_t start_col = column;
    pos++; // Skip opening quote
    column++;

    std::string value;
    if (source[pos] == '\\' && pos + 1 < source.length()) {
        pos++;
        column++;
        char escape = source[pos];
        switch (escape) {
            case 'n': value = "\n"; break;
            case 't': value = "\t"; break;
            case 'r': value = "\r"; break;
            case '\\': value = "\\"; break;
            case '\'': value = "'"; break;
            case '0': value = std::string(1, '\0'); break;
            default: value = std::string(1, escape); break;
        }
        pos++;
        column++;
    } else if (pos < source.length()) {
        value = std::string(1, source[pos]);
        pos++;
        column++;
    }

    if (pos < source.length() && source[pos] == '\'') {
        pos++; // Skip closing quote
        column++;
    }

    return {CTokenType::CharLiteral, value, start_line, start_col};
}

CToken CLexer::readOperator() {
    size_t start_line = line;
    size_t start_col = column;
    char c = nextChar();
    char next = peekChar();

    switch (c) {
        case '+':
            if (next == '+') { nextChar(); return {CTokenType::Increment, "++", start_line, start_col}; }
            if (next == '=') { nextChar(); return {CTokenType::PlusAssign, "+=", start_line, start_col}; }
            return {CTokenType::Plus, "+", start_line, start_col};

        case '-':
            if (next == '-') { nextChar(); return {CTokenType::Decrement, "--", start_line, start_col}; }
            if (next == '=') { nextChar(); return {CTokenType::MinusAssign, "-=", start_line, start_col}; }
            if (next == '>') { nextChar(); return {CTokenType::Arrow, "->", start_line, start_col}; }
            return {CTokenType::Minus, "-", start_line, start_col};

        case '*':
            if (next == '=') { nextChar(); return {CTokenType::StarAssign, "*=", start_line, start_col}; }
            return {CTokenType::Star, "*", start_line, start_col};

        case '/':
            if (next == '=') { nextChar(); return {CTokenType::SlashAssign, "/=", start_line, start_col}; }
            return {CTokenType::Slash, "/", start_line, start_col};

        case '%':
            if (next == '=') { nextChar(); return {CTokenType::PercentAssign, "%=", start_line, start_col}; }
            return {CTokenType::Percent, "%", start_line, start_col};

        case '&':
            if (next == '&') { nextChar(); return {CTokenType::LogicalAnd, "&&", start_line, start_col}; }
            if (next == '=') { nextChar(); return {CTokenType::AmpAssign, "&=", start_line, start_col}; }
            return {CTokenType::Ampersand, "&", start_line, start_col};

        case '|':
            if (next == '|') { nextChar(); return {CTokenType::LogicalOr, "||", start_line, start_col}; }
            if (next == '=') { nextChar(); return {CTokenType::PipeAssign, "|=", start_line, start_col}; }
            return {CTokenType::Pipe, "|", start_line, start_col};

        case '^':
            if (next == '=') { nextChar(); return {CTokenType::CaretAssign, "^=", start_line, start_col}; }
            return {CTokenType::Caret, "^", start_line, start_col};

        case '~':
            return {CTokenType::Tilde, "~", start_line, start_col};

        case '<':
            if (next == '<') {
                nextChar();
                if (peekChar() == '=') { nextChar(); return {CTokenType::LShiftAssign, "<<=", start_line, start_col}; }
                return {CTokenType::LShift, "<<", start_line, start_col};
            }
            if (next == '=') { nextChar(); return {CTokenType::LessEq, "<=", start_line, start_col}; }
            return {CTokenType::Less, "<", start_line, start_col};

        case '>':
            if (next == '>') {
                nextChar();
                if (peekChar() == '=') { nextChar(); return {CTokenType::RShiftAssign, ">>=", start_line, start_col}; }
                return {CTokenType::RShift, ">>", start_line, start_col};
            }
            if (next == '=') { nextChar(); return {CTokenType::GreaterEq, ">=", start_line, start_col}; }
            return {CTokenType::Greater, ">", start_line, start_col};

        case '=':
            if (next == '=') { nextChar(); return {CTokenType::Equal, "==", start_line, start_col}; }
            return {CTokenType::Assign, "=", start_line, start_col};

        case '!':
            if (next == '=') { nextChar(); return {CTokenType::NotEqual, "!=", start_line, start_col}; }
            return {CTokenType::LogicalNot, "!", start_line, start_col};

        case '?':
            return {CTokenType::Question, "?", start_line, start_col};

        case ':':
            return {CTokenType::Colon, ":", start_line, start_col};

        case '.':
            return {CTokenType::Dot, ".", start_line, start_col};

        case '(':
            return {CTokenType::LParen, "(", start_line, start_col};

        case ')':
            return {CTokenType::RParen, ")", start_line, start_col};

        case '[':
            return {CTokenType::LBracket, "[", start_line, start_col};

        case ']':
            return {CTokenType::RBracket, "]", start_line, start_col};

        case '{':
            return {CTokenType::LBrace, "{", start_line, start_col};

        case '}':
            return {CTokenType::RBrace, "}", start_line, start_col};

        case ';':
            return {CTokenType::Semicolon, ";", start_line, start_col};

        case ',':
            return {CTokenType::Comma, ",", start_line, start_col};

        case '#':
            if (next == '#') { nextChar(); return {CTokenType::HashHash, "##", start_line, start_col}; }
            return {CTokenType::Hash, "#", start_line, start_col};

        default:
            return {CTokenType::Invalid, std::string(1, c), start_line, start_col};
    }
}

CToken CLexer::readPreprocessor() {
    size_t start_line = line;
    size_t start_col = column;
    pos++; // Skip #
    column++;

    skipWhitespace();

    size_t directive_start = pos;
    while (pos < source.length() && isAlpha(source[pos])) {
        pos++;
        column++;
    }

    std::string directive = source.substr(directive_start, pos - directive_start);

    // Read rest of line
    size_t args_start = pos;
    while (pos < source.length() && source[pos] != '\n') {
        if (source[pos] == '\\' && pos + 1 < source.length() && source[pos + 1] == '\n') {
            // Line continuation
            pos += 2;
            line++;
            column = 1;
        } else {
            pos++;
            column++;
        }
    }

    std::string full_directive = directive + source.substr(args_start, pos - args_start);
    return {CTokenType::PreprocessorDirective, full_directive, start_line, start_col};
}

bool CLexer::isDigit(char c) const {
    return std::isdigit(c);
}

bool CLexer::isAlpha(char c) const {
    return std::isalpha(c) || c == '_';
}

bool CLexer::isAlnum(char c) const {
    return std::isalnum(c) || c == '_';
}

char CLexer::peekChar(size_t offset) const {
    size_t idx = pos + offset;
    return (idx < source.length()) ? source[idx] : '\0';
}

char CLexer::nextChar() {
    if (pos < source.length()) {
        char c = source[pos++];
        if (c == '\n') {
            line++;
            column = 1;
        } else {
            column++;
        }
        return c;
    }
    return '\0';
}

// CParser implementation (partial - key functions)

CParser::CParser(const std::string& source) : lexer(source) {
    current_module = builder.makeModule("input.c", ir::SourceLang::C);
}

std::shared_ptr<ir::Module> CParser::parse() {
    parseTranslationUnit();
    return current_module;
}

void CParser::parseTranslationUnit() {
    while (!lexer.isEof()) {
        auto token = lexer.peekToken();

        if (token.type == CTokenType::PreprocessorDirective) {
            lexer.advance();
            handlePreprocessorDirective(token.value);
        } else if (token.type == CTokenType::Eof) {
            break;
        } else {
            parseExternalDeclaration();
        }
    }
}

void CParser::parseExternalDeclaration() {
    // This is a simplified implementation
    // A full parser would handle all declaration forms

    auto decl = parseDeclaration();
    if (decl) {
        current_module->declarations.push_back(decl);
    }
}

ir::DeclPtr CParser::parseDeclaration() {
    // Parse declaration specifiers (type, storage class, etc.)
    auto type = parseTypeSpecifier();
    if (!type) {
        error("Expected type specifier");
        return nullptr;
    }

    // Parse declarator (name, pointers, arrays, etc.)
    auto decl = parseDeclarator(type);

    // Check for function definition
    auto token = lexer.peekToken();
    if (token.type == CTokenType::LBrace) {
        // Function definition
        if (decl && decl->kind == ir::DeclKind::Function) {
            auto params = parseParameterList();
            auto func = parseFunction(decl->name, decl->type, params);
            if (func) {
                current_module->functions.push_back(func);
            }
        }
    } else {
        // Variable declaration
        if (token.type == CTokenType::Assign) {
            lexer.advance();
            decl->initializer = parseExpression();
        }
        consume(CTokenType::Semicolon, "Expected ';' after declaration");
    }

    return decl;
}

ir::TypePtr CParser::parseTypeSpecifier() {
    auto token = lexer.peekToken();

    if (token.type == CTokenType::Keyword) {
        if (token.value == "void") {
            lexer.advance();
            return builder.makeVoidType();
        } else if (token.value == "char") {
            lexer.advance();
            return builder.makeIntType(8, true);
        } else if (token.value == "short") {
            lexer.advance();
            return builder.makeIntType(16, true);
        } else if (token.value == "int") {
            lexer.advance();
            return builder.makeIntType(32, true);
        } else if (token.value == "long") {
            lexer.advance();
            // Check for long long
            if (lexer.peekToken().type == CTokenType::Keyword &&
                lexer.peekToken().value == "long") {
                lexer.advance();
                return builder.makeIntType(64, true);
            }
            return builder.makeIntType(64, true);
        } else if (token.value == "float") {
            lexer.advance();
            return builder.makeFloatType(32);
        } else if (token.value == "double") {
            lexer.advance();
            return builder.makeFloatType(64);
        } else if (token.value == "unsigned") {
            lexer.advance();
            // Parse unsigned type
            auto next = lexer.peekToken();
            if (next.type == CTokenType::Keyword) {
                if (next.value == "char") {
                    lexer.advance();
                    return builder.makeIntType(8, false);
                } else if (next.value == "short") {
                    lexer.advance();
                    return builder.makeIntType(16, false);
                } else if (next.value == "int") {
                    lexer.advance();
                    return builder.makeIntType(32, false);
                } else if (next.value == "long") {
                    lexer.advance();
                    if (lexer.peekToken().value == "long") {
                        lexer.advance();
                        return builder.makeIntType(64, false);
                    }
                    return builder.makeIntType(64, false);
                }
            }
            return builder.makeIntType(32, false);  // Default unsigned
        } else if (token.value == "struct") {
            return parseStructOrUnion();
        } else if (token.value == "union") {
            return parseStructOrUnion();
        } else if (token.value == "enum") {
            return parseEnum();
        }
    } else if (token.type == CTokenType::Identifier) {
        // Check if it's a typedef name
        if (isTypedefName(token.value)) {
            return parseTypedefName();
        }
    }

    return nullptr;
}

ir::ExprPtr CParser::parseExpression() {
    return parseAssignmentExpression();
}

ir::ExprPtr CParser::parseAssignmentExpression() {
    auto expr = parseConditionalExpression();

    auto token = lexer.peekToken();
    if (token.type == CTokenType::Assign ||
        token.type == CTokenType::PlusAssign ||
        token.type == CTokenType::MinusAssign ||
        token.type == CTokenType::StarAssign ||
        token.type == CTokenType::SlashAssign) {
        lexer.advance();
        auto right = parseAssignmentExpression();
        return builder.makeBinaryOp(token.value, expr, right);
    }

    return expr;
}

ir::ExprPtr CParser::parseConditionalExpression() {
    auto expr = parseLogicalOrExpression();

    if (lexer.peekToken().type == CTokenType::Question) {
        lexer.advance();
        auto then_expr = parseExpression();
        consume(CTokenType::Colon, "Expected ':' in conditional expression");
        auto else_expr = parseConditionalExpression();

        // Create ternary expression
        auto ternary = std::make_shared<ir::Expression>();
        ternary->kind = ir::ExprKind::TernaryOp;
        ternary->type = then_expr->type;  // Assume same type
        ternary->data = std::make_tuple(expr, then_expr, else_expr);
        return ternary;
    }

    return expr;
}

ir::ExprPtr CParser::parsePrimaryExpression() {
    auto token = lexer.nextToken();

    switch (token.type) {
        case CTokenType::IntLiteral: {
            int64_t val = std::stoll(token.value);
            return builder.makeIntLiteral(val, builder.makeIntType(32, true));
        }

        case CTokenType::FloatLiteral: {
            double val = std::stod(token.value);
            auto expr = std::make_shared<ir::Expression>();
            expr->kind = ir::ExprKind::FloatLiteral;
            expr->type = builder.makeFloatType(64);
            expr->data = val;
            return expr;
        }

        case CTokenType::StringLiteral:
            return builder.makeStringLiteral(token.value);

        case CTokenType::CharLiteral: {
            auto expr = std::make_shared<ir::Expression>();
            expr->kind = ir::ExprKind::CharLiteral;
            expr->type = builder.makeIntType(8, true);
            expr->data = token.value.empty() ? '\0' : token.value[0];
            return expr;
        }

        case CTokenType::Identifier: {
            auto symbol = lookupSymbol(token.value);
            if (symbol) {
                return builder.makeIdentifier(token.value, symbol->type);
            } else {
                // Assume it's a function or will be resolved later
                return builder.makeIdentifier(token.value, nullptr);
            }
        }

        case CTokenType::LParen: {
            auto expr = parseExpression();
            consume(CTokenType::RParen, "Expected ')' after expression");
            return expr;
        }

        default:
            error("Unexpected token in expression: " + token.value);
            return nullptr;
    }
}

// Helper functions

bool CParser::expect(CTokenType type) {
    return lexer.peekToken().type == type;
}

bool CParser::match(CTokenType type) {
    if (expect(type)) {
        lexer.advance();
        return true;
    }
    return false;
}

bool CParser::matchKeyword(const std::string& keyword) {
    auto token = lexer.peekToken();
    if (token.type == CTokenType::Keyword && token.value == keyword) {
        lexer.advance();
        return true;
    }
    return false;
}

void CParser::consume(CTokenType type, const std::string& error_msg) {
    if (!match(type)) {
        error(error_msg);
    }
}

void CParser::error(const std::string& msg) {
    auto token = lexer.peekToken();
    std::stringstream ss;
    ss << "Error at line " << token.line << ", column " << token.column << ": " << msg;
    errors.push_back(ss.str());
}

bool CParser::isTypedefName(const std::string& name) const {
    return std::find(typedefs.begin(), typedefs.end(), name) != typedefs.end();
}

// Stub implementations for remaining functions
ir::DeclPtr CParser::parseDeclarator(ir::TypePtr base_type) {
    auto decl = std::make_shared<ir::Declaration>();
    decl->type = base_type;

    // Check for pointer
    while (match(CTokenType::Star)) {
        decl->type = builder.makePointerType(decl->type);
    }

    // Get name
    auto token = lexer.peekToken();
    if (token.type == CTokenType::Identifier) {
        decl->name = token.value;
        lexer.advance();
    }

    // Check for array
    while (match(CTokenType::LBracket)) {
        // Simple constant size array for now
        auto size_token = lexer.nextToken();
        size_t size = 0;
        if (size_token.type == CTokenType::IntLiteral) {
            size = std::stoull(size_token.value);
        }
        consume(CTokenType::RBracket, "Expected ']' after array size");
        decl->type = builder.makeArrayType(decl->type, size);
    }

    // Check for function
    if (match(CTokenType::LParen)) {
        decl->kind = ir::DeclKind::Function;
        // Parameters would be parsed here
        consume(CTokenType::RParen, "Expected ')' after parameters");
    } else {
        decl->kind = ir::DeclKind::Variable;
    }

    return decl;
}

std::vector<std::pair<std::string, ir::TypePtr>> CParser::parseParameterList() {
    std::vector<std::pair<std::string, ir::TypePtr>> params;
    // Simplified - would parse full parameter list
    return params;
}

ir::FuncPtr CParser::parseFunction(const std::string& name, ir::TypePtr return_type,
                                  std::vector<std::pair<std::string, ir::TypePtr>> params) {
    consume(CTokenType::LBrace, "Expected '{' at start of function body");

    std::vector<ir::StmtPtr> body;
    while (!expect(CTokenType::RBrace)) {
        auto stmt = parseStatement();
        if (stmt) {
            body.push_back(stmt);
        }
    }

    consume(CTokenType::RBrace, "Expected '}' at end of function body");

    return builder.makeFunction(name, return_type, params, body);
}

ir::StmtPtr CParser::parseStatement() {
    auto token = lexer.peekToken();

    if (token.type == CTokenType::Keyword) {
        if (token.value == "if") return parseSelectionStatement();
        if (token.value == "while" || token.value == "do" || token.value == "for")
            return parseIterationStatement();
        if (token.value == "return" || token.value == "break" ||
            token.value == "continue" || token.value == "goto")
            return parseJumpStatement();
    }

    if (token.type == CTokenType::LBrace) {
        return parseCompoundStatement();
    }

    // Default to expression statement
    return parseExpressionStatement();
}

ir::StmtPtr CParser::parseCompoundStatement() {
    consume(CTokenType::LBrace, "Expected '{'");

    std::vector<ir::StmtPtr> stmts;
    while (!expect(CTokenType::RBrace)) {
        auto stmt = parseStatement();
        if (stmt) {
            stmts.push_back(stmt);
        }
    }

    consume(CTokenType::RBrace, "Expected '}'");
    return builder.makeBlock(stmts);
}

ir::StmtPtr CParser::parseExpressionStatement() {
    if (match(CTokenType::Semicolon)) {
        // Empty statement
        return builder.makeExprStmt(nullptr);
    }

    auto expr = parseExpression();
    consume(CTokenType::Semicolon, "Expected ';' after expression");
    return builder.makeExprStmt(expr);
}

ir::StmtPtr CParser::parseSelectionStatement() {
    consumeKeyword("if", "Expected 'if'");
    consume(CTokenType::LParen, "Expected '(' after 'if'");
    auto cond = parseExpression();
    consume(CTokenType::RParen, "Expected ')' after condition");

    auto then_stmt = parseStatement();
    ir::StmtPtr else_stmt = nullptr;

    if (matchKeyword("else")) {
        else_stmt = parseStatement();
    }

    return builder.makeIf(cond, then_stmt, else_stmt);
}

ir::StmtPtr CParser::parseIterationStatement() {
    if (matchKeyword("while")) {
        consume(CTokenType::LParen, "Expected '(' after 'while'");
        auto cond = parseExpression();
        consume(CTokenType::RParen, "Expected ')' after condition");
        auto body = parseStatement();
        return builder.makeWhile(cond, body);
    }

    // Handle do-while and for loops similarly
    return nullptr;
}

ir::StmtPtr CParser::parseJumpStatement() {
    if (matchKeyword("return")) {
        ir::ExprPtr expr = nullptr;
        if (!expect(CTokenType::Semicolon)) {
            expr = parseExpression();
        }
        consume(CTokenType::Semicolon, "Expected ';' after return");
        return builder.makeReturn(expr);
    }

    // Handle break, continue, goto
    return nullptr;
}

// Remaining expression parsing functions follow similar pattern
ir::ExprPtr CParser::parseLogicalOrExpression() {
    auto left = parseLogicalAndExpression();
    while (match(CTokenType::LogicalOr)) {
        auto right = parseLogicalAndExpression();
        left = builder.makeBinaryOp("||", left, right);
    }
    return left;
}

ir::ExprPtr CParser::parseLogicalAndExpression() {
    auto left = parseBitwiseOrExpression();
    while (match(CTokenType::LogicalAnd)) {
        auto right = parseBitwiseOrExpression();
        left = builder.makeBinaryOp("&&", left, right);
    }
    return left;
}

// Stub remaining expression functions
ir::ExprPtr CParser::parseBitwiseOrExpression() { return parseBitwiseXorExpression(); }
ir::ExprPtr CParser::parseBitwiseXorExpression() { return parseBitwiseAndExpression(); }
ir::ExprPtr CParser::parseBitwiseAndExpression() { return parseEqualityExpression(); }
ir::ExprPtr CParser::parseEqualityExpression() { return parseRelationalExpression(); }
ir::ExprPtr CParser::parseRelationalExpression() { return parseShiftExpression(); }
ir::ExprPtr CParser::parseShiftExpression() { return parseAdditiveExpression(); }
ir::ExprPtr CParser::parseAdditiveExpression() { return parseMultiplicativeExpression(); }
ir::ExprPtr CParser::parseMultiplicativeExpression() { return parseCastExpression(); }
ir::ExprPtr CParser::parseCastExpression() { return parseUnaryExpression(); }
ir::ExprPtr CParser::parseUnaryExpression() { return parsePostfixExpression(); }
ir::ExprPtr CParser::parsePostfixExpression() { return parsePrimaryExpression(); }

// Stub remaining type functions
ir::TypePtr CParser::parseStructOrUnion() { return nullptr; }
ir::TypePtr CParser::parseEnum() { return nullptr; }
ir::TypePtr CParser::parseTypedefName() { return nullptr; }

// Stub remaining helper functions
void CParser::handlePreprocessorDirective(const std::string& directive) {}
void CParser::consumeKeyword(const std::string& keyword, const std::string& error_msg) {
    if (!matchKeyword(keyword)) {
        error(error_msg);
    }
}
void CParser::addSymbol(const std::string& name, ir::TypePtr type, bool is_function) {}
CParser::SymbolInfo* CParser::lookupSymbol(const std::string& name) { return nullptr; }

} // namespace parsers
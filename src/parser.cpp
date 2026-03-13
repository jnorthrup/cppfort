//===----------------------------------------------------------------------===//
// src/parser.cpp - Hand-Written Parser Implementation
// TrikeShed Math-Based SoN Compiler
// 100% Hand-Written Parser - No LLM-generated internals
//===----------------------------------------------------------------------===//

#include "cppfort_parser.h"
#include <cctype>
#include <charconv>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <algorithm>

namespace cppfort {

// ============================================================================
// Parser Implementation
// ============================================================================

Parser::Parser() = default;
Parser::~Parser() = default;

// ---------------------------------------------------------------------------
// Lexer Implementation
// ---------------------------------------------------------------------------

void Parser::skip_whitespace() {
    while (position_ < source_.size()) {
        char c = source_[position_];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (c == '\n') {
                current_line_++;
                current_column_ = 1;
            } else {
                current_column_++;
            }
            position_++;
        } else if (c == '/' && position_ + 1 < source_.size()) {
            if (source_[position_ + 1] == '/') {
                while (position_ < source_.size() && source_[position_] != '\n') {
                    position_++;
                }
            } else if (source_[position_ + 1] == '*') {
                position_ += 2;
                while (position_ + 1 < source_.size()) {
                    if (source_[position_] == '*' && source_[position_ + 1] == '/') {
                        position_ += 2;
                        break;
                    }
                    if (source_[position_] == '\n') {
                        current_line_++;
                        current_column_ = 1;
                    }
                    position_++;
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }
}

bool Parser::match(char c) {
    skip_whitespace();
    if (position_ < source_.size() && source_[position_] == c) {
        position_++;
        current_column_++;
        return true;
    }
    return false;
}

bool Parser::match(std::string_view str) {
    skip_whitespace();
    if (position_ + str.size() <= source_.size() &&
        source_.substr(position_, str.size()) == str) {
        position_ += str.size();
        current_column_ += str.size();
        return true;
    }
    return false;
}

char Parser::peek() {
    skip_whitespace();
    if (position_ < source_.size()) {
        return source_[position_];
    }
    return '\0';
}

char Parser::consume() {
    skip_whitespace();
    if (position_ < source_.size()) {
        char c = source_[position_++];
        current_column_++;
        return c;
    }
    return '\0';
}

Parser::Token Parser::next_token() {
    skip_whitespace();

    Token tok;
    tok.line = current_line_;
    tok.column = current_column_;

    if (position_ >= source_.size()) {
        tok.type = TokenType::EndOfFile;
        tok.text = "";
        return tok;
    }

    char c = source_[position_];

    // Multi-character operators first
    if (position_ + 1 < source_.size()) {
        std::string_view two = source_.substr(position_, 2);
        if (two == "->" || two == "::" || two == "==" || two == "!=" ||
            two == "<=" || two == ">=" || two == "++" || two == "--" ||
            two == "&&" || two == "||" || two == "<<" || two == ">>" ||
            two == "+=" || two == "-=" || two == "*=" || two == "/=" ||
            two == "%=" || two == "&=" || two == "|=" || two == "^=" ||
            two == "..") {
            tok.type = TokenType::Operator;
            tok.text = two;
            position_ += 2;
            current_column_ += 2;
            return tok;
        }
    }

    if (position_ + 2 < source_.size()) {
        std::string_view three = source_.substr(position_, 3);
        if (three == "..." || three == "<=>" || three == "..=" || three == "..<") {
            tok.type = TokenType::Operator;
            tok.text = three;
            position_ += 3;
            current_column_ += 3;
            return tok;
        }
    }

    // Single-character punctuation
    switch (c) {
        case '[': tok.type = TokenType::LBracket; break;
        case ']': tok.type = TokenType::RBracket; break;
        case '(': tok.type = TokenType::LParen; break;
        case ')': tok.type = TokenType::RParen; break;
        case '{': tok.type = TokenType::LBrace; break;
        case '}': tok.type = TokenType::RBrace; break;
        case ',': tok.type = TokenType::Comma; break;
        case '.': tok.type = TokenType::Dot; break;
        case ':': tok.type = TokenType::Colon; break;
        default: tok.type = TokenType::EndOfFile; break;
    }

    if (tok.type != TokenType::EndOfFile && tok.type != TokenType::Operator) {
        tok.text = source_.substr(position_, 1);
        position_++;
        current_column_++;
        return tok;
    }

    // Identifiers and keywords
    if (std::isalpha(c) || c == '_') {
        size_t start = position_;
        while (position_ < source_.size() &&
               (std::isalnum(source_[position_]) || source_[position_] == '_')) {
            position_++;
        }
        tok.type = TokenType::Identifier;
        tok.text = source_.substr(start, position_ - start);
        current_column_ += tok.text.size();
        return tok;
    }

    // Numbers
    if (std::isdigit(c) || (c == '.' && position_ + 1 < source_.size() && std::isdigit(source_[position_ + 1]))) {
        size_t start = position_;
        bool has_dot = (c == '.');
        position_++;
        while (position_ < source_.size()) {
            if (std::isdigit(source_[position_])) {
                position_++;
            } else if (source_[position_] == '.' && !has_dot) {
                has_dot = true;
                position_++;
            } else if (source_[position_] == 'e' || source_[position_] == 'E') {
                position_++;
                if (position_ < source_.size() && (source_[position_] == '+' || source_[position_] == '-')) {
                    position_++;
                }
            } else {
                break;
            }
        }
        tok.type = TokenType::Number;
        tok.text = source_.substr(start, position_ - start);
        current_column_ += tok.text.size();
        return tok;
    }

    // Strings
    if (c == '"') {
        size_t start = position_ + 1;
        position_++;
        while (position_ < source_.size() && source_[position_] != '"') {
            if (source_[position_] == '\\' && position_ + 1 < source_.size()) {
                position_++;
            }
            position_++;
        }
        if (position_ < source_.size()) position_++;
        tok.type = TokenType::String;
        tok.text = source_.substr(start, position_ - start - 1);
        current_column_ += tok.text.size() + 2;
        return tok;
    }

    // Single-character operators
    tok.type = TokenType::Operator;
    tok.text = source_.substr(position_, 1);
    position_++;
    current_column_++;
    return tok;
}

// ============================================================================
// Keywords Set
// ============================================================================

static const std::unordered_set<std::string> keywords = {
    "as", "assert", "auto", "await", "base", "break", "case",
    "catch", "channel", "concept", "const", "constexpr",
    "continue", "copy", "coroutineScope", "decltype", "default",
    "do", "else", "enum", "explicit", "false", "final", "for",
    "forward", "forward_ref", "func", "if", "implicit", "import",
    "in", "in_ref", "inout", "inspect", "interface", "is",
    "launch", "let", "move", "namespace", "next", "noexcept",
    "operator", "out", "override", "parallel_for", "post", "pre",
    "private", "protected", "public", "requires", "return",
    "select", "static_assert", "struct", "switch", "template",
    "that", "this", "throw", "throws", "true", "try", "type",
    "union", "using", "virtual", "while",
    // TrikeShed-specific
    "chart", "atlas", "manifold", "coords", "project", "embed", "locate", "transition", "lowered"
};

static bool is_keyword(std::string_view text) {
    return keywords.find(std::string(text)) != keywords.end();
}

// ============================================================================
// Parsing Implementation - 15 Precedence Levels
// ============================================================================

ParseResult Parser::parse(std::string_view source) {
    source_ = source;
    position_ = 0;
    current_line_ = 1;
    current_column_ = 1;
    clear_errors();

    auto ast = std::make_unique<CanonicalAST>();

    std::vector<std::unique_ptr<ASTNode>> statements;
    while (peek() != '\0') {
        auto stmt = parse_statement();
        if (stmt) {
            statements.push_back(std::move(stmt));
        } else {
            skip_whitespace();
            if (peek() != '\0') {
                errors_.push_back(ParseError(current_line_, current_column_, 
                    "Unexpected token at line " + std::to_string(current_line_)));
                consume();
            }
            break;
        }
    }

    auto root = std::make_unique<ASTNode>(NodeTag::canonical_ast_tag, "program");
    for (auto& stmt : statements) {
        root->children.push_back(std::move(stmt));
    }

    ast->set_root(std::move(root));

    ParseResult result;
    result.ast = std::move(ast);
    result.errors = std::move(errors_);
    return result;
}

ParseResult Parser::parse_file(std::string_view path) {
    std::ifstream file((std::string(path)));
    if (!file.is_open()) {
        ParseResult result;
        result.errors.push_back(ParseError(0, 0, "Could not open file: " + std::string(path)));
        return result;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    auto result = parse(content);
    if (result.ast) {
        result.ast->add_dependency(path);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Statement Parsing
// ---------------------------------------------------------------------------

std::unique_ptr<ASTNode> Parser::parse_statement() {
    skip_whitespace();
    if (peek() == '\0') {
        return nullptr;
    }

// These handle their own keyword matching
  if (auto chart_node = parse_chart()) {
    return chart_node;
  }

  if (auto atlas_node = parse_atlas()) {
    return atlas_node;
  }

  if (auto manifold_node = parse_manifold()) {
    return manifold_node;
  }

  // parse_coordinates handles its own "coords" keyword matching
  if (auto coords = parse_coordinates()) {
    return coords;
  }

  // Check for return statement
  if (match("return")) {
    auto ret = std::make_unique<ASTNode>(NodeTag::indexed_tag, "return");
    skip_whitespace();
    auto expr = parse_expression();
    if (expr) {
        ret->children.push_back(std::move(expr));
    }
    match(';');
    return ret;
  }

    // Check for unified declaration: identifier followed by colon
    if (std::isalpha(peek()) || peek() == '_') {
        size_t id_start = position_;
        while (position_ < source_.size() &&
               (std::isalnum(source_[position_]) || source_[position_] == '_')) {
            position_++;
        }
        std::string_view ident = source_.substr(id_start, position_ - id_start);
        
        skip_whitespace();
        if (peek() == ':') {
            consume(); // consume the colon
            return parse_unified_declaration(ident);
        }
        // Reset position if not colon
        position_ = id_start;
    }

  if (match("let") || match("const")) {
        return parse_declaration();
    }

    return parse_expression();
}

// ---------------------------------------------------------------------------
// Declaration Parsing
// ---------------------------------------------------------------------------

std::unique_ptr<ASTNode> Parser::parse_declaration() {
    auto decl = std::make_unique<ASTNode>(NodeTag::canonical_ast_tag, "decl");
    decl->line = current_line_;
    decl->column = current_column_;

    skip_whitespace();
    if (std::isalpha(peek()) || peek() == '_') {
        size_t start = position_;
        while (position_ < source_.size() &&
               (std::isalnum(source_[position_]) || source_[position_] == '_')) {
            position_++;
        }
        auto name = source_.substr(start, position_ - start);
        auto name_node = std::make_unique<ASTNode>(NodeTag::indexed_tag, name);
        name_node->line = current_line_;
        name_node->column = current_column_;
        decl->children.push_back(std::move(name_node));
    }

    skip_whitespace();
    if (match(':')) {
        auto type = parse_type_specifier();
        if (type) {
            decl->children.push_back(std::move(type));
        }
    }

    skip_whitespace();
    if (match('=') || match("==")) {
        auto init = parse_expression();
        if (init) {
            decl->children.push_back(std::move(init));
        }
    }

    match(';');
    return decl;
}

std::unique_ptr<ASTNode> Parser::parse_type_specifier() {
    skip_whitespace();
    auto expr = parse_expression();
    return expr;
}

// ---------------------------------------------------------------------------
// Unified Declaration Parsing (main: () -> int = { ... })
// ---------------------------------------------------------------------------

std::unique_ptr<ASTNode> Parser::parse_unified_declaration(std::string_view name) {
    auto decl = std::make_unique<ASTNode>(NodeTag::canonical_ast_tag, std::string(name));
    decl->line = current_line_;
    decl->column = current_column_;

    // Parse param list: () OR type specifier
    skip_whitespace();
    if (match('(')) {
        // Function with params
        auto params = std::make_unique<ASTNode>(NodeTag::indexed_tag, "params");
        skip_whitespace();
        if (!match(')')) {
            // Parse parameters
            while (true) {
                skip_whitespace();
                if (peek() == ')') {
                    consume();
                    break;
                }
                auto param = std::make_unique<ASTNode>(NodeTag::indexed_tag, "param");
                skip_whitespace();
                if (std::isalpha(peek()) || peek() == '_') {
                    size_t pstart = position_;
                    while (position_ < source_.size() &&
                           (std::isalnum(source_[position_]) || source_[position_] == '_')) {
                        position_++;
                    }
                    param->children.push_back(std::make_unique<ASTNode>(NodeTag::indexed_tag, 
                        source_.substr(pstart, position_ - pstart)));
                }
                skip_whitespace();
                if (match(',')) {
                    params->children.push_back(std::move(param));
                } else if (peek() == ')') {
                    consume();
                    params->children.push_back(std::move(param));
                    break;
                } else {
                    params->children.push_back(std::move(param));
                    break;
                }
            }
        }
        decl->children.push_back(std::move(params));
        
        // Function return type: -> int
        skip_whitespace();
        if (match("->")) {
            skip_whitespace();
            size_t rstart = position_;
            while (position_ < source_.size() &&
                   (std::isalpha(source_[position_]) || source_[position_] == '_')) {
                position_++;
            }
            if (position_ > rstart) {
                auto rettype = std::make_unique<ASTNode>(NodeTag::coordinates_tag,
                    source_.substr(rstart, position_ - rstart));
                decl->children.push_back(std::move(rettype));
            }
        }
    } else {
        // Variable: type specifier only (no params)
        // Parse type: int, string, etc.
        size_t tstart = position_;
        while (position_ < source_.size() &&
               (std::isalnum(source_[position_]) || source_[position_] == '_')) {
            position_++;
        }
        if (position_ > tstart) {
            auto type = std::make_unique<ASTNode>(NodeTag::coordinates_tag,
                source_.substr(tstart, position_ - tstart));
            decl->children.push_back(std::move(type));
        }
    }

    // Parse body/initializer: = { ... } or = expression or just ;
    skip_whitespace();
    if (match('=')) {
        skip_whitespace();
        if (match('{')) {
            auto block = std::make_unique<ASTNode>(NodeTag::indexed_tag, "block");
            // Parse statements in block
            while (true) {
                skip_whitespace();
                if (peek() == '}') {
                    consume();
                    break;
                }
                auto stmt = parse_statement();
                if (stmt) {
                    block->children.push_back(std::move(stmt));
                } else {
                    break;
                }
            }
            decl->children.push_back(std::move(block));
        } else {
            auto expr = parse_expression();
            if (expr) {
                decl->children.push_back(std::move(expr));
            }
        }
    }

    match(';');
    return decl;
}

// ---------------------------------------------------------------------------
// Expression Parsing - 15 Precedence Levels
// Level 15: Assignment (right associative)
// ---------------------------------------------------------------------------

std::unique_ptr<ASTNode> Parser::parse_expression() {
    return parse_assignment_expression();
}

// Level 15: Assignment
std::unique_ptr<ASTNode> Parser::parse_assignment_expression() {
    auto left = parse_pipeline_expression();
    if (!left) return nullptr;

    skip_whitespace();
    std::string_view op;
    if (match('=')) op = "=";
    else if (match("+=")) op = "+=";
    else if (match("-=")) op = "-=";
    else if (match("*=")) op = "*=";
    else if (match("/=")) op = "/=";
    else if (match("%=")) op = "%=";
    else if (match("<<=")) op = "<<=";
    else if (match(">>=")) op = ">>=";
    else if (match("&=")) op = "&=";
    else if (match("|=")) op = "|=";
    else if (match("^=")) op = "^=";
    else return left;

    auto right = parse_assignment_expression();
    if (!right) return left;

    auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, std::string(op));
    node->line = current_line_;
    node->column = current_column_;
    node->children.push_back(std::move(left));
    node->children.push_back(std::move(right));
    return node;
}

// Level 14: Pipeline
std::unique_ptr<ASTNode> Parser::parse_pipeline_expression() {
    auto left = parse_ternary_expression();
    if (!left) return nullptr;

    skip_whitespace();
    while (match("|>")) {
        auto right = parse_ternary_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, "pipe");
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 13: Ternary
std::unique_ptr<ASTNode> Parser::parse_ternary_expression() {
    auto condition = parse_logical_or_expression();
    if (!condition) return nullptr;

    skip_whitespace();
    if (!match('?')) return condition;

    auto then_expr = parse_expression();
    if (!then_expr) return condition;

    skip_whitespace();
    if (!match(':')) return condition;

    auto else_expr = parse_ternary_expression();
    if (!else_expr) return condition;

    auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, "?:");
    node->line = current_line_;
    node->column = current_column_;
    node->children.push_back(std::move(condition));
    node->children.push_back(std::move(then_expr));
    node->children.push_back(std::move(else_expr));
    return node;
}

// Level 12: Logical OR
std::unique_ptr<ASTNode> Parser::parse_logical_or_expression() {
    auto left = parse_logical_and_expression();
    if (!left) return nullptr;

    skip_whitespace();
    while (match("||")) {
        auto right = parse_logical_and_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, "||");
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 11: Logical AND
std::unique_ptr<ASTNode> Parser::parse_logical_and_expression() {
    auto left = parse_bitwise_or_expression();
    if (!left) return nullptr;

    skip_whitespace();
    while (match("&&")) {
        auto right = parse_bitwise_or_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, "&&");
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 10: Bitwise OR
std::unique_ptr<ASTNode> Parser::parse_bitwise_or_expression() {
    auto left = parse_bitwise_xor_expression();
    if (!left) return nullptr;

    skip_whitespace();
    while (match('|')) {
        auto right = parse_bitwise_xor_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, "|");
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 9: Bitwise XOR
std::unique_ptr<ASTNode> Parser::parse_bitwise_xor_expression() {
    auto left = parse_bitwise_and_expression();
    if (!left) return nullptr;

    skip_whitespace();
    while (match('^')) {
        auto right = parse_bitwise_and_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, "^");
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 8: Bitwise AND
std::unique_ptr<ASTNode> Parser::parse_bitwise_and_expression() {
    auto left = parse_equality_expression();
    if (!left) return nullptr;

    skip_whitespace();
    while (match('&')) {
        auto right = parse_equality_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, "&");
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 7: Equality
std::unique_ptr<ASTNode> Parser::parse_equality_expression() {
    auto left = parse_comparison_expression();
    if (!left) return nullptr;

    skip_whitespace();
    std::string_view op;
    while (match("==") || match("!=")) {
        if (source_.substr(position_ - 2, 2) == "==") op = "==";
        else op = "!=";
        
        auto right = parse_comparison_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, std::string(op));
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 6: Comparison
std::unique_ptr<ASTNode> Parser::parse_comparison_expression() {
    auto left = parse_range_expression();
    if (!left) return nullptr;

    skip_whitespace();
    std::string_view op;
    while (match('<') || match('>') || match("<=") || match(">=") || match("<=>")) {
        if (source_.substr(position_ - 1, 1) == "<") {
            if (position_ > 1 && source_[position_ - 2] == '<') op = "<<";
            else if (peek() == '=') { consume(); op = "<="; }
            else if (peek() == '>') { consume(); op = "<>"; }
            else op = "<";
        } else if (source_.substr(position_ - 1, 1) == ">") {
            if (position_ > 1 && source_[position_ - 2] == '>') op = ">>";
            else if (peek() == '=') { consume(); op = ">="; }
            else op = ">";
        } else if (match("<=>")) {
            op = "<=>";
        } else {
            break;
        }

        auto right = parse_range_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, std::string(op));
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 5: Range
std::unique_ptr<ASTNode> Parser::parse_range_expression() {
    auto left = parse_shift_expression();
    if (!left) return nullptr;

    skip_whitespace();
    while (match("..=") || match("..<") || match("..")) {
        std::string_view op;
        if (source_.substr(position_ - 2, 2) == "..=") op = "..=";
        else if (source_.substr(position_ - 2, 2) == "..<") op = "..<";
        else op = "..";

        auto right = parse_shift_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, std::string(op));
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 4: Shift
std::unique_ptr<ASTNode> Parser::parse_shift_expression() {
    auto left = parse_additive_expression();
    if (!left) return nullptr;

    skip_whitespace();
    while (match("<<") || match(">>")) {
        std::string_view op = source_.substr(position_ - 2, 2);
        auto right = parse_additive_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, std::string(op));
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 3: Additive
std::unique_ptr<ASTNode> Parser::parse_additive_expression() {
    auto left = parse_multiplicative_expression();
    if (!left) return nullptr;

    skip_whitespace();
    while (match('+') || match('-')) {
        std::string_view op = source_.substr(position_ - 1, 1);
        auto right = parse_multiplicative_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, std::string(op));
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 2: Multiplicative
std::unique_ptr<ASTNode> Parser::parse_multiplicative_expression() {
    auto left = parse_prefix_expression();
    if (!left) return nullptr;

    skip_whitespace();
    while (match('*') || match('/') || match('%')) {
        std::string_view op = source_.substr(position_ - 1, 1);
        auto right = parse_prefix_expression();
        if (!right) return left;

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, std::string(op));
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(left));
        node->children.push_back(std::move(right));
        left = std::move(node);
    }

    return left;
}

// Level 1: Prefix (right associative)
std::unique_ptr<ASTNode> Parser::parse_prefix_expression() {
    skip_whitespace();

    // Unary prefix operators
    if (match("await")) {
        auto operand = parse_prefix_expression();
        if (!operand) return nullptr;
        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, "await");
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(operand));
        return node;
    }

    if (match("launch")) {
        auto operand = parse_prefix_expression();
        if (!operand) return nullptr;
        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, "launch");
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(operand));
        return node;
    }

    if (match("select")) {
        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, "select");
        node->line = current_line_;
        node->column = current_column_;
        if (match('{')) {
            while (!match('}')) {
                auto expr = parse_expression();
                if (expr) node->children.push_back(std::move(expr));
            }
        }
        return node;
    }

    if (match("move") || match("forward") || match("copy")) {
        auto op = source_.substr(position_ - 4, 4);
        if (source_[position_ - 4] == 'm') op = "move";
        else if (source_[position_ - 4] == 'f') op = "forward";
        else op = "copy";
        
        auto operand = parse_prefix_expression();
        if (!operand) return nullptr;
        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, std::string(op));
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(operand));
        return node;
    }

    // Unary operators: + - ! ~ ++ -- & *
    char prefix_op = '\0';
    if (match('+')) prefix_op = '+';
    else if (match('-')) prefix_op = '-';
    else if (match('!')) prefix_op = '!';
    else if (match('~')) prefix_op = '~';
    else if (match("++")) prefix_op = '+';
    else if (match("--")) prefix_op = '-';
    else if (match('&')) prefix_op = '&';
    else if (match('*')) prefix_op = '*';

    if (prefix_op) {
        auto operand = parse_prefix_expression();
        if (!operand) return nullptr;
        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, std::string(1, prefix_op));
        node->line = current_line_;
        node->column = current_column_;
        node->children.push_back(std::move(operand));
        return node;
    }

    return parse_postfix_expression();
}

// Level 0: Postfix (left associative)
std::unique_ptr<ASTNode> Parser::parse_postfix_expression() {
    auto expr = parse_primary_expression();
    if (!expr) return nullptr;

    while (true) {
        skip_whitespace();

        // Method call: .method(args)
        if (match('.')) {
            skip_whitespace();
            size_t start = position_;
            while (position_ < source_.size() &&
                   (std::isalnum(source_[position_]) || source_[position_] == '_')) {
                position_++;
            }
            std::string_view method = source_.substr(start, position_ - start);

            if (method == "project") {
                if (match('(')) {
                    auto args = parse_argument_list();
                    match(')');
                    auto node = std::make_unique<ASTNode>(NodeTag::chart_project_tag, "project");
                    node->line = current_line_;
                    node->column = current_column_;
                    node->children.push_back(std::move(expr));
                    for (auto& arg : args) {
                        node->children.push_back(std::move(arg));
                    }
                    expr = std::move(node);
                    continue;
                }
            } else if (method == "embed") {
                if (match('(')) {
                    auto args = parse_argument_list();
                    match(')');
                    auto node = std::make_unique<ASTNode>(NodeTag::chart_embed_tag, "embed");
                    node->line = current_line_;
                    node->column = current_column_;
                    node->children.push_back(std::move(expr));
                    for (auto& arg : args) {
                        node->children.push_back(std::move(arg));
                    }
                    expr = std::move(node);
                    continue;
                }
            } else if (method == "locate") {
                if (match('(')) {
                    auto args = parse_argument_list();
                    match(')');
                    auto node = std::make_unique<ASTNode>(NodeTag::atlas_locate_tag, "locate");
                    node->line = current_line_;
                    node->column = current_column_;
                    node->children.push_back(std::move(expr));
                    for (auto& arg : args) {
                        node->children.push_back(std::move(arg));
                    }
                    expr = std::move(node);
                    continue;
                }
            } else if (method == "lowered") {
                if (match('(')) {
                    match(')');
                    auto node = std::make_unique<ASTNode>(NodeTag::lower_dense_tag, "lowered");
                    node->line = current_line_;
                    node->column = current_column_;
                    node->children.push_back(std::move(expr));
                    expr = std::move(node);
                    continue;
                }
            } else if (method == "transition") {
                if (match('(')) {
                    auto args = parse_argument_list();
                    match(')');
                    auto node = std::make_unique<ASTNode>(NodeTag::transition_tag, "transition");
                    node->line = current_line_;
                    node->column = current_column_;
                    node->children.push_back(std::move(expr));
                    for (auto& arg : args) {
                        node->children.push_back(std::move(arg));
                    }
                    expr = std::move(node);
                    continue;
                }
            }

            // Generic member access
            auto member = std::make_unique<ASTNode>(NodeTag::indexed_tag, std::string(method));
            member->line = current_line_;
            member->column = current_column_;

            if (match('(')) {
                auto args = parse_argument_list();
                match(')');
                auto call = std::make_unique<ASTNode>(NodeTag::indexed_tag, "call");
                call->line = current_line_;
                call->column = current_column_;
                call->children.push_back(std::move(expr));
                call->children.push_back(std::move(member));
                for (auto& arg : args) {
                    call->children.push_back(std::move(arg));
                }
                expr = std::move(call);
                continue;
            }

            auto access = std::make_unique<ASTNode>(NodeTag::indexed_tag, ".");
            access->line = current_line_;
            access->column = current_column_;
            access->children.push_back(std::move(expr));
            access->children.push_back(std::move(member));
            expr = std::move(access);
            continue;
        }

        // Subscript: [expr]
        if (match('[')) {
            auto args = parse_argument_list();
            match(']');
            auto subscript = std::make_unique<ASTNode>(NodeTag::indexed_tag, "[]");
            subscript->line = current_line_;
            subscript->column = current_column_;
            subscript->children.push_back(std::move(expr));
            for (auto& arg : args) {
                subscript->children.push_back(std::move(arg));
            }
            expr = std::move(subscript);
            continue;
        }

        // Function call: (args)
        if (match('(')) {
            auto args = parse_argument_list();
            match(')');
            auto call = std::make_unique<ASTNode>(NodeTag::indexed_tag, "call");
            call->line = current_line_;
            call->column = current_column_;
            call->children.push_back(std::move(expr));
            for (auto& arg : args) {
                call->children.push_back(std::move(arg));
            }
            expr = std::move(call);
            continue;
        }

        // Postfix increment/decrement
        if (match("++") || match("--")) {
            std::string_view op = source_.substr(position_ - 2, 2);
            auto postfix = std::make_unique<ASTNode>(NodeTag::indexed_tag, std::string(op));
            postfix->line = current_line_;
            postfix->column = current_column_;
            postfix->children.push_back(std::move(expr));
            expr = std::move(postfix);
            continue;
        }

        break;
    }

    return expr;
}

std::vector<std::unique_ptr<ASTNode>> Parser::parse_argument_list() {
    std::vector<std::unique_ptr<ASTNode>> args;
    while (true) {
        if (match(')')) {
            position_--;
            current_column_--;
            break;
        }
        auto arg = parse_expression();
        if (arg) {
            args.push_back(std::move(arg));
        }
        if (!match(',')) break;
    }
    return args;
}

// Primary expression
std::unique_ptr<ASTNode> Parser::parse_primary_expression() {
    skip_whitespace();
    char c = peek();

    // Number literal
    if (std::isdigit(c) || c == '.') {
        size_t start = position_;
        bool has_dot = (c == '.');
        position_++;
        while (position_ < source_.size()) {
            if (std::isdigit(source_[position_])) {
                position_++;
            } else if (source_[position_] == '.' && !has_dot) {
                has_dot = true;
                position_++;
            } else if (source_[position_] == 'e' || source_[position_] == 'E') {
                position_++;
                if (position_ < source_.size() && (source_[position_] == '+' || source_[position_] == '-')) {
                    position_++;
                }
            } else {
                break;
            }
        }
        auto node = std::make_unique<ASTNode>(NodeTag::coordinates_tag,
            source_.substr(start, position_ - start));
        node->line = current_line_;
        node->column = current_column_;
        return node;
    }

    // String literal
    if (c == '"') {
        position_++;
        size_t start = position_;
        while (position_ < source_.size() && source_[position_] != '"') {
            if (source_[position_] == '\\' && position_ + 1 < source_.size()) {
                position_++;
            }
            position_++;
        }
        if (position_ < source_.size()) position_++;
        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag,
            source_.substr(start, position_ - start - 1));
        node->line = current_line_;
        node->column = current_column_;
        return node;
    }

    // Identifier or keyword
    if (std::isalpha(c) || c == '_') {
        size_t start = position_;
        while (position_ < source_.size() &&
               (std::isalnum(source_[position_]) || source_[position_] == '_')) {
            position_++;
        }
        std::string_view ident = source_.substr(start, position_ - start);

        // Check for coords[...]
        if (ident == "coords") {
            return parse_coordinates();
        }

        // Check for chart declaration
        if (ident == "chart") {
            return parse_chart();
        }

        // Check for atlas literal
        if (ident == "atlas") {
            return parse_atlas();
        }

        // Check for manifold
        if (ident == "manifold") {
            return parse_manifold();
        }

        // Check for literals
        if (ident == "true" || ident == "false") {
            auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, ident);
            node->line = current_line_;
            node->column = current_column_;
            return node;
        }

        // Check for this/that
        if (ident == "this" || ident == "that") {
            auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, ident);
            node->line = current_line_;
            node->column = current_column_;
            return node;
        }

        // Check for α (alpha) transform
        if (ident == "alpha" || ident == "α") {
            return parse_alpha();
        }

        // Check for join operator
        if (ident == "join") {
            return parse_join();
        }

        auto node = std::make_unique<ASTNode>(NodeTag::indexed_tag, ident);
        node->line = current_line_;
        node->column = current_column_;
        return node;
    }

    // Parenthesized expression
    if (match('(')) {
        auto expr = parse_expression();
        if (!expr) return nullptr;
        skip_whitespace();
        if (!match(')')) {
            errors_.push_back(ParseError(current_line_, current_column_, "Expected ')'"));
        }
        return expr;
    }

    // List expression [...]
    if (match('[')) {
        auto list = std::make_unique<ASTNode>(NodeTag::indexed_tag, "list");
        list->line = current_line_;
        list->column = current_column_;

        while (true) {
            if (match(']')) break;
            auto elem = parse_expression();
            if (elem) {
                list->children.push_back(std::move(elem));
            }
            if (!match(',')) {
                match(']');
                break;
            }
        }
        return list;
    }

    // Grouped expression with comma
    if (match('(')) {
        auto group = std::make_unique<ASTNode>(NodeTag::indexed_tag, "group");
        group->line = current_line_;
        group->column = current_column_;

        while (true) {
            if (match(')')) break;
            auto elem = parse_expression();
            if (elem) {
                group->children.push_back(std::move(elem));
            }
            if (!match(',')) {
                match(')');
                break;
            }
        }
        return group;
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
// TrikeShed-Specific Parsing
// ---------------------------------------------------------------------------

std::unique_ptr<ASTNode> Parser::parse_coordinates() {
    if (!match("coords")) {
        return nullptr;
    }

    if (!match('[')) {
        return nullptr;
    }

    auto coords_node = std::make_unique<ASTNode>(NodeTag::coordinates_tag, "coords");
    coords_node->line = current_line_;
    coords_node->column = current_column_;

    bool first = true;
    while (true) {
        if (match(']')) {
            break;
        }
        if (!first && !match(',')) {
            errors_.push_back(ParseError(current_line_, current_column_, "Expected ',' or ']'"));
            break;
        }
        first = false;

        auto expr = parse_expression();
        if (expr) {
            coords_node->children.push_back(std::move(expr));
        }
    }

    return coords_node;
}

std::unique_ptr<ASTNode> Parser::parse_join() {
    auto left = parse_expression();
    if (!left) return nullptr;

    skip_whitespace();
    if (match('j')) {
        auto right = parse_expression();
        if (right) {
            auto join_node = std::make_unique<ASTNode>(NodeTag::join_tag, "j");
            join_node->children.push_back(std::move(left));
            join_node->children.push_back(std::move(right));
            return join_node;
        }
    }

    return left;
}

std::unique_ptr<ASTNode> Parser::parse_alpha() {
    skip_whitespace();
    auto expr = parse_expression();
    if (!expr) return nullptr;

    auto alpha_node = std::make_unique<ASTNode>(NodeTag::normalize_tag, "alpha");
    alpha_node->line = current_line_;
    alpha_node->column = current_column_;
    alpha_node->children.push_back(std::move(expr));
    return alpha_node;
}

std::unique_ptr<ASTNode> Parser::parse_chart() {
    if (!match("chart")) {
        return nullptr;
    }

    auto chart_node = std::make_unique<ASTNode>(NodeTag::chart_project_tag, "chart");
    chart_node->line = current_line_;
    chart_node->column = current_column_;

    skip_whitespace();
    if (std::isalpha(peek()) || peek() == '_') {
        size_t start = position_;
        while (position_ < source_.size() &&
               (std::isalnum(source_[position_]) || source_[position_] == '_')) {
            position_++;
        }
        auto name = source_.substr(start, position_ - start);
        auto name_node = std::make_unique<ASTNode>(NodeTag::indexed_tag, name);
        name_node->line = current_line_;
        name_node->column = current_column_;
        chart_node->children.push_back(std::move(name_node));
    }

    if (match('(')) {
        skip_whitespace();
        if (std::isalpha(peek()) || peek() == '_') {
            size_t start = position_;
            while (position_ < source_.size() &&
                   (std::isalnum(source_[position_]) || source_[position_] == '_')) {
                position_++;
            }
            auto param = source_.substr(start, position_ - start);
            auto param_node = std::make_unique<ASTNode>(NodeTag::indexed_tag, param);
            param_node->line = current_line_;
            param_node->column = current_column_;
            chart_node->children.push_back(std::move(param_node));
        }
        match(')');
    }

    if (match('{')) {
        while (!match('}') && peek() != '\0') {
            skip_whitespace();
            if (match("project")) {
                auto project_node = std::make_unique<ASTNode>(NodeTag::chart_project_tag, "project");
                project_node->line = current_line_;
                project_node->column = current_column_;
                if (match('(')) {
                    auto arg = parse_expression();
                    if (arg) project_node->children.push_back(std::move(arg));
                    match(')');
                }
                chart_node->children.push_back(std::move(project_node));
            } else if (match("embed")) {
                auto embed_node = std::make_unique<ASTNode>(NodeTag::chart_embed_tag, "embed");
                embed_node->line = current_line_;
                embed_node->column = current_column_;
                if (match('(')) {
                    auto arg = parse_expression();
                    if (arg) embed_node->children.push_back(std::move(arg));
                    match(')');
                }
                chart_node->children.push_back(std::move(embed_node));
            } else {
                consume();
            }
        }
    }

    return chart_node;
}

std::unique_ptr<ASTNode> Parser::parse_atlas() {
    if (!match("atlas")) {
        return nullptr;
    }

    auto atlas_node = std::make_unique<ASTNode>(NodeTag::atlas_locate_tag, "atlas");
    atlas_node->line = current_line_;
    atlas_node->column = current_column_;

    if (match('[')) {
        while (!match(']') && peek() != '\0') {
            auto expr = parse_expression();
            if (expr) {
                atlas_node->children.push_back(std::move(expr));
            }
            if (!match(',')) {
                match(']');
                break;
            }
        }
    }

    return atlas_node;
}

std::unique_ptr<ASTNode> Parser::parse_manifold() {
    if (!match("manifold")) {
        return nullptr;
    }

    auto manifold_node = std::make_unique<ASTNode>(NodeTag::transition_tag, "manifold");
    manifold_node->line = current_line_;
    manifold_node->column = current_column_;

    skip_whitespace();
    if (std::isalpha(peek()) || peek() == '_') {
        size_t start = position_;
        while (position_ < source_.size() &&
               (std::isalnum(source_[position_]) || source_[position_] == '_')) {
            position_++;
        }
        auto name = source_.substr(start, position_ - start);
        auto name_node = std::make_unique<ASTNode>(NodeTag::indexed_tag, name);
        manifold_node->children.push_back(std::move(name_node));
    }

    if (match('=')) {
        auto expr = parse_expression();
        if (expr) {
            manifold_node->children.push_back(std::move(expr));
        }
    }

    return manifold_node;
}

std::unique_ptr<ASTNode> Parser::parse_transition() {
    auto expr = parse_primary_expression();
    if (!expr) return nullptr;

    skip_whitespace();
    if (match('.') && match("transition")) {
        if (match('(')) {
            auto transition_node = std::make_unique<ASTNode>(NodeTag::transition_tag, "transition");
            transition_node->children.push_back(std::move(expr));

            auto from = parse_expression();
            if (from) transition_node->children.push_back(std::move(from));
            match(',');
            auto to = parse_expression();
            if (to) transition_node->children.push_back(std::move(to));
            match(',');
            auto coords = parse_expression();
            if (coords) transition_node->children.push_back(std::move(coords));

            match(')');
            return transition_node;
        }
    }

    return expr;
}

// ============================================================================
// Normalization Implementation
// Per TrikeShed gospel: "TrikeShed Notation as Front-End Sugar Only"
// Normalize early into a small canonical AST
// ============================================================================

std::unique_ptr<CanonicalAST> Parser::normalize(CanonicalAST* ast) {
    if (!ast || !ast->root()) {
        return nullptr;
    }

    auto normalized = std::make_unique<CanonicalAST>();
    auto new_root = normalize_node(ast->root());
    normalized->set_root(std::move(new_root));

    return normalized;
}

std::unique_ptr<ASTNode> Parser::normalize_node(ASTNode* node) {
    if (!node) return nullptr;

    auto result = std::make_unique<ASTNode>(node->tag, node->text);
    result->line = node->line;
    result->column = node->column;

    for (auto& child : node->children) {
        auto normalized_child = normalize_node(child.get());
        if (normalized_child) {
            result->children.push_back(std::move(normalized_child));
        }
    }

    switch (node->tag) {
        case NodeTag::coordinates_tag:
            return normalize_coordinates(node);
        case NodeTag::chart_project_tag:
            return normalize_chart_project(node);
        case NodeTag::chart_embed_tag:
            return normalize_chart_embed(node);
        case NodeTag::atlas_locate_tag:
            return normalize_atlas_locate(node);
        case NodeTag::lower_dense_tag:
            return normalize_lower_dense(node);
        case NodeTag::indexed_tag:
            if (node->text == "project" || node->text == "embed" || 
                node->text == "locate" || node->text == "lowered") {
                return normalize_node_from_indexed(node);
            }
            break;
        default:
            break;
    }

    return result;
}

std::unique_ptr<ASTNode> Parser::normalize_node_from_indexed(ASTNode* node) {
    if (node->text == "project") {
        return normalize_chart_project(node);
    } else if (node->text == "embed") {
        return normalize_chart_embed(node);
    } else if (node->text == "locate") {
        return normalize_atlas_locate(node);
    } else if (node->text == "lowered") {
        return normalize_lower_dense(node);
    }
    return std::make_unique<ASTNode>(node->tag, node->text);
}

std::unique_ptr<ASTNode> Parser::normalize_coordinates(ASTNode* node) {
    auto normalized = std::make_unique<ASTNode>(NodeTag::coordinates_tag, "coordinates");
    normalized->line = node->line;
    normalized->column = node->column;

    for (auto& child : node->children) {
        normalized->children.push_back(std::move(child));
    }

    return normalized;
}

std::unique_ptr<ASTNode> Parser::normalize_chart_project(ASTNode* node) {
    auto normalized = std::make_unique<ASTNode>(NodeTag::chart_project_tag, "chart_project");
    normalized->line = node->line;
    normalized->column = node->column;

    for (auto& child : node->children) {
        normalized->children.push_back(std::move(child));
    }

    return normalized;
}

std::unique_ptr<ASTNode> Parser::normalize_chart_embed(ASTNode* node) {
    auto normalized = std::make_unique<ASTNode>(NodeTag::chart_embed_tag, "chart_embed");
    normalized->line = node->line;
    normalized->column = node->column;

    for (auto& child : node->children) {
        normalized->children.push_back(std::move(child));
    }

    return normalized;
}

std::unique_ptr<ASTNode> Parser::normalize_atlas_locate(ASTNode* node) {
    auto normalized = std::make_unique<ASTNode>(NodeTag::atlas_locate_tag, "atlas_locate");
    normalized->line = node->line;
    normalized->column = node->column;

    for (auto& child : node->children) {
        normalized->children.push_back(std::move(child));
    }

    return normalized;
}

std::unique_ptr<ASTNode> Parser::normalize_lower_dense(ASTNode* node) {
    auto normalized = std::make_unique<ASTNode>(NodeTag::lower_dense_tag, "lower_dense");
    normalized->line = node->line;
    normalized->column = node->column;

    for (auto& child : node->children) {
        normalized->children.push_back(std::move(child));
    }

    return normalized;
}

} // namespace cppfort

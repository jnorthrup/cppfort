#include "c_parser.h"
#include <stdexcept>

namespace cppfort::c {

// ============================================================================
// CParser Implementation - Recursive Descent
// ============================================================================

const CToken& CParser::peek() const {
    if (_pos >= _tokens.size()) {
        throw std::runtime_error("Unexpected end of file");
    }
    return _tokens[_pos];
}

const CToken& CParser::advance() {
    if (_pos >= _tokens.size()) {
        throw std::runtime_error("Unexpected end of file");
    }
    return _tokens[_pos++];
}

bool CParser::match(CTokenType type) {
    if (peek().type == type) {
        advance();
        return true;
    }
    return false;
}

bool CParser::expect(CTokenType type) {
    if (!match(type)) {
        error("Expected token type " + std::to_string(static_cast<int>(type)));
        return false;
    }
    return true;
}

void CParser::error(const std::string& message) {
    const auto& tok = peek();
    throw std::runtime_error(
        tok.filename + ":" + std::to_string(tok.line) + ":" +
        std::to_string(tok.column) + ": " + message
    );
}

// ============================================================================
// Expression Parsing - C Precedence Levels (14 levels)
// ============================================================================

// Primary: literals, identifiers, (expr)
std::unique_ptr<CASTNode> CParser::parsePrimaryExpression() {
    const auto& tok = peek();

    // Integer literal
    if (tok.type == CTokenType::INTEGER_LITERAL) {
        auto node = std::make_unique<CASTNode>(CASTNodeType::INTEGER_CONST, tok.line, tok.column);
        node->value = tok.text;
        advance();
        return node;
    }

    // Float literal
    if (tok.type == CTokenType::FLOAT_LITERAL) {
        auto node = std::make_unique<CASTNode>(CASTNodeType::FLOAT_CONST, tok.line, tok.column);
        node->value = tok.text;
        advance();
        return node;
    }

    // String literal
    if (tok.type == CTokenType::STRING_LITERAL) {
        auto node = std::make_unique<CASTNode>(CASTNodeType::STRING_CONST, tok.line, tok.column);
        node->value = tok.text;
        advance();
        return node;
    }

    // Character literal
    if (tok.type == CTokenType::CHAR_LITERAL) {
        auto node = std::make_unique<CASTNode>(CASTNodeType::CHAR_CONST, tok.line, tok.column);
        node->value = tok.text;
        advance();
        return node;
    }

    // Identifier
    if (tok.type == CTokenType::IDENTIFIER) {
        auto node = std::make_unique<CASTNode>(CASTNodeType::IDENTIFIER_REF, tok.line, tok.column);
        node->value = tok.text;
        advance();
        return node;
    }

    // Parenthesized expression
    if (tok.type == CTokenType::LPAREN) {
        advance(); // (
        auto expr = parseExpression();
        expect(CTokenType::RPAREN);
        return expr;
    }

    error("Expected primary expression");
    return nullptr;
}

// Postfix: x[i], x(args), x.m, x->m, x++, x--
std::unique_ptr<CASTNode> CParser::parsePostfixExpression() {
    auto expr = parsePrimaryExpression();

    while (true) {
        const auto& tok = peek();

        // Array subscript: expr[index]
        if (tok.type == CTokenType::LBRACKET) {
            advance();
            auto node = std::make_unique<CASTNode>(CASTNodeType::ARRAY_SUBSCRIPT, tok.line, tok.column);
            node->addChild(std::move(expr));
            node->addChild(parseExpression());
            expect(CTokenType::RBRACKET);
            expr = std::move(node);
            continue;
        }

        // Function call: expr(args)
        if (tok.type == CTokenType::LPAREN) {
            advance();
            auto node = std::make_unique<CASTNode>(CASTNodeType::CALL_EXPR, tok.line, tok.column);
            node->addChild(std::move(expr));

            // Parse arguments
            if (peek().type != CTokenType::RPAREN) {
                do {
                    node->addChild(parseAssignmentExpression());
                } while (match(CTokenType::COMMA));
            }
            expect(CTokenType::RPAREN);
            expr = std::move(node);
            continue;
        }

        // Member access: expr.member
        if (tok.type == CTokenType::DOT) {
            advance();
            auto node = std::make_unique<CASTNode>(CASTNodeType::MEMBER_ACCESS, tok.line, tok.column);
            node->addChild(std::move(expr));
            auto member = std::make_unique<CASTNode>(CASTNodeType::IDENTIFIER_REF, peek().line, peek().column);
            member->value = expect(CTokenType::IDENTIFIER) ? _tokens[_pos - 1].text : "";
            node->addChild(std::move(member));
            expr = std::move(node);
            continue;
        }

        // Pointer member access: expr->member
        if (tok.type == CTokenType::ARROW) {
            advance();
            auto node = std::make_unique<CASTNode>(CASTNodeType::MEMBER_ACCESS, tok.line, tok.column);
            // Desugar: expr->member = (*expr).member
            auto deref = std::make_unique<CASTNode>(CASTNodeType::POINTER_DEREF, tok.line, tok.column);
            deref->addChild(std::move(expr));
            node->addChild(std::move(deref));
            auto member = std::make_unique<CASTNode>(CASTNodeType::IDENTIFIER_REF, peek().line, peek().column);
            member->value = expect(CTokenType::IDENTIFIER) ? _tokens[_pos - 1].text : "";
            node->addChild(std::move(member));
            expr = std::move(node);
            continue;
        }

        // Postfix increment/decrement: expr++ / expr--
        if (tok.type == CTokenType::INCREMENT || tok.type == CTokenType::DECREMENT) {
            advance();
            auto node = std::make_unique<CASTNode>(CASTNodeType::UNARY_EXPR, tok.line, tok.column);
            node->value = tok.text; // "++" or "--"
            node->addChild(std::move(expr));
            expr = std::move(node);
            continue;
        }

        break;
    }

    return expr;
}

// Cast and unary: (type)x, ++x, --x, +x, -x, !x, ~x, *x, &x, sizeof
std::unique_ptr<CASTNode> CParser::parseUnaryExpression() {
    const auto& tok = peek();

    // Prefix increment/decrement
    if (tok.type == CTokenType::INCREMENT || tok.type == CTokenType::DECREMENT) {
        advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::UNARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(parseUnaryExpression());
        return node;
    }

    // Unary operators: +, -, !, ~, *, &
    if (tok.type == CTokenType::PLUS || tok.type == CTokenType::MINUS ||
        tok.type == CTokenType::EXCLAIM || tok.type == CTokenType::TILDE ||
        tok.type == CTokenType::STAR || tok.type == CTokenType::AMPERSAND) {
        advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::UNARY_EXPR, tok.line, tok.column);
        node->value = tok.text;

        // Special handling for * (dereference) and & (address-of)
        if (tok.type == CTokenType::STAR) {
            node->type = CASTNodeType::POINTER_DEREF;
        } else if (tok.type == CTokenType::AMPERSAND) {
            node->type = CASTNodeType::ADDRESS_OF;
        }

        node->addChild(parseUnaryExpression());
        return node;
    }

    // sizeof operator
    if (tok.type == CTokenType::SIZEOF) {
        advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::SIZEOF_EXPR, tok.line, tok.column);

        // sizeof can take either (type) or unary-expression
        if (peek().type == CTokenType::LPAREN) {
            // Could be sizeof(type) or sizeof(expr)
            // Simplified: assume it's an expression for now
            advance(); // (
            node->addChild(parseExpression());
            expect(CTokenType::RPAREN);
        } else {
            node->addChild(parseUnaryExpression());
        }
        return node;
    }

    return parsePostfixExpression();
}

std::unique_ptr<CASTNode> CParser::parseCastExpression() {
    // Simplified cast detection - production would need lookahead to distinguish (type)expr from (expr)
    return parseUnaryExpression();
}

// Multiplicative: *, /, %
std::unique_ptr<CASTNode> CParser::parseMultiplicativeExpression() {
    auto left = parseCastExpression();

    while (true) {
        const auto& tok = peek();
        if (tok.type != CTokenType::STAR && tok.type != CTokenType::SLASH && tok.type != CTokenType::PERCENT) {
            break;
        }
        advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::BINARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(std::move(left));
        node->addChild(parseCastExpression());
        left = std::move(node);
    }

    return left;
}

// Additive: +, -
std::unique_ptr<CASTNode> CParser::parseAdditiveExpression() {
    auto left = parseMultiplicativeExpression();

    while (true) {
        const auto& tok = peek();
        if (tok.type != CTokenType::PLUS && tok.type != CTokenType::MINUS) {
            break;
        }
        advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::BINARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(std::move(left));
        node->addChild(parseMultiplicativeExpression());
        left = std::move(node);
    }

    return left;
}

// Shift: <<, >>
std::unique_ptr<CASTNode> CParser::parseShiftExpression() {
    auto left = parseAdditiveExpression();

    while (true) {
        const auto& tok = peek();
        if (tok.type != CTokenType::LSHIFT && tok.type != CTokenType::RSHIFT) {
            break;
        }
        advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::BINARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(std::move(left));
        node->addChild(parseAdditiveExpression());
        left = std::move(node);
    }

    return left;
}

// Relational: <, >, <=, >=
std::unique_ptr<CASTNode> CParser::parseRelationalExpression() {
    auto left = parseShiftExpression();

    while (true) {
        const auto& tok = peek();
        if (tok.type != CTokenType::LT && tok.type != CTokenType::GT &&
            tok.type != CTokenType::LE && tok.type != CTokenType::GE) {
            break;
        }
        advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::BINARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(std::move(left));
        node->addChild(parseShiftExpression());
        left = std::move(node);
    }

    return left;
}

// Equality: ==, !=
std::unique_ptr<CASTNode> CParser::parseEqualityExpression() {
    auto left = parseRelationalExpression();

    while (true) {
        const auto& tok = peek();
        if (tok.type != CTokenType::EQ && tok.type != CTokenType::NE) {
            break;
        }
        advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::BINARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(std::move(left));
        node->addChild(parseRelationalExpression());
        left = std::move(node);
    }

    return left;
}

// Bitwise AND: &
std::unique_ptr<CASTNode> CParser::parseBitwiseAndExpression() {
    auto left = parseEqualityExpression();

    while (peek().type == CTokenType::AMPERSAND) {
        const auto& tok = advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::BINARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(std::move(left));
        node->addChild(parseEqualityExpression());
        left = std::move(node);
    }

    return left;
}

// Bitwise XOR: ^
std::unique_ptr<CASTNode> CParser::parseBitwiseXorExpression() {
    auto left = parseBitwiseAndExpression();

    while (peek().type == CTokenType::CARET) {
        const auto& tok = advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::BINARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(std::move(left));
        node->addChild(parseBitwiseAndExpression());
        left = std::move(node);
    }

    return left;
}

// Bitwise OR: |
std::unique_ptr<CASTNode> CParser::parseBitwiseOrExpression() {
    auto left = parseBitwiseXorExpression();

    while (peek().type == CTokenType::PIPE) {
        const auto& tok = advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::BINARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(std::move(left));
        node->addChild(parseBitwiseXorExpression());
        left = std::move(node);
    }

    return left;
}

// Logical AND: &&
std::unique_ptr<CASTNode> CParser::parseLogicalAndExpression() {
    auto left = parseBitwiseOrExpression();

    while (peek().type == CTokenType::AND) {
        const auto& tok = advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::BINARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(std::move(left));
        node->addChild(parseBitwiseOrExpression());
        left = std::move(node);
    }

    return left;
}

// Logical OR: ||
std::unique_ptr<CASTNode> CParser::parseLogicalOrExpression() {
    auto left = parseLogicalAndExpression();

    while (peek().type == CTokenType::OR) {
        const auto& tok = advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::BINARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(std::move(left));
        node->addChild(parseLogicalAndExpression());
        left = std::move(node);
    }

    return left;
}

// Conditional: ? :
std::unique_ptr<CASTNode> CParser::parseConditionalExpression() {
    auto cond = parseLogicalOrExpression();

    if (peek().type == CTokenType::QUESTION) {
        const auto& tok = advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::CONDITIONAL_EXPR, tok.line, tok.column);
        node->addChild(std::move(cond));
        node->addChild(parseExpression());
        expect(CTokenType::COLON);
        node->addChild(parseConditionalExpression());
        return node;
    }

    return cond;
}

// Assignment: =, +=, -=, *=, /=, %=, &=, |=, ^=, <<=, >>=
std::unique_ptr<CASTNode> CParser::parseAssignmentExpression() {
    auto left = parseConditionalExpression();

    const auto& tok = peek();
    if (tok.type == CTokenType::ASSIGN ||
        tok.type == CTokenType::PLUS_ASSIGN || tok.type == CTokenType::MINUS_ASSIGN ||
        tok.type == CTokenType::STAR_ASSIGN || tok.type == CTokenType::SLASH_ASSIGN ||
        tok.type == CTokenType::PERCENT_ASSIGN || tok.type == CTokenType::AMPERSAND_ASSIGN ||
        tok.type == CTokenType::PIPE_ASSIGN || tok.type == CTokenType::CARET_ASSIGN ||
        tok.type == CTokenType::LSHIFT_ASSIGN || tok.type == CTokenType::RSHIFT_ASSIGN) {
        advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::BINARY_EXPR, tok.line, tok.column);
        node->value = tok.text;
        node->addChild(std::move(left));
        node->addChild(parseAssignmentExpression());  // Right associative
        return node;
    }

    return left;
}

// Expression: comma-separated assignment expressions
std::unique_ptr<CASTNode> CParser::parseExpression() {
    auto left = parseAssignmentExpression();

    while (peek().type == CTokenType::COMMA) {
        const auto& tok = advance();
        auto node = std::make_unique<CASTNode>(CASTNodeType::COMMA_EXPR, tok.line, tok.column);
        node->addChild(std::move(left));
        node->addChild(parseAssignmentExpression());
        left = std::move(node);
    }

    return left;
}

// Stub implementations for now - will implement statement/declaration parsing next
std::unique_ptr<CASTNode> CParser::parseStatement() {
    // TODO: Implement
    return nullptr;
}

std::unique_ptr<CASTNode> CParser::parseCompoundStatement() {
    // TODO: Implement
    return nullptr;
}

std::unique_ptr<CASTNode> CParser::parse() {
    auto root = std::make_unique<CASTNode>(CASTNodeType::TRANSLATION_UNIT);

    while (peek().type != CTokenType::EOF_TOKEN) {
        // TODO: Parse top-level declarations
        advance();  // Skip for now
    }

    return root;
}

// Node emission stub
ir::Node* CParser::emit(const CASTNode* ast) {
    // TODO: Implement AST→Node emission
    return nullptr;
}

} // namespace cppfort::c

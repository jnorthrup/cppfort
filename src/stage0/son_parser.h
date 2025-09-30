#ifndef CPPFORT_SON_PARSER_H
#define CPPFORT_SON_PARSER_H

#include "node.h"
#include <string>
#include <memory>

namespace cppfort::ir {

/**
 * Sea of Nodes Parser - builds graph directly during parsing.
 * Following Simple compiler Chapter 1-3 approach.
 * No AST - direct graph construction.
 */
class SoNParser {
private:
    std::string _source;
    size_t _position;

    // Global START node (like Simple's Parser.START)
    StartNode* START;

    // Current control node for tracking control flow
    Node* _ctrl;

    // Scope node for managing lexical scopes (Chapter 3)
    ScopeNode* _scope;

    // Simple lexer functionality
    void skipWhitespace();
    bool peek(const std::string& expected);
    bool consume(const std::string& expected);
    int parseInteger();
    std::string parseIdentifier();
    char peek();
    char advance();
    bool isEOF() const;
    bool isAlpha(char c) const;
    bool isDigit(char c) const;
    bool isAlphaNum(char c) const;

public:
    SoNParser();
    ~SoNParser();

    /**
     * Parse a program and return the graph's terminal node.
     * Chapter 3: supports declarations, blocks, assignments, returns.
     */
    Node* parse(const std::string& source);

    /**
     * Parse a program: a sequence of statements.
     */
    Node* parseProgram();

    /**
     * Parse a statement.
     * Chapter 3: declarations, blocks, assignments, returns.
     */
    Node* parseStatement();

    /**
     * Parse a declaration: "int <identifier> = <expression>;"
     */
    Node* parseDeclaration();

    /**
     * Parse a block: "{" statements "}"
     */
    Node* parseBlock();

    /**
     * Parse an assignment: "<identifier> = <expression>;"
     */
    Node* parseAssignment();
    Node* parseIf();

    /**
     * Parse a return statement: "return <integer>;"
     */
    Node* parseReturn();

    /**
     * Parse an expression.
     * Chapter 2: full arithmetic expressions with precedence.
     */
    Node* parseExpression();

    /**
     * Parse additive expression (+ and -).
     * Chapter 2: handles left-to-right associativity.
     */
    Node* parseAddition();

    /**
     * Parse multiplicative expression (* and /).
     * Chapter 2: higher precedence than addition.
     */
    Node* parseMultiplication();

    /**
     * Parse shift expressions (<<, >>, >>>).
     * Chapter 16: higher precedence than addition, lower than multiplication.
     */
    Node* parseShifts();

    /**
     * Parse bitwise AND expression (&).
     * Chapter 16: higher precedence than shifts, lower than comparisons.
     */
    Node* parseBitwiseAnd();

    /**
     * Parse bitwise XOR expression (^).
     * Chapter 16: higher precedence than bitwise AND, lower than bitwise OR.
     */
    Node* parseBitwiseXor();

    /**
     * Parse bitwise OR expression (|).
     * Chapter 16: higher precedence than bitwise XOR, lower than comparisons.
     */
    Node* parseBitwiseOr();

    /**
     * Parse unary expression (unary minus).
     * Chapter 2: highest precedence.
     */
    Node* parseUnary();

    /**
     * Parse primary expression (literals, identifiers, and parentheses).
     * Chapter 3: adds identifier lookup.
     */
    Node* parsePrimary();

    /**
     * Get the START node of the graph.
     */
    StartNode* getStart() const { return START; }

    /**
     * Visualize the graph (for debugging).
     * Returns a simple text representation.
     */
    std::string visualize() const;
};

} // namespace cppfort::ir

#endif // CPPFORT_SON_PARSER_H

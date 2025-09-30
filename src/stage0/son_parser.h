#ifndef CPPFORT_SON_PARSER_H
#define CPPFORT_SON_PARSER_H

#include "node.h"
#include <string>
#include <memory>

namespace cppfort::ir {

/**
 * Sea of Nodes Parser - builds graph directly during parsing.
 * Following Simple compiler Chapter 1 approach.
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

    // Simple lexer functionality
    void skipWhitespace();
    bool peek(const std::string& expected);
    bool consume(const std::string& expected);
    int parseInteger();
    char peek();
    char advance();
    bool isEOF() const;

public:
    SoNParser();
    ~SoNParser();

    /**
     * Parse a program and return the graph's terminal node.
     * For Chapter 1, we only handle "return <integer>;"
     */
    Node* parse(const std::string& source);

    /**
     * Parse a statement.
     * Chapter 1: only return statements.
     */
    Node* parseStatement();

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
     * Parse unary expression (unary minus).
     * Chapter 2: highest precedence.
     */
    Node* parseUnary();

    /**
     * Parse primary expression (literals and parentheses).
     * Chapter 2: base case for recursion.
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
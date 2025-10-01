#include "son_parser.h"
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <algorithm>
#include <set>
#include <iostream>

namespace cppfort::ir {

SoNParser::SoNParser() : _position(0), START(nullptr), _ctrl(nullptr), _scope(nullptr) {
}

SoNParser::~SoNParser() {
    // Cleanup nodes if needed
    // In a real implementation, we'd have proper memory management
}

Node* SoNParser::parse(const std::string& source) {
    _source = source;
    _position = 0;

    // Create the START node - entry point to the function
    START = new StartNode();
    _ctrl = START;  // Current control is at start

    // Create the ScopeNode for managing variables (Chapter 3)
    _scope = new ScopeNode();

    // Parse the program (Chapter 3: multiple statements)
    Node* ret = parseProgram();

    // Skip trailing whitespace
    skipWhitespace();

    // Ensure we consumed all input
    if (!isEOF()) {
        throw std::runtime_error("Expected EOF but found: " + std::string(1, peek()));
    }

    return ret;
}

Node* SoNParser::parseProgram() {
    Node* ret = nullptr;

    // Parse statements until we hit EOF or return
    while (!isEOF()) {
        skipWhitespace();
        if (isEOF()) break;

        Node* stmt = parseStatement();

        // If it's a return node, we're done
        if (dynamic_cast<ReturnNode*>(stmt)) {
            ret = stmt;
            break;
        }

        // For Band 1: Accept any statement as valid result
        // (returns may be inside if branches)
        if (stmt) {
            ret = stmt;
        }
    }

    return ret;
}

Node* SoNParser::parseStatement() {
    skipWhitespace();

    // Chapter 3: Support various statement types

    // Block statement
    if (peek() == '{') {
        return parseBlock();
    }

    // If statement
    if (peek("if")) {
        // parseIf handles control and scope merging
        return parseIf();
    }

    // Declaration statement
    if (peek("int")) {
        return parseDeclaration();
    }

    // Return statement
    if (peek("return")) {
        return parseReturn();
    }

    // Assignment or expression statement
    if (isAlpha(peek())) {
        return parseAssignment();
    }

    throw std::runtime_error("Expected statement (declaration, block, assignment, or return)");
}

Node* SoNParser::parseReturn() {
    // Consume "return"
    if (!consume("return")) {
        throw std::runtime_error("Expected 'return' keyword");
    }

    skipWhitespace();

    // Parse the expression (Chapter 1: integer literal)
    Node* value = parseExpression();

    skipWhitespace();

    // Consume semicolon
    if (!consume(";")) {
        throw std::runtime_error("Expected ';' after return statement");
    }

    // Create ReturnNode with current control and the value
    ReturnNode* ret = new ReturnNode(_ctrl, value);

    return ret;
}

Node* SoNParser::parseExpression() {
    // Chapter 2+: comparisons bind looser than bitwise OR
    // Chapter 16: bitwise operations with proper precedence
    Node* lhs = parseBitwiseOr();
    while (true) {
        skipWhitespace();
        if (peek("==")) {
            consume("==");
            Node* rhs = parseBitwiseOr();
            lhs = (new EQNode(lhs, rhs))->peephole();
        } else if (peek("<")) {
            consume("<");
            Node* rhs = parseBitwiseOr();
            lhs = (new LTNode(lhs, rhs))->peephole();
        } else {
            break;
        }
    }
    return lhs;
}

Node* SoNParser::parseAddition() {
    Node* lhs = parseMultiplication();

    while (true) {
        skipWhitespace();

        if (peek() == '+') {
            advance();
            Node* rhs = parseMultiplication();
            // Create AddNode and apply peephole optimization
            lhs = (new AddNode(lhs, rhs))->peephole();
        } else if (peek() == '-') {
            // This is subtraction (unary minus is handled in parseUnary)
            advance();
            Node* rhs = parseMultiplication();
            // Create SubNode and apply peephole optimization
            lhs = (new SubNode(lhs, rhs))->peephole();
        } else {
            break;
        }
    }

    return lhs;
}

Node* SoNParser::parseMultiplication() {
    Node* lhs = parseUnary();

    while (true) {
        skipWhitespace();

        if (peek() == '*') {
            advance();
            Node* rhs = parseUnary();
            // Create MulNode and apply peephole optimization
            lhs = (new MulNode(lhs, rhs))->peephole();
        } else if (peek() == '/') {
            advance();
            Node* rhs = parseUnary();
            // Create DivNode and apply peephole optimization
            lhs = (new DivNode(lhs, rhs))->peephole();
        } else {
            break;
        }
    }

    return lhs;
}

// ============================================================================
// Chapter 16: Bitwise Operation Parsing
// ============================================================================

Node* SoNParser::parseBitwiseOr() {
    Node* lhs = parseBitwiseXor();

    while (true) {
        skipWhitespace();

        if (peek() == '|') {
            advance();
            Node* rhs = parseBitwiseXor();
            lhs = (new OrNode(lhs, rhs))->peephole();
        } else {
            break;
        }
    }

    return lhs;
}

Node* SoNParser::parseBitwiseXor() {
    Node* lhs = parseBitwiseAnd();

    while (true) {
        skipWhitespace();

        if (peek() == '^') {
            advance();
            Node* rhs = parseBitwiseAnd();
            lhs = (new XorNode(lhs, rhs))->peephole();
        } else {
            break;
        }
    }

    return lhs;
}

Node* SoNParser::parseBitwiseAnd() {
    Node* lhs = parseShifts();

    while (true) {
        skipWhitespace();

        if (peek() == '&') {
            advance();
            Node* rhs = parseShifts();
            lhs = (new AndNode(lhs, rhs))->peephole();
        } else {
            break;
        }
    }

    return lhs;
}

Node* SoNParser::parseShifts() {
    Node* lhs = parseAddition();

    while (true) {
        skipWhitespace();

        if (peek("<<")) {
            consume("<<");
            Node* rhs = parseAddition();
            lhs = (new ShlNode(lhs, rhs))->peephole();
        } else if (peek(">>>")) {
            consume(">>>");
            Node* rhs = parseAddition();
            lhs = (new AShrNode(lhs, rhs))->peephole();
        } else if (peek(">>")) {
            consume(">>");
            Node* rhs = parseAddition();
            lhs = (new LShrNode(lhs, rhs))->peephole();
        } else {
            break;
        }
    }

    return lhs;
}

Node* SoNParser::parseUnary() {
    skipWhitespace();

    if (peek() == '-') {
        // This is unary minus
        advance();
        // Recursively parse unary to handle multiple minuses like --5
        Node* value = parseUnary();
        // Create MinusNode and apply peephole optimization
        return (new MinusNode(value))->peephole();
    }

    return parsePrimary();
}

Node* SoNParser::parsePrimary() {
    skipWhitespace();

    // Handle parentheses
    if (peek() == '(') {
        advance();
        Node* expr = parseExpression();
        skipWhitespace();
        if (peek() != ')') {
            throw std::runtime_error("Expected ')'");
        }
        advance();
        return expr;
    }

    // Handle integer literals
    if (std::isdigit(peek()) || (peek() == '0')) {
        int value = parseInteger();
        // Create ConstantNode with START as input for graph walking
        return new ConstantNode(value, START);
    }

    // Handle identifiers (Chapter 3)
    if (isAlpha(peek())) {
        std::string name = parseIdentifier();
        Node* value = _scope->lookup(name);
        if (value == nullptr) {
            throw std::runtime_error("Undefined variable: " + name);
        }
        return value;
    }

    throw std::runtime_error("Expected integer literal, identifier, or '('");
}

// INSTRUCTION: If statement parsing pattern
Node* SoNParser::parseIf() {
    consume("if");
    consume("(");
    Node* pred = parseExpression();  // Comparison expression
    consume(")");

    // Create If node with current control and predicate
    IfNode* ifNode = new IfNode(_ctrl, pred);

    // Create projections for true/false branches
    ProjNode* ifTrue = new ProjNode(ifNode, 0);
    ProjNode* ifFalse = new ProjNode(ifNode, 1);

    // Duplicate scope for both branches
    ScopeNode* thenScope = _scope->duplicate();
    ScopeNode* elseScope = _scope->duplicate();

    // Parse then branch
    _ctrl = ifTrue;
    _scope = thenScope;
    Node* thenBody = parseStatement();
    (void)thenBody; // control comes from _ctrl
    Node* thenCtrl = _ctrl;  // Save control after then

    // Parse else branch (if present)
    _ctrl = ifFalse;
    _scope = elseScope;
    Node* elseCtrl = _ctrl;
    skipWhitespace();
    if (peek("else")) { consume("else"); parseStatement(); elseCtrl = _ctrl; }

    // Merge control and scopes
    RegionNode* region = new RegionNode(thenCtrl, elseCtrl);
    _ctrl = region;

    // Merge scopes with phi creation
    // Build union of variable names across both scopes
    auto s1 = thenScope->currentBindings();
    auto s2 = elseScope->currentBindings();
    // Start from pre-if scope to maintain stack depth
    ScopeNode* merged = _scope->duplicate();
    // Start with pre-if bindings
    for (const auto& [name, node] : _scope->currentBindings()) {
        (void)name; (void)node;
    }
    // Union
    std::set<std::string> all;
    for (const auto& [k,_] : s1) all.insert(k);
    for (const auto& [k,_] : s2) all.insert(k);
    for (const auto& name : all) {
        Node* v1 = s1.count(name) ? s1[name] : nullptr;
        Node* v2 = s2.count(name) ? s2[name] : nullptr;
        if (v1 && v2) {
            if (v1 == v2) {
                merged->define(name, v1);
            } else {
                // Two-phase Phi: region may be set now
                auto phi = new PhiNode(name, region, v1, v2);
                region->addPhi(phi);
                merged->define(name, phi);
            }
        } else if (v1) {
            merged->define(name, v1);
        } else if (v2) {
            merged->define(name, v2);
        }
    }
    _scope = merged;

    return region;
}

Node* SoNParser::parseDeclaration() {
    // Consume "int"
    if (!consume("int")) {
        throw std::runtime_error("Expected 'int' keyword");
    }

    skipWhitespace();

    // Parse identifier
    std::string name = parseIdentifier();

    skipWhitespace();

    // Consume "="
    if (!consume("=")) {
        throw std::runtime_error("Expected '=' in declaration");
    }

    skipWhitespace();

    // Parse expression
    Node* value = parseExpression();

    skipWhitespace();

    // Consume ";"
    if (!consume(";")) {
        throw std::runtime_error("Expected ';' after declaration");
    }

    // Define the variable in the current scope
    _scope->define(name, value);

    return nullptr;  // Declarations don't produce a value
}

Node* SoNParser::parseBlock() {
    // Consume "{"
    if (!consume("{")) {
        throw std::runtime_error("Expected '{'");
    }

    // Push a new scope
    _scope->push();

    // Parse statements inside the block
    Node* ret = nullptr;
    while (!peek("}") && !isEOF()) {
        skipWhitespace();
        if (peek("}")) break;

        Node* stmt = parseStatement();

        // If it's a return, save it
        if (dynamic_cast<ReturnNode*>(stmt)) {
            ret = stmt;
            break;
        }
    }

    skipWhitespace();

    // Consume "}"
    if (!consume("}")) {
        throw std::runtime_error("Expected '}'");
    }

    // Pop the scope
    _scope->pop();

    return ret;
}

Node* SoNParser::parseAssignment() {
    // Parse identifier
    std::string name = parseIdentifier();

    skipWhitespace();

    // Consume "="
    if (!consume("=")) {
        throw std::runtime_error("Expected '=' in assignment");
    }

    skipWhitespace();

    // Parse expression
    Node* value = parseExpression();

    skipWhitespace();

    // Consume ";"
    if (!consume(";")) {
        throw std::runtime_error("Expected ';' after assignment");
    }

    // Update the variable
    _scope->update(name, value);

    return nullptr;  // Assignments don't produce a value
}

void SoNParser::skipWhitespace() {
    while (!isEOF() && std::isspace(_source[_position])) {
        _position++;
    }
}

bool SoNParser::peek(const std::string& expected) {
    size_t pos = _position;
    for (char c : expected) {
        if (pos >= _source.length() || _source[pos] != c) {
            return false;
        }
        pos++;
    }
    // Make sure it's not part of a larger identifier (only for alphabetic keywords)
    if (std::isalpha(expected[0]) && pos < _source.length() && isAlphaNum(_source[pos])) {
        return false;
    }
    return true;
}

bool SoNParser::consume(const std::string& expected) {
    if (peek(expected)) {
        _position += expected.length();
        return true;
    }
    return false;
}

int SoNParser::parseInteger() {
    std::string numStr;
    bool negative = false;

    if (peek() == '-') {
        negative = true;
        advance();
    }

    if (!std::isdigit(peek())) {
        throw std::runtime_error("Expected digit");
    }

    while (!isEOF() && std::isdigit(peek())) {
        numStr += advance();
    }

    int value = std::stoi(numStr);
    return negative ? -value : value;
}

char SoNParser::peek() {
    if (isEOF()) {
        return '\0';
    }
    return _source[_position];
}

char SoNParser::advance() {
    if (isEOF()) {
        return '\0';
    }
    return _source[_position++];
}

bool SoNParser::isEOF() const {
    return _position >= _source.length();
}

std::string SoNParser::parseIdentifier() {
    std::string id;

    if (!isAlpha(peek())) {
        throw std::runtime_error("Expected identifier");
    }

    // First character must be alphabetic or underscore
    id += advance();

    // Subsequent characters can be alphanumeric or underscore
    while (!isEOF() && isAlphaNum(peek())) {
        id += advance();
    }

    return id;
}

bool SoNParser::isAlpha(char c) const {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

bool SoNParser::isDigit(char c) const {
    return c >= '0' && c <= '9';
}

bool SoNParser::isAlphaNum(char c) const {
    return isAlpha(c) || isDigit(c);
}

std::string SoNParser::visualize() const {
    std::ostringstream ss;
    ss << "Sea of Nodes Graph:\n";
    ss << "===================\n";

    if (START == nullptr) {
        ss << "(empty graph)\n";
        return ss.str();
    }

    // Simple text visualization showing nodes and connections
    ss << "Nodes:\n";

    // Walk from START
    std::vector<Node*> visited;
    std::vector<Node*> worklist;
    worklist.push_back(START);

    while (!worklist.empty()) {
        Node* n = worklist.back();
        worklist.pop_back();

        // Check if already visited
        if (std::find(visited.begin(), visited.end(), n) != visited.end()) {
            continue;
        }
        visited.push_back(n);

        // Print node info
        ss << "  " << n->toString();
        if (n->isCFG()) {
            ss << " [CFG]";
        }
        ss << "\n";

        // Print inputs
        if (!n->_inputs.empty()) {
            ss << "    inputs: ";
            for (size_t i = 0; i < n->_inputs.size(); i++) {
                if (n->_inputs[i] != nullptr) {
                    ss << "[" << i << "]=" << n->_inputs[i]->_nid << " ";
                }
            }
            ss << "\n";
        }

        // Add outputs to worklist
        for (Node* out : n->_outputs) {
            worklist.push_back(out);
        }
    }

    return ss.str();
}

} // namespace cppfort::ir

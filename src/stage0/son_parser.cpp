#include "son_parser.h"
#include <stdexcept>
#include <sstream>
#include <cctype>
#include <algorithm>

namespace cppfort::ir {

SoNParser::SoNParser() : _position(0), START(nullptr), _ctrl(nullptr) {
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

    // Parse the program (for Chapter 1: just one return statement)
    Node* ret = parseStatement();

    // Skip trailing whitespace
    skipWhitespace();

    // Ensure we consumed all input
    if (!isEOF()) {
        throw std::runtime_error("Expected EOF but found: " + std::string(1, peek()));
    }

    return ret;
}

Node* SoNParser::parseStatement() {
    skipWhitespace();

    // Chapter 1: Only return statements
    if (peek("return")) {
        return parseReturn();
    }

    throw std::runtime_error("Expected 'return' statement");
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
    skipWhitespace();

    // Chapter 1: Only integer literals
    if (std::isdigit(peek()) || peek() == '-') {
        int value = parseInteger();
        // Create ConstantNode with START as input for graph walking
        return new ConstantNode(value, START);
    }

    throw std::runtime_error("Expected integer literal");
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
    // Make sure it's not part of a larger identifier
    if (pos < _source.length() && std::isalnum(_source[pos])) {
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
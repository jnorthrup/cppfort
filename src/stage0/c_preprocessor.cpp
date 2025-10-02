#include "c_parser.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>

namespace cppfort::c {

// ============================================================================
// CPreprocessor Implementation
// ============================================================================

void CPreprocessor::defineMacro(const std::string& name, const std::string& value) {
    _macros[name] = value;
}

void CPreprocessor::undefineMacro(const std::string& name) {
    _macros.erase(name);
}

bool CPreprocessor::isDefined(const std::string& name) const {
    return _macros.find(name) != _macros.end();
}

// Simple macro expansion (production would handle function-like macros, ## and # operators)
std::vector<CToken> CPreprocessor::expand(const std::vector<CToken>& tokens) {
    std::vector<CToken> result;

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto& tok = tokens[i];

        // Check if this identifier is a macro
        if (tok.type == CTokenType::IDENTIFIER) {
            auto it = _macros.find(tok.text);
            if (it != _macros.end()) {
                // Replace with macro value
                // Production: would tokenize macro value and handle recursive expansion
                CToken expanded(CTokenType::INTEGER_LITERAL, it->second, tok.filename, tok.line, tok.column);
                result.push_back(expanded);
                continue;
            }
        }

        result.push_back(tok);
    }

    return result;
}

std::vector<CToken> CPreprocessor::processInclude(const std::string& filename) {
    // Find file in include paths
    for (const auto& path : _includePaths) {
        std::string fullPath = path + "/" + filename;
        std::ifstream file(fullPath);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string contents = buffer.str();

            // Tokenize included file
            CLexer lexer(contents, fullPath);
            return lexer.tokenize();
        }
    }

    // Try current directory
    std::ifstream file(filename);
    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string contents = buffer.str();

        CLexer lexer(contents, filename);
        return lexer.tokenize();
    }

    // File not found - return empty
    return {};
}

bool CPreprocessor::evaluateConditional(const std::string& condition) {
    // Simple constant expression evaluation for preprocessor conditionals
    // Production would need full C constant expression evaluator

    // Handle "defined(MACRO)"
    std::regex definedRegex(R"(defined\s*\(\s*(\w+)\s*\))");
    std::smatch match;
    std::string conditionCopy = condition;

    while (std::regex_search(conditionCopy, match, definedRegex)) {
        std::string macro = match[1].str();
        bool isDef = isDefined(macro);
        conditionCopy = conditionCopy.substr(0, match.position()) +
                        (isDef ? "1" : "0") +
                        conditionCopy.substr(match.position() + match.length());
    }

    // Simple integer literal evaluation
    if (conditionCopy.find_first_not_of("0123456789 \t") == std::string::npos) {
        try {
            int value = std::stoi(conditionCopy);
            return value != 0;
        } catch (...) {
            return false;
        }
    }

    // Check for simple macro substitution
    if (conditionCopy.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == std::string::npos) {
        auto it = _macros.find(conditionCopy);
        if (it != _macros.end()) {
            try {
                int value = std::stoi(it->second);
                return value != 0;
            } catch (...) {
                return true;  // Defined but not integer = true
            }
        }
        return false;  // Undefined = false
    }

    // Simple operators: && || ! == != < > <= >=
    // This is a minimal implementation - production needs full expression parser

    // Check for simple comparisons
    if (conditionCopy.find("==") != std::string::npos) {
        auto pos = conditionCopy.find("==");
        std::string left = conditionCopy.substr(0, pos);
        std::string right = conditionCopy.substr(pos + 2);
        // Trim whitespace
        left.erase(0, left.find_first_not_of(" \t"));
        left.erase(left.find_last_not_of(" \t") + 1);
        right.erase(0, right.find_first_not_of(" \t"));
        right.erase(right.find_last_not_of(" \t") + 1);
        return left == right;
    }

    if (conditionCopy.find("!=") != std::string::npos) {
        auto pos = conditionCopy.find("!=");
        std::string left = conditionCopy.substr(0, pos);
        std::string right = conditionCopy.substr(pos + 2);
        left.erase(0, left.find_first_not_of(" \t"));
        left.erase(left.find_last_not_of(" \t") + 1);
        right.erase(0, right.find_first_not_of(" \t"));
        right.erase(right.find_last_not_of(" \t") + 1);
        return left != right;
    }

    // Default: undefined behavior = false
    return false;
}

} // namespace cppfort::c

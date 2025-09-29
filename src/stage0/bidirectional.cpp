#include "bidirectional.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "lexer.h"
#include "parser.h"

namespace cppfort::stage0 {

// Forward declarations for internal classes
class BidirectionalTranspiler::Cpp2Parser {
public:
    Cpp2Parser() = default;
    TranslationUnit parse(const std::string& source, const std::string& filename);
};

class BidirectionalTranspiler::CppParser {
public:
    CppParser() = default;
    TranslationUnit parse(const std::string& source, const std::string& filename);
};

class BidirectionalTranspiler::Cpp2Emitter {
public:
    Cpp2Emitter() = default;
    std::string emit(const TranslationUnit& unit, const TransformOptions& options);
};

class BidirectionalTranspiler::CppEmitter {
public:
    CppEmitter() = default;
    std::string emit(const TranslationUnit& unit, const TransformOptions& options);
};

// BidirectionalTranspiler implementation

BidirectionalTranspiler::BidirectionalTranspiler()
    : m_cpp2_parser(std::make_shared<Cpp2Parser>()),
      m_cpp_parser(std::make_shared<CppParser>()),
      m_cpp2_emitter(std::make_shared<Cpp2Emitter>()),
      m_cpp_emitter(std::make_shared<CppEmitter>()) {
}

TranslationUnit BidirectionalTranspiler::parse_cpp2(const std::string& source, const std::string& filename) {
    return m_cpp2_parser->parse(source, filename);
}

TranslationUnit BidirectionalTranspiler::parse_cpp(const std::string& source, const std::string& filename) {
    return m_cpp_parser->parse(source, filename);
}

std::string BidirectionalTranspiler::emit_cpp2(const TranslationUnit& unit, const TransformOptions& options) {
    return m_cpp2_emitter->emit(unit, options);
}

std::string BidirectionalTranspiler::emit_cpp(const TranslationUnit& unit, const TransformOptions& options) {
    return m_cpp_emitter->emit(unit, options);
}

std::string BidirectionalTranspiler::roundtrip_cpp2(const std::string& source, const std::string& filename) {
    auto ast = parse_cpp2(source, filename);
    TransformOptions options;
    options.target_cpp2 = true;
    return emit_cpp2(ast, options);
}

std::string BidirectionalTranspiler::roundtrip_cpp(const std::string& source, const std::string& filename) {
    auto ast = parse_cpp(source, filename);
    TransformOptions options;
    options.target_cpp2 = false;
    return emit_cpp(ast, options);
}

// Cpp2Parser implementation - uses robust lexer/parser
TranslationUnit BidirectionalTranspiler::Cpp2Parser::parse(const std::string& source, const std::string& filename) {
    try {
        Lexer lexer(source, filename);
        auto tokens = lexer.tokenize();

        Parser parser(std::move(tokens), source);
        return parser.parse();
    } catch (const LexError& e) {
        throw std::runtime_error("Lexer error in " + filename + ": " + e.what());
    } catch (const ParseError& e) {
        throw std::runtime_error("Parse error in " + filename + ": " + e.what());
    }
}

// CppParser implementation - basic C++ parsing (placeholder)
TranslationUnit BidirectionalTranspiler::CppParser::parse(const std::string& source, const std::string& filename) {
    // For now, implement basic C++ main function parsing
    // TODO: Implement robust C++ parser similar to Cpp2Parser
    TranslationUnit unit;

    // Very basic C++ main function detection (no regex)
    auto main_pos = source.find("int main()");
    if (main_pos != std::string::npos) {
        auto brace_start = source.find('{', main_pos);
        if (brace_start != std::string::npos) {
            auto brace_end = source.find('}', brace_start);
            if (brace_end != std::string::npos) {
                FunctionDecl main_func;
                main_func.name = "main";
                main_func.return_type = "int";

                std::string body_content = source.substr(brace_start + 1, brace_end - brace_start - 1);
                Block block;

                // Very basic statement parsing by lines
                std::istringstream iss(body_content);
                std::string line;
                while (std::getline(iss, line)) {
                    line.erase(line.begin(), std::find_if(line.begin(), line.end(),
                        [](unsigned char ch) { return !std::isspace(ch); }));
                    line.erase(std::find_if(line.rbegin(), line.rend(),
                        [](unsigned char ch) { return !std::isspace(ch); }).base(), line.end());

                    if (line.find("return") == 0) {
                        ReturnStmt ret;
                        auto ret_pos = line.find("return");
                        if (ret_pos != std::string::npos) {
                            std::string expr = line.substr(ret_pos + 6);
                            expr.erase(expr.begin(), std::find_if(expr.begin(), expr.end(),
                                [](unsigned char ch) { return !std::isspace(ch); }));
                            expr.erase(std::find_if(expr.rbegin(), expr.rend(),
                                [](unsigned char ch) { return !std::isspace(ch); }).base(), expr.end());
                            if (!expr.empty() && expr.back() == ';') {
                                expr.pop_back();
                            }
                            if (!expr.empty()) {
                                ret.expression = expr;
                            }
                        }
                        block.statements.push_back(std::move(ret));
                    } else if (!line.empty() && line.back() == ';') {
                        ExpressionStmt expr;
                        expr.expression = line.substr(0, line.size() - 1);
                        block.statements.push_back(std::move(expr));
                    }
                }

                main_func.body = block;
                unit.functions.push_back(main_func);
            }
        }
    }

    return unit;
}

// Cpp2Emitter implementation
std::string BidirectionalTranspiler::Cpp2Emitter::emit(const TranslationUnit& unit, const TransformOptions& options) {
    std::string output;

    for (const auto& fn : unit.functions) {
        if (fn.name == "main") {
            output += "main: ()";
            if (fn.return_type) {
                output += " -> " + *fn.return_type;
            }
            output += " = {\n";
            if (std::holds_alternative<Block>(fn.body)) {
                const auto& block = std::get<Block>(fn.body);
                for (const auto& stmt : block.statements) {
                    std::visit([&output](const auto& s) {
                        using T = std::decay_t<decltype(s)>;
                        if constexpr (std::is_same_v<T, ExpressionStmt>) {
                            output += "    " + s.expression + "\n";
                        } else if constexpr (std::is_same_v<T, ReturnStmt>) {
                            output += "    return";
                            if (s.expression) {
                                output += " " + *s.expression;
                            }
                            output += "\n";
                        }
                    }, stmt);
                }
            }
            output += "}\n";
        }
    }

    return output;
}

// CppEmitter implementation - simplified version
std::string BidirectionalTranspiler::CppEmitter::emit(const TranslationUnit& unit, const TransformOptions& options) {
    std::string output = "#include <iostream>\n\n";

    for (const auto& fn : unit.functions) {
        // Emit function signature
        std::string ret;
        if (fn.return_type) {
            ret = *fn.return_type;
        } else if (fn.name == "main") {
            ret = "int";
        } else {
            ret = "void";
        }
        // Convert cpp2 name and parameters to C++ signature
        std::string sig = ret + " " + fn.name + "(";
        for (size_t i = 0; i < fn.parameters.size(); ++i) {
            const auto& p = fn.parameters[i];
            sig += p.type + " " + p.name;
            if (i + 1 < fn.parameters.size()) sig += ", ";
        }
        sig += ") {\n";

        output += sig;

        if (std::holds_alternative<Block>(fn.body)) {
            const auto& block = std::get<Block>(fn.body);
            for (const auto& stmt : block.statements) {
                std::visit([&output](const auto& s) {
                    using T = std::decay_t<decltype(s)>;

                    auto emit_line = [&](std::string line) {
                        // Trim whitespace
                        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front()))) line.erase(line.begin());
                        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) line.pop_back();
                        // Remove trailing semicolons to avoid double-';'
                        if (!line.empty() && line.back() == ';') line.pop_back();
                        output += "    " + line + ";\n";
                    };

                    if constexpr (std::is_same_v<T, ExpressionStmt>) {
                        std::string expr = s.expression;
                        // Trim and strip semicolons
                        while (!expr.empty() && std::isspace(static_cast<unsigned char>(expr.front()))) expr.erase(expr.begin());
                        while (!expr.empty() && std::isspace(static_cast<unsigned char>(expr.back()))) expr.pop_back();
                        while (!expr.empty() && expr.back() == ';') expr.pop_back();

                        // Detect cpp2 var decl (basic check for ':')
                        auto colon_pos = expr.find(':');
                        if (colon_pos != std::string::npos) {
                            std::string name = expr.substr(0, colon_pos);
                            std::string rest = expr.substr(colon_pos + 1);
                            auto equals_pos = rest.find('=');
                            std::string type;
                            std::string init;
                            if (equals_pos != std::string::npos) {
                                type = rest.substr(0, equals_pos);
                                init = rest.substr(equals_pos + 1);
                            } else {
                                type = rest;
                            }
                            // trim
                            while (!name.empty() && std::isspace(static_cast<unsigned char>(name.front()))) name.erase(name.begin());
                            while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back()))) name.pop_back();
                            while (!type.empty() && std::isspace(static_cast<unsigned char>(type.front()))) type.erase(type.begin());
                            while (!type.empty() && std::isspace(static_cast<unsigned char>(type.back()))) type.pop_back();
                            while (!init.empty() && std::isspace(static_cast<unsigned char>(init.front()))) init.erase(init.begin());
                            while (!init.empty() && std::isspace(static_cast<unsigned char>(init.back()))) init.pop_back();
                            if (type.empty()) type = "auto";
                            std::string out_line = type + " " + name;
                            if (!init.empty()) out_line += " = " + init;
                            emit_line(std::move(out_line));
                        } else {
                            emit_line(expr);
                        }
                    } else if constexpr (std::is_same_v<T, ReturnStmt>) {
                        std::string line = "return";
                        if (s.expression) {
                            line += " " + *s.expression;
                        }
                        emit_line(line);
                    }
                }, stmt);
            }
        }

        output += "}\n";
    }

    return output;
}

} // namespace cppfort::stage0
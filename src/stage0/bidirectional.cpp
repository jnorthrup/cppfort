#include "bidirectional.h"

#include <cctype>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "lexer.h"
#include "parser.h"

namespace cppfort::stage0 {

namespace {

bool looks_like_initializer_list(const std::string& expr) {
    if (expr.size() < 2 || expr.front() != '(' || expr.back() != ')') {
        return false;
    }

    std::size_t depth = 0;
    bool has_top_level_comma = false;
    for (std::size_t i = 1; i + 1 < expr.size(); ++i) {
        char c = expr[i];
        if (c == '(') {
            ++depth;
        } else if (c == ')') {
            if (depth == 0) {
                return false;
            }
            --depth;
        } else if (c == ',' && depth == 0) {
            has_top_level_comma = true;
        }
    }

    return depth == 0 && has_top_level_comma;
}

std::string fix_expression_tokens(std::string expr) {
    for (size_t pos = 0; pos + 1 < expr.size(); ++pos) {
        if (std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_') {
            size_t start = pos;
            while (pos < expr.size() && (std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_')) {
                ++pos;
            }
            if (pos < expr.size() && expr[pos] == '*') {
                std::string ident = expr.substr(start, pos - start);
                expr.replace(start, ident.size() + 1, std::string("*") + ident);
                pos = start + 1 + ident.size();
            }
        }
    }

    for (size_t pos = 0; pos + 1 < expr.size(); ++pos) {
        if ((std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_') && expr[pos + 1] == '&') {
            size_t end = pos;
            size_t start = pos;
            while (start > 0 && (std::isalnum(static_cast<unsigned char>(expr[start - 1])) || expr[start - 1] == '_')) {
                --start;
            }
            std::string ident = expr.substr(start, end - start + 1);
            expr.replace(start, ident.size() + 1, std::string("&") + ident);
            pos = start + 1 + ident.size();
        }
    }

    static const std::unordered_set<std::string> k_c_ufcs_names {
        "fprintf", "fclose", "fflush", "fread", "fwrite", "fgets", "fputs",
        "fscanf", "fgetc", "fputc", "fopen", "freopen"
    };

    for (size_t pos = 0; pos + 2 < expr.size(); ++pos) {
        if (!std::isalnum(static_cast<unsigned char>(expr[pos])) && expr[pos] != '_') {
            continue;
        }

        size_t obj_start = pos;
        while (pos < expr.size() && (std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_')) {
            ++pos;
        }

        if (pos + 1 >= expr.size() || expr[pos] != '.') {
            continue;
        }

        if (!std::isalnum(static_cast<unsigned char>(expr[pos + 1])) && expr[pos + 1] != '_') {
            continue;
        }

        size_t method_start = pos + 1;
        size_t method_end = method_start;
        while (method_end < expr.size() && (std::isalnum(static_cast<unsigned char>(expr[method_end])) || expr[method_end] == '_')) {
            ++method_end;
        }

        if (method_end >= expr.size() || expr[method_end] != '(') {
            continue;
        }

        std::string method = expr.substr(method_start, method_end - method_start);
        if (!k_c_ufcs_names.contains(method)) {
            continue;
        }

        size_t paren_count = 1;
        size_t args_end = method_end + 1;
        while (args_end < expr.size() && paren_count > 0) {
            if (expr[args_end] == '(') {
                ++paren_count;
            } else if (expr[args_end] == ')') {
                --paren_count;
            }
            ++args_end;
        }

        if (args_end > expr.size()) {
            break;
        }

        std::string obj = expr.substr(obj_start, pos - obj_start);
        std::string args = expr.substr(method_end + 1, args_end - method_end - 2);
        std::string replacement = method + "(" + obj;
        if (!args.empty()) {
            replacement += ", " + args;
        }
        replacement += ")";

        expr.replace(obj_start, args_end - obj_start, replacement);
        pos = obj_start + replacement.size();
    }

    if (expr.size() >= 4 && expr.substr(0, 4) == "_ = ") {
        expr = "(void)(" + expr.substr(4) + ")";
    }

    return expr;
}

std::string normalize_space(std::string text) {
    std::string result;
    bool in_space = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!in_space) {
                result.push_back(' ');
                in_space = true;
            }
        } else {
            result.push_back(c);
            in_space = false;
        }
    }
    return result;
}

} // namespace

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

                // More robust statement parsing: scan the function body and
                // collect top-level statements while tracking nested
                // parentheses/braces/brackets/angles and string/char literals.
                auto trim = [](std::string& s) {
                    // trim in place
                    auto not_ws = [](unsigned char ch){ return !std::isspace(ch); };
                    while (!s.empty() && !not_ws(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
                    while (!s.empty() && !not_ws(static_cast<unsigned char>(s.back()))) s.pop_back();
                };

                const std::string& src = body_content;
                size_t n = src.size();
                size_t i = 0;
                while (i < n) {
                    // skip leading whitespace and newlines
                    while (i < n && std::isspace(static_cast<unsigned char>(src[i]))) ++i;
                    if (i >= n) break;

                    size_t start = i;
                    int paren = 0, brace = 0, bracket = 0, angle = 0;
                    bool in_string = false;
                    char string_delim = '\0';
                    bool saw_open_brace = false;

                    for (; i < n; ++i) {
                        char c = src[i];

                        if (in_string) {
                            if (c == '\\') {
                                // skip escaped char
                                ++i;
                                continue;
                            }
                            if (c == string_delim) {
                                in_string = false;
                                string_delim = '\0';
                            }
                            continue;
                        }

                        if (c == '"' || c == '\'') {
                            in_string = true;
                            string_delim = c;
                            continue;
                        }

                        if (c == '(') { ++paren; continue; }
                        if (c == ')') { if (paren > 0) --paren; continue; }
                        if (c == '{') { ++brace; saw_open_brace = true; continue; }
                        if (c == '}') { if (brace > 0) --brace; if (brace == 0 && saw_open_brace) { ++i; break; } continue; }
                        if (c == '[') { ++bracket; continue; }
                        if (c == ']') { if (bracket > 0) --bracket; continue; }
                        if (c == '<') { ++angle; continue; }
                        if (c == '>') { if (angle > 0) --angle; continue; }

                        // semicolon ends a top-level statement when depths are zero
                        if (c == ';' && paren == 0 && brace == 0 && bracket == 0 && angle == 0) {
                            ++i; // include semicolon
                            break;
                        }
                    }

                    std::string stmt = src.substr(start, i - start);
                    trim(stmt);
                    if (stmt.empty()) continue;

                    // Handle 'return' specially
                    auto starts_with = [](const std::string& s, const std::string& pref){
                        if (s.size() < pref.size()) return false;
                        return s.compare(0, pref.size(), pref) == 0;
                    };

                    if (starts_with(stmt, "return") && (stmt.size() == 6 || std::isspace(static_cast<unsigned char>(stmt[6])) || stmt[6] == ';')) {
                        // strip leading 'return'
                        std::string rest = stmt.substr(6);
                        trim(rest);
                        if (!rest.empty() && rest.back() == ';') rest.pop_back();
                        ReturnStmt ret;
                        if (!rest.empty()) ret.expression = rest;
                        block.statements.push_back(std::move(ret));
                    } else {
                        // Push everything else as an ExpressionStmt; remove trailing ';' if present
                        if (!stmt.empty() && stmt.back() == ';') stmt.pop_back();
                        ExpressionStmt expr;
                        expr.expression = stmt;
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
                        if constexpr (std::is_same_v<T, VariableDecl>) {
                            output += "    " + s.name;
                            if (!s.type.empty()) {
                                output += ": " + s.type;
                            }
                            if (s.initializer && !s.initializer->empty()) {
                                output += " = " + *s.initializer;
                            }
                            output += "\n";
                        } else if constexpr (std::is_same_v<T, ExpressionStmt>) {
                            output += "    " + s.expression + "\n";
                        } else if constexpr (std::is_same_v<T, ReturnStmt>) {
                            output += "    return";
                            if (s.expression) {
                                output += " " + *s.expression;
                    }
                    output += "\n";
                }
                else if constexpr (std::is_same_v<T, AssertStmt>) {
                    output += "    assert " + s.condition;
                    if (s.category && !s.category->empty()) {
                        output += " // " + *s.category;
                    }
                    output += "\n";
                } else if constexpr (std::is_same_v<T, ForChainStmt>) {
                    output += "    // for-chain loop not yet supported in cpp2 emitter\n";
                } else if constexpr (std::is_same_v<T, RawStmt>) {
                    output += "    " + s.text + "\n";
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
    std::string output;

    auto needs_cstdio = false;

    auto scan_for_stdio = [&](const std::string& text) {
        if (text.find("fopen") != std::string::npos ||
            text.find("fprintf") != std::string::npos ||
            text.find("fclose") != std::string::npos ||
            text.find("fread") != std::string::npos ||
            text.find("fwrite") != std::string::npos) {
            needs_cstdio = true;
        }
    };

    std::function<void(const Statement&)> scan_statement;
    scan_statement = [&](const Statement& stmt) {
        std::visit([&](const auto& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, VariableDecl>) {
                if (node.initializer) {
                    scan_for_stdio(*node.initializer);
                }
            } else if constexpr (std::is_same_v<T, ExpressionStmt>) {
                scan_for_stdio(node.expression);
            } else if constexpr (std::is_same_v<T, ReturnStmt>) {
                if (node.expression) {
                    scan_for_stdio(*node.expression);
                }
            } else if constexpr (std::is_same_v<T, AssertStmt>) {
                scan_for_stdio(node.condition);
                if (node.category) {
                    scan_for_stdio(*node.category);
                }
            } else if constexpr (std::is_same_v<T, ForChainStmt>) {
                scan_for_stdio(node.range_expression);
                if (node.next_expression) {
                    scan_for_stdio(*node.next_expression);
                }
                for (const auto& inner : node.body.statements) {
                    scan_statement(inner);
                }
            } else if constexpr (std::is_same_v<T, RawStmt>) {
                scan_for_stdio(node.text);
            }
        }, stmt);
    };

    for (const auto& fn : unit.functions) {
        if (std::holds_alternative<Block>(fn.body)) {
            const auto& block = std::get<Block>(fn.body);
            for (const auto& stmt : block.statements) {
                scan_statement(stmt);
            }
        } else if (std::holds_alternative<ExpressionBody>(fn.body)) {
            const auto& expr_body = std::get<ExpressionBody>(fn.body);
            scan_for_stdio(expr_body.expression);
        }
    }

    auto append_line = [&](int indent, const std::string& text) {
        constexpr int spaces_per_indent = 4;
        output.append(static_cast<std::size_t>(indent * spaces_per_indent), ' ');
        output.append(text);
        output.push_back('\n');
    };

    std::function<void(const Statement&, int)> emit_statement_cpp;
    emit_statement_cpp = [&](const Statement& stmt, int indent) {
        std::visit([&](const auto& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, VariableDecl>) {
                std::string type = normalize_space(s.type);
                if (type.empty()) type = "auto";
                std::string line = type + " " + s.name;
                if (s.initializer && !s.initializer->empty()) {
                    auto init = fix_expression_tokens(normalize_space(*s.initializer));
                    if (looks_like_initializer_list(init)) {
                        init.front() = '{';
                        init.back() = '}';
                    }
                    line += " = " + init;
                }
                append_line(indent, line + ';');
            } else if constexpr (std::is_same_v<T, ExpressionStmt>) {
                append_line(indent, fix_expression_tokens(normalize_space(s.expression)) + ';');
            } else if constexpr (std::is_same_v<T, ReturnStmt>) {
                if (s.expression && !s.expression->empty()) {
                    append_line(indent, "return " + fix_expression_tokens(normalize_space(*s.expression)) + ';');
                } else {
                    append_line(indent, "return;");
                }
            } else if constexpr (std::is_same_v<T, AssertStmt>) {
                std::string line = "assert(" + normalize_space(s.condition) + ");";
                if (s.category && !s.category->empty()) {
                    line += " // " + *s.category;
                }
                append_line(indent, line);
            } else if constexpr (std::is_same_v<T, ForChainStmt>) {
                auto resolve_type = [&]() {
                    if (!s.loop_parameter.type.empty()) {
                        return normalize_space(s.loop_parameter.type);
                    }
                    switch (s.loop_parameter.kind) {
                        case ParameterKind::In:
                            return std::string("const auto&");
                        case ParameterKind::InOut:
                        case ParameterKind::Out:
                            return std::string("auto&");
                        case ParameterKind::Move:
                        case ParameterKind::Forward:
                            return std::string("auto&&");
                        default:
                            return std::string("auto");
                    }
                };

                const auto loop_type = resolve_type();
                const auto loop_var = s.loop_parameter.name.empty() ? std::string("item") : s.loop_parameter.name;
                const auto range = fix_expression_tokens(normalize_space(s.range_expression));

                append_line(indent, "for (" + loop_type + " " + loop_var + " : " + range + ") {");
                for (const auto& inner : s.body.statements) {
                    emit_statement_cpp(inner, indent + 1);
                }
                if (s.next_expression) {
                    append_line(indent + 1, fix_expression_tokens(normalize_space(*s.next_expression)) + ";");
                }
                append_line(indent, "}");
            } else if constexpr (std::is_same_v<T, RawStmt>) {
                append_line(indent, s.text);
            }
        }, stmt);
    };

    if (options.include_preamble) {
        output += "// Generated by cppfort stage0 transpiler\n";
        output += "#include <cstdint>\n";
        output += "#include <iostream>\n";
        output += "#include <string>\n";
        output += "#include <string_view>\n";
        output += "#include <vector>\n";
        if (needs_cstdio) {
            output += "#include <cstdio>\n";
        }
        output.push_back('\n');
    }

    for (const auto& inc : unit.includes) {
        std::string line = "#include ";
        if (inc.is_system) {
            line += std::string("<") + inc.path + ">";
        } else {
            line += std::string("\"") + inc.path + "\"";
        }
        append_line(0, line);
    }

    if (!unit.includes.empty()) {
        output.push_back('\n');
    }

    for (const auto& raw : unit.raw_declarations) {
        if (!raw.text.empty()) {
            if (!output.empty() && output.back() != '\n') {
                output.push_back('\n');
            }
            output += raw.text;
            if (output.back() != '\n') {
                output.push_back('\n');
            }
            output.push_back('\n');
        }
    }

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
        auto emit_param_type = [&](const Parameter& p) {
            std::string type = p.type;
            if (type.empty()) type = "auto";
            switch (p.kind) {
                case ParameterKind::In:
                    return std::string("cpp2::impl::in<") + type + ">";
                case ParameterKind::InOut:
                case ParameterKind::Out:
                    return type + "&";
                case ParameterKind::Copy:
                    return std::string("cpp2::impl::copy<") + type + ">";
                case ParameterKind::Move:
                    return std::string("cpp2::impl::move<") + type + ">";
                case ParameterKind::Forward:
                    return type + "&&";
                default:
                    return type;
            }
        };

        std::string sig = ret + " " + fn.name + "(";
        for (size_t i = 0; i < fn.parameters.size(); ++i) {
            const auto& p = fn.parameters[i];
            sig += emit_param_type(p) + " " + p.name;
            if (i + 1 < fn.parameters.size()) sig += ", ";
        }
        sig += ") {";

        append_line(0, sig);

        if (std::holds_alternative<Block>(fn.body)) {
            const auto& block = std::get<Block>(fn.body);
            for (const auto& stmt : block.statements) {
                emit_statement_cpp(stmt, 1);
            }
        }

        append_line(0, "}");
        output.push_back('\n');
    }

    return output;
}

} // namespace cppfort::stage0

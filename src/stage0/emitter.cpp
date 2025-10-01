#include "emitter.h"

#include <cctype>
#include <functional>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <unordered_set>
#include <utility>

namespace cppfort::stage0 {
namespace {
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
    if (!result.empty() && result.front() == ' ') {
        result.erase(result.begin());
    }
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result;
}

std::string trim_copy(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    return text;
}

// Fix simple cpp2 type/expr idioms into valid C++ forms.
// Examples:
//  - type: "*std::string" -> "std::string*"
//  - expr: "s&" -> "&s"
//  - expr: "ps* = ..." -> "*ps = ..."
//  - simple UFCS pattern: ":cout << ... std" -> "std::cout << ..."
std::string fix_prefix_type(std::string type) {
    // move leading '*' and '&' to the end
    std::string prefix;
    while (!type.empty() && (type.front() == '*' || type.front() == '&')) {
        prefix.push_back(type.front());
        type.erase(type.begin());
    }
    // append in reverse order for readability: '*' then '&'
    for (char c : prefix) {
        if (c == '*') type += '*';
        else if (c == '&') type += '&';
    }
    return type;
}

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
                // closing parenthesis before the end means grouped expression
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
    // Simple postfix deref: identifier* -> *identifier
    // Use a manual scan to avoid regex dependency
    for (size_t pos = 0; pos + 1 < expr.size(); ++pos) {
        if (std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_') {
            // find end of ident
            size_t start = pos;
            while (pos < expr.size() && (std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_')) ++pos;
            if (pos < expr.size() && expr[pos] == '*') {
                std::string ident = expr.substr(start, pos - start);
                // replace ident* with *ident
                expr.replace(start, ident.size() + 1, std::string("*") + ident);
                // continue after the inserted sequence
                pos = start + 1 + ident.size();
            }
        }
    }

    // Simple address-of suffix: something& -> &something (only for simple tokens)
    for (size_t pos = 0; pos + 1 < expr.size(); ++pos) {
        if ((std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_') && expr[pos+1] == '&') {
            // find start of ident
            size_t end = pos;
            size_t start = pos;
            while (start > 0 && (std::isalnum(static_cast<unsigned char>(expr[start-1])) || expr[start-1] == '_')) --start;
            std::string ident = expr.substr(start, end - start + 1);
            // replace ident& with &ident
            expr.replace(start, ident.size() + 1, std::string("&") + ident);
            pos = start + 1 + ident.size();
        }
    }

    // Simple UFCS pattern: find a ' std' token and a preceding ':id'
    size_t std_pos = expr.find(" std");
    if (std_pos != std::string::npos) {
        // find last ':' before std_pos
        size_t col_pos = expr.rfind(':', std_pos);
        if (col_pos != std::string::npos && col_pos + 1 < expr.size()) {
            // read identifier after ':'
            size_t id_start = col_pos + 1;
            size_t id_end = id_start;
            while (id_end < expr.size() && (std::isalnum(static_cast<unsigned char>(expr[id_end])) || expr[id_end] == '_')) ++id_end;
            if (id_end > id_start) {
                std::string id = expr.substr(id_start, id_end - id_start);
                // remove the ':id' and the ' std' and replace with 'std.id'
                expr.erase(std_pos, 4); // remove " std"
                expr.erase(col_pos, id.size() + 1); // remove ":id"
                expr.insert(col_pos, std::string("std::") + id);
            }
        }
    }

    // UFCS for selected C stdio-like functions: obj.fprintf(args) -> fprintf(obj, args)
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

        // find the end of the argument list
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

    // Handle discard pattern: _ = expr -> (void)expr
    if (expr.size() >= 4 && expr.substr(0, 4) == "_ = ") {
        expr = "(void)(" + expr.substr(4) + ")";
    }

    return expr;
}
}

// ... existing code ...
std::string Emitter::emit(const TranslationUnit& unit, const EmitOptions& options) const {
    switch (options.backend) {
        case EmitBackend::Cpp:
            return emit_cpp(unit, options);
        case EmitBackend::Mlir:
            return emit_mlir(unit, options);
    }
    return {};
}

std::string Emitter::emit_cpp(const TranslationUnit& unit, const EmitOptions& options) const {
// ... existing code ...

            expr.find("fclose") != std::string::npos ||
            expr.find("fread") != std::string::npos ||
            expr.find("fwrite") != std::string::npos) {
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
            } else if constexpr (std::is_same_v<T, ForChainStmt>) {
                scan_for_stdio(node.range_expression);
                if (node.next_expression) {
                    scan_for_stdio(*node.next_expression);
                }
                for (const auto& inner : node.body.statements) {
                    scan_statement(inner);
                }
            } else if constexpr (std::is_same_v<T, AssertStmt>) {
                scan_for_stdio(node.condition);
                if (node.category) {
                    scan_for_stdio(*node.category);
                }
            } else if constexpr (std::is_same_v<T, RawStmt>) {
                scan_for_stdio(node.text);
            }
        }, stmt);
    };

    // Scan function bodies
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
        output += "\n";
        // Cpp2 type aliases
        output += "using i8 = std::int8_t;\n";
        output += "using i16 = std::int16_t;\n";
        output += "using i32 = std::int32_t;\n";
        output += "using i64 = std::int64_t;\n";
        output += "using u8 = std::uint8_t;\n";
        output += "using u16 = std::uint16_t;\n";
        output += "using u32 = std::uint32_t;\n";
        output += "using u64 = std::uint64_t;\n\n";
    }

    bool first = true;
    for (const auto& type : unit.types) {
        if (!first) {
            output += '\n';
        }
        emit_type(type, output, 0);
        first = false;
    }

    for (const auto& fn : unit.functions) {
        if (!first) {
            output += '\n';
        }
        emit_function(fn, output, 0);
        first = false;
    }

    return output;
std::string Emitter::emit_mlir(const TranslationUnit& unit, const EmitOptions& /*options*/) const {
    using namespace cppfort::ir;

    // Initialize MLIR infrastructure up-front. We rely on textual assembly
    // emission for now; in a follow-up we can wire in MLIRContext.
    std::ostringstream os;
    os << "module {\n";

    PatternMatcher matcher;

    // Emit each function as a MLIR func.func with simple lowering. For now we
    // only support expression bodies and basic arithmetic statements, but the
    // matcher provides the extension point to grow coverage.
    for (const auto& fn : unit.functions) {
        const std::string functionName = fn.name.empty() ? "anon" : fn.name;
        os << "  func.func @" << functionName << "() -> i32 {\n";

        if (std::holds_alternative<Block>(fn.body)) {
            const auto& block = std::get<Block>(fn.body);
            for (const auto& stmt : block.statements) {
                std::visit(
                    [&](const auto& node) {
                        using T = std::decay_t<decltype(node)>;
                        if constexpr (std::is_same_v<T, ExpressionStmt>) {
                            // Build a synthetic Sea-of-Nodes representation so
                            // we can invoke the pattern matcher. For now we
                            // treat raw expression text as opaque MLIR.
                            os << "    " << node.expression << "\n";
                        } else if constexpr (std::is_same_v<T, ReturnStmt>) {
                            if (node.expression) {
                                os << "    return " << *node.expression << " : i32\n";
                            } else {
                                os << "    return \n";
                            }
                        } else {
                            // TODO: Expand support by threading nodes through
                            // MLIR emitter once the Sea-of-Nodes bridge lands.
                        }
                    },
                    stmt);
            }
        } else if (std::holds_alternative<ExpressionBody>(fn.body)) {
            const auto& expr = std::get<ExpressionBody>(fn.body);
            os << "    %0 = " << expr.expression << "\n";
            os << "    return %0 : i32\n";
        }

        os << "  }\n";
    }

    os << "}\n";
    return os.str();
}
// ... existing code ...

            }
            line += ';';
            Emitter::append_line(out, line, indent);
        } else if constexpr (std::is_same_v<T, ExpressionStmt>) {
            Emitter::append_line(out, fix_expression_tokens(normalize_space(node.expression)) + ';', indent);
        } else if constexpr (std::is_same_v<T, ReturnStmt>) {
            if (node.expression.has_value() && !node.expression->empty()) {
                Emitter::append_line(out, "return " + normalize_space(*node.expression) + ';', indent);
            } else {
                Emitter::append_line(out, "return;", indent);
            }
        } else if constexpr (std::is_same_v<T, AssertStmt>) {
            std::string line = "assert(" + normalize_space(node.condition) + ");";
            if (node.category && !node.category->empty()) {
                line += " // " + *node.category;
            }
            Emitter::append_line(out, line, indent);
        } else if constexpr (std::is_same_v<T, ForChainStmt>) {
            auto param_type = node.loop_parameter.type;
            const auto kind = node.loop_parameter.kind;

            auto resolve_type = [&]() {
                if (!param_type.empty()) {
                    return normalize_space(param_type);
                }
                switch (kind) {
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
            const auto loop_var = node.loop_parameter.name.empty() ? std::string("item") : node.loop_parameter.name;
            const auto range = fix_expression_tokens(normalize_space(node.range_expression));

            Emitter::append_line(out,
                "for (" + loop_type + " " + loop_var + " : " + range + ") {", indent);
            for (const auto& inner : node.body.statements) {
                emit_statement(inner, out, indent + 1);
            }
            if (node.next_expression) {
                Emitter::append_line(out, fix_expression_tokens(normalize_space(*node.next_expression)) + ";", indent + 1);
            }
            Emitter::append_line(out, "}", indent);
        } else if constexpr (std::is_same_v<T, RawStmt>) {
            Emitter::append_line(out, node.text, indent);
        }
    }, stmt);
}

void Emitter::append_line(std::string& out, std::string_view text, int indent) {
    out.append(indent_string(indent));
    out.append(text);
    out.push_back('\n');
}

std::string Emitter::indent_string(int indent) {
    constexpr int spaces_per_indent = 4;
    return std::string(static_cast<std::size_t>(indent * spaces_per_indent), ' ');
}

} // namespace cppfort::stage0

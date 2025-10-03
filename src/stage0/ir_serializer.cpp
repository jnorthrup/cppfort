#include "ir_serializer.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <vector>

namespace cppfort::stage0 {

namespace {
    // Helper to split string by delimiter
    ::std::vector<::std::string> split(const ::std::string& str, char delim) {
        ::std::vector<::std::string> result;
        ::std::stringstream ss(str);
        ::std::string item;
        while (::std::getline(ss, item, delim)) {
            result.push_back(item);
        }
        return result;
    }

    // Helper to trim whitespace
    ::std::string trim(const ::std::string& str) {
        auto start = str.find_first_not_of(" \t\n\r");
        if (start == ::std::string::npos) return "";
        auto end = str.find_last_not_of(" \t\n\r");
        return str.substr(start, end - start + 1);
    }
}

::std::string IRSerializer::escape(const ::std::string& str) {
    ::std::string result;
    result.reserve(str.size() + 10);
    for (char c : str) {
        switch (c) {
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            default: result += c; break;
        }
    }
    return result;
}

::std::string IRSerializer::unescape(const ::std::string& str) {
    ::std::string result;
    result.reserve(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
        if (str[i] == '\\' && i + 1 < str.size()) {
            switch (str[i + 1]) {
                case 'n': result += '\n'; ++i; break;
                case 'r': result += '\r'; ++i; break;
                case 't': result += '\t'; ++i; break;
                case '\\': result += '\\'; ++i; break;
                case '"': result += '"'; ++i; break;
                default: result += str[i]; break;
            }
        } else {
            result += str[i];
        }
    }
    return result;
}

::std::string IRSerializer::indent(int level) {
    return ::std::string(level * 2, ' ');
}

::std::string IRSerializer::parameterKindToString(ParameterKind kind) {
    switch (kind) {
        case ParameterKind::Default: return "default";
        case ParameterKind::In: return "in";
        case ParameterKind::InOut: return "inout";
        case ParameterKind::Out: return "out";
        case ParameterKind::Copy: return "copy";
        case ParameterKind::Move: return "move";
        case ParameterKind::Forward: return "forward";
        default: return "default";
    }
}

ParameterKind IRSerializer::stringToParameterKind(const ::std::string& str) {
    if (str == "in") return ParameterKind::In;
    if (str == "inout") return ParameterKind::InOut;
    if (str == "out") return ParameterKind::Out;
    if (str == "copy") return ParameterKind::Copy;
    if (str == "move") return ParameterKind::Move;
    if (str == "forward") return ParameterKind::Forward;
    return ParameterKind::Default;
}

void IRSerializer::serializeParameter(::std::ostringstream& out, const Parameter& param, int ind) {
    out << indent(ind) << "@param " << param.name << " "
        << parameterKindToString(param.kind) << " \""
        << escape(param.type) << "\"\n";
}

void IRSerializer::serializeStatement(::std::ostringstream& out, const Statement& stmt, int ind) {
    ::std::visit([&](const auto& s) {
        using T = ::std::decay_t<decltype(s)>;
        if constexpr (::std::is_same_v<T, VariableDecl>) {
            out << indent(ind) << "@var " << s.name << " \"" << escape(s.type) << "\"";
            if (s.initializer) {
                out << " = \"" << escape(*s.initializer) << "\"";
            }
            out << "\n";
        } else if constexpr (::std::is_same_v<T, ExpressionStmt>) {
            out << indent(ind) << "@expr \"" << escape(s.expression) << "\"\n";
        } else if constexpr (::std::is_same_v<T, ReturnStmt>) {
            out << indent(ind) << "@return";
            if (s.expression) {
                out << " \"" << escape(*s.expression) << "\"";
            }
            out << "\n";
        } else if constexpr (::std::is_same_v<T, AssertStmt>) {
            out << indent(ind) << "@assert \"" << escape(s.condition) << "\"";
            if (s.category) {
                out << " category=\"" << escape(*s.category) << "\"";
            }
            out << "\n";
        } else if constexpr (::std::is_same_v<T, ForChainStmt>) {
            out << indent(ind) << "@for \"" << escape(s.range_expression) << "\"\n";
            serializeParameter(out, s.loop_parameter, ind + 1);
            if (s.next_expression) {
                out << indent(ind + 1) << "@next \"" << escape(*s.next_expression) << "\"\n";
            }
            serializeBlock(out, s.body, ind + 1);
        } else if constexpr (::std::is_same_v<T, RawStmt>) {
            out << indent(ind) << "@raw \"" << escape(s.text) << "\"\n";
        }
    }, stmt);
}

void IRSerializer::serializeBlock(::std::ostringstream& out, const Block& block, int ind) {
    out << indent(ind) << "@block\n";
    for (const auto& stmt : block.statements) {
        serializeStatement(out, stmt, ind + 1);
    }
    out << indent(ind) << "@endblock\n";
}

void IRSerializer::serializeFunction(::std::ostringstream& out, const FunctionDecl& fn, int ind) {
    out << indent(ind) << "@function " << fn.name;
    if (fn.return_type) {
        out << " -> \"" << escape(*fn.return_type) << "\"";
    }
    out << "\n";

    // Serialize parameters
    for (const auto& param : fn.parameters) {
        serializeParameter(out, param, ind + 1);
    }

    // Serialize body
    if (::std::holds_alternative<Block>(fn.body)) {
        serializeBlock(out, ::std::get<Block>(fn.body), ind + 1);
    } else {
        const auto& expr_body = ::std::get<ExpressionBody>(fn.body);
        out << indent(ind + 1) << "@expr_body \"" << escape(expr_body.expression) << "\"\n";
    }

    out << indent(ind) << "@endfunction\n";
}

void IRSerializer::serializeType(::std::ostringstream& out, const TypeDecl& type, int ind) {
    out << indent(ind) << "@type " << type.name << "\n";
    out << indent(ind + 1) << "@body \"" << escape(type.body) << "\"\n";
    out << indent(ind) << "@endtype\n";
}

::std::string IRSerializer::serialize(const TranslationUnit& unit) {
    ::std::ostringstream out;
    out << "@ir_version 1.0\n";
    out << "@translation_unit\n";

    // Serialize includes
    for (const auto& inc : unit.includes) {
        out << indent(1) << "@include " << (inc.is_system ? "<" : "\"")
            << inc.path << (inc.is_system ? ">" : "\"") << "\n";
    }

    // Serialize types
    for (const auto& type : unit.types) {
        serializeType(out, type, 1);
    }

    // Serialize functions
    for (const auto& fn : unit.functions) {
        serializeFunction(out, fn, 1);
    }

    // Serialize raw declarations
    for (const auto& raw : unit.raw_declarations) {
        out << indent(1) << "@raw_decl \"" << escape(raw.text) << "\"\n";
    }

    out << "@end_translation_unit\n";
    return out.str();
}

TranslationUnit IRSerializer::deserialize(const ::std::string& ir_text) {
    TranslationUnit unit;
    auto lines = split(ir_text, '\n');

    ::std::vector<::std::string> context_stack;
    Block* current_block = nullptr;
    FunctionDecl* current_function = nullptr;
    TypeDecl* current_type = nullptr;
    ForChainStmt* current_for = nullptr;

    for (size_t line_num = 0; line_num < lines.size(); ++line_num) {
        const auto& line = lines[line_num];
        auto trimmed = trim(line);

        if (trimmed.empty() || trimmed[0] != '@') continue;

        // Parse directive
        size_t space_pos = trimmed.find(' ');
        ::std::string directive = (space_pos != ::std::string::npos)
            ? trimmed.substr(1, space_pos - 1)
            : trimmed.substr(1);
        ::std::string args = (space_pos != ::std::string::npos)
            ? trim(trimmed.substr(space_pos + 1))
            : "";

        if (directive == "ir_version" || directive == "translation_unit") {
            continue;
        } else if (directive == "include") {
            IncludeDecl inc;
            if (args.size() >= 2 && args[0] == '<') {
                inc.is_system = true;
                auto end = args.find('>');
                inc.path = args.substr(1, end - 1);
            } else if (args.size() >= 2 && args[0] == '"') {
                inc.is_system = false;
                auto end = args.find('"', 1);
                inc.path = args.substr(1, end - 1);
            }
            unit.includes.push_back(inc);
        } else if (directive == "function") {
            unit.functions.emplace_back();
            current_function = &unit.functions.back();

            // Parse name and return type
            auto arrow_pos = args.find(" -> ");
            if (arrow_pos != ::std::string::npos) {
                current_function->name = trim(args.substr(0, arrow_pos));
                auto ret_str = trim(args.substr(arrow_pos + 4));
                if (ret_str.size() >= 2 && ret_str.front() == '"' && ret_str.back() == '"') {
                    current_function->return_type = unescape(ret_str.substr(1, ret_str.size() - 2));
                }
            } else {
                current_function->name = args;
            }
        } else if (directive == "param" && current_function) {
            Parameter param;
            ::std::istringstream iss(args);
            iss >> param.name;

            ::std::string kind_str;
            iss >> kind_str;
            param.kind = stringToParameterKind(kind_str);

            // Rest is the type in quotes
            ::std::string rest;
            ::std::getline(iss, rest);
            rest = trim(rest);
            if (rest.size() >= 2 && rest.front() == '"' && rest.back() == '"') {
                param.type = unescape(rest.substr(1, rest.size() - 2));
            }

            current_function->parameters.push_back(param);
        } else if (directive == "block") {
            Block block;
            if (current_function) {
                current_function->body = block;
                current_block = &::std::get<Block>(current_function->body);
            } else if (current_for) {
                current_for->body = block;
                current_block = &current_for->body;
            }
        } else if (directive == "expr_body" && current_function) {
            ExpressionBody expr_body;
            if (args.size() >= 2 && args.front() == '"' && args.back() == '"') {
                expr_body.expression = unescape(args.substr(1, args.size() - 2));
            }
            current_function->body = expr_body;
        } else if (directive == "var" && current_block) {
            VariableDecl var;

            // Parse: name "type" = "init"
            auto first_quote = args.find('"');
            var.name = trim(args.substr(0, first_quote));

            auto second_quote = args.find('"', first_quote + 1);
            var.type = unescape(args.substr(first_quote + 1, second_quote - first_quote - 1));

            auto equals_pos = args.find('=', second_quote);
            if (equals_pos != ::std::string::npos) {
                auto init_start = args.find('"', equals_pos);
                auto init_end = args.rfind('"');
                if (init_start != ::std::string::npos && init_end != ::std::string::npos) {
                    var.initializer = unescape(args.substr(init_start + 1, init_end - init_start - 1));
                }
            }

            current_block->statements.push_back(var);
        } else if (directive == "expr" && current_block) {
            ExpressionStmt expr;
            if (args.size() >= 2 && args.front() == '"' && args.back() == '"') {
                expr.expression = unescape(args.substr(1, args.size() - 2));
            }
            current_block->statements.push_back(expr);
        } else if (directive == "return" && current_block) {
            ReturnStmt ret;
            if (!args.empty() && args.front() == '"' && args.back() == '"') {
                ret.expression = unescape(args.substr(1, args.size() - 2));
            }
            current_block->statements.push_back(ret);
        } else if (directive == "assert" && current_block) {
            AssertStmt assert_stmt;

            auto first_quote = args.find('"');
            auto second_quote = args.find('"', first_quote + 1);
            assert_stmt.condition = unescape(args.substr(first_quote + 1, second_quote - first_quote - 1));

            auto cat_pos = args.find("category=", second_quote);
            if (cat_pos != ::std::string::npos) {
                auto cat_start = args.find('"', cat_pos);
                auto cat_end = args.rfind('"');
                if (cat_start != ::std::string::npos && cat_end != ::std::string::npos && cat_start != cat_end) {
                    assert_stmt.category = unescape(args.substr(cat_start + 1, cat_end - cat_start - 1));
                }
            }

            current_block->statements.push_back(assert_stmt);
        } else if (directive == "raw" && current_block) {
            RawStmt raw;
            if (args.size() >= 2 && args.front() == '"' && args.back() == '"') {
                raw.text = unescape(args.substr(1, args.size() - 2));
            }
            current_block->statements.push_back(raw);
        } else if (directive == "endblock") {
            current_block = nullptr;
        } else if (directive == "endfunction") {
            current_function = nullptr;
        } else if (directive == "type") {
            unit.types.emplace_back();
            current_type = &unit.types.back();
            current_type->name = args;
        } else if (directive == "body" && current_type) {
            if (args.size() >= 2 && args.front() == '"' && args.back() == '"') {
                current_type->body = unescape(args.substr(1, args.size() - 2));
            }
        } else if (directive == "endtype") {
            current_type = nullptr;
        } else if (directive == "raw_decl") {
            RawDecl raw;
            if (args.size() >= 2 && args.front() == '"' && args.back() == '"') {
                raw.text = unescape(args.substr(1, args.size() - 2));
            }
            unit.raw_declarations.push_back(raw);
        }
    }

    return unit;
}

} // namespace cppfort::stage0

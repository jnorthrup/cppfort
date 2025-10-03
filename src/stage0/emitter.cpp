#include "emitter.h"

#include <cctype>
#include <functional>
#include <iostream>
#include <regex>
#include <sstream>
#include <typeinfo>
#include <unordered_set>
#include <utility>

#include "ir/sea_of_nodes.h"
#include "ir_serializer.h"

namespace cppfort::stage0 {
namespace {
::std::string normalize_space(::std::string text) {
    ::std::string result;
    bool in_space = false;
    for (char c : text) {
        if (::std::isspace(static_cast<unsigned char>(c))) {
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

::std::string trim_copy(::std::string text) {
    while (!text.empty() && ::std::isspace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && ::std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    return text;
}

// Fix simple cpp2 type/expr idioms into valid C++ forms.
// Examples:
//  - type: "*::std::string" -> "::std::string*"
//  - expr: "s&" -> "&s"
//  - expr: "ps* = ..." -> "*ps = ..."
//  - simple UFCS pattern: ":cout << ... std" -> "::std::cout << ..."
::std::string fix_prefix_type(::std::string type) {
    // move leading '*' and '&' to the end
    ::std::string prefix;
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

bool looks_like_initializer_list(const ::std::string& expr) {
    if (expr.size() < 2 || expr.front() != '(' || expr.back() != ')') {
        return false;
    }

    ::std::size_t depth = 0;
    bool has_top_level_comma = false;
    for (::std::size_t i = 1; i + 1 < expr.size(); ++i) {
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

::std::string fix_expression_tokens(::std::string expr) {
    // Simple postfix deref: identifier* -> cpp2::impl::deref(identifier)
    // Use a manual scan to avoid regex dependency
    for (size_t pos = 0; pos + 1 < expr.size(); ++pos) {
        if (::std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_') {
            // find end of ident
            size_t start = pos;
            while (pos < expr.size() && (::std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_')) ++pos;
            if (pos < expr.size() && expr[pos] == '*') {
                ::std::string ident = expr.substr(start, pos - start);
                // replace ident* with cpp2::impl::deref(ident)
                ::std::string replacement = "cpp2::impl::deref(" + ident + ")";
                expr.replace(start, ident.size() + 1, replacement);
                // continue after the inserted sequence
                pos = start + replacement.size();
            }
        }
    }

    auto normalize_prefix_marker = [&](char marker) {
        size_t pos = 0;
        while ((pos = expr.find(marker, pos)) != ::std::string::npos) {
            // Find the previous non-space character to determine context.
            char prev = '\0';
            for (size_t i = pos; i > 0; --i) {
                char c = expr[i - 1];
                if (!::std::isspace(static_cast<unsigned char>(c))) {
                    prev = c;
                    break;
                }
            }

            if (!(prev == '<' || prev == '(' || prev == ',' || prev == '=' || prev == ':')) {
                ++pos;
                continue;
            }

            size_t type_start = pos + 1;
            while (type_start < expr.size() && ::std::isspace(static_cast<unsigned char>(expr[type_start]))) {
                ++type_start;
            }

            if (type_start >= expr.size()) {
                break;
            }

            size_t type_end = type_start;
            int angle_depth = 0;
            while (type_end < expr.size()) {
                char c = expr[type_end];
                if (c == '<') {
                    ++angle_depth;
                } else if (c == '>') {
                    if (angle_depth == 0) {
                        break;
                    }
                    --angle_depth;
                } else if ((c == ',' || c == ')') && angle_depth == 0) {
                    break;
                } else if (::std::isspace(static_cast<unsigned char>(c)) && angle_depth == 0) {
                    break;
                }
                ++type_end;
            }

            if (type_end == type_start) {
                ++pos;
                continue;
            }

            ::std::string raw_type = expr.substr(type_start, type_end - type_start);
            ::std::string trimmed = trim_copy(raw_type);
            if (trimmed.empty()) {
                ++pos;
                continue;
            }

            ::std::string replacement = fix_prefix_type(::std::string(1, marker) + trimmed);
            expr.replace(pos, type_end - pos, replacement);
            pos += replacement.size();
        }
    };

    normalize_prefix_marker('*');
    normalize_prefix_marker('&');

    auto replace_move_tokens = [&]() {
        size_t pos = 0;
        auto is_delim = [](char c) {
            return ::std::isspace(static_cast<unsigned char>(c)) || c == '(' || c == ')' || c == '{' ||
                   c == '}' || c == ';' || c == ',' || c == '[' || c == ']';
        };

        while ((pos = expr.find("move", pos)) != ::std::string::npos) {
            size_t token_start = pos;
            size_t token_end = pos + 4;

            char prev = token_start == 0 ? '\0' : expr[token_start - 1];
            char next = token_end < expr.size() ? expr[token_end] : '\0';
            bool prev_ok = token_start == 0 || is_delim(prev);
            bool next_ok = token_end >= expr.size() || is_delim(next);
            if (!prev_ok || !next_ok) {
                pos = token_end;
                continue;
            }

            size_t expr_start = token_end;
            while (expr_start < expr.size() && ::std::isspace(static_cast<unsigned char>(expr[expr_start]))) {
                ++expr_start;
            }
            if (expr_start >= expr.size()) {
                break;
            }

            ::std::string inner;
            size_t replace_end = expr_start;
            if (expr[expr_start] == '(') {
                int depth = 1;
                size_t i = expr_start + 1;
                while (i < expr.size() && depth > 0) {
                    if (expr[i] == '(') {
                        ++depth;
                    } else if (expr[i] == ')') {
                        --depth;
                    }
                    ++i;
                }
                if (depth != 0) {
                    pos = expr_start;
                    continue;
                }
                inner = expr.substr(expr_start + 1, i - expr_start - 2);
                replace_end = i;
            } else {
                size_t i = expr_start;
                while (i < expr.size() && ( ::std::isalnum(static_cast<unsigned char>(expr[i])) || expr[i] == '_' )) {
                    ++i;
                }
                if (i == expr_start) {
                    pos = expr_start;
                    continue;
                }
                inner = expr.substr(expr_start, i - expr_start);
                replace_end = i;
            }

            ::std::string trimmed = trim_copy(inner);
            if (trimmed.empty()) {
                pos = replace_end;
                continue;
            }

            ::std::string replacement = "std::move(" + trimmed + ")";
            expr.replace(token_start, replace_end - token_start, replacement);
            pos = token_start + replacement.size();
        }
    };

    replace_move_tokens();

    auto replace_assignment_parens = [&]() {
        size_t pos = 0;
        while ((pos = expr.find('=', pos)) != ::std::string::npos) {
            if (pos > 0 && expr[pos - 1] == '=') {
                ++pos;
                continue; // skip equality operators
            }

            size_t open = pos + 1;
            while (open < expr.size() && ::std::isspace(static_cast<unsigned char>(expr[open]))) {
                ++open;
            }
            if (open >= expr.size() || expr[open] != '(') {
                ++pos;
                continue;
            }

            size_t close = open + 1;
            int depth = 1;
            while (close < expr.size() && depth > 0) {
                char c = expr[close];
                if (c == '(') {
                    ++depth;
                } else if (c == ')') {
                    --depth;
                }
                ++close;
            }

            if (depth != 0) {
                break; // unbalanced
            }

            size_t content_start = open + 1;
            size_t inner_length = close > open + 1 && (close - open) > 2
                ? (close - open - 2)
                : 0;
            ::std::string inside = inner_length > 0
                ? expr.substr(content_start, inner_length)
                : ::std::string{};

            ::std::string trimmed = trim_copy(inside);

            expr.replace(open, close - open, ::std::string("{") + trimmed + "}");
            pos = open + trimmed.size() + 2; // account for braces
        }
    };

    replace_assignment_parens();

    // Convert CPP2 lambda syntax :(params) = body to C++ lambda syntax
    auto convert_lambdas = [&]() {
        size_t pos = 0;
        while ((pos = expr.find(':', pos)) != ::std::string::npos) {
            // Check if next char is '('
            size_t paren_pos = pos + 1;
            while (paren_pos < expr.size() && ::std::isspace(static_cast<unsigned char>(expr[paren_pos]))) {
                ++paren_pos;
            }
            if (paren_pos >= expr.size() || expr[paren_pos] != '(') {
                ++pos;
                continue;
            }

            // Find matching ')'
            size_t close_paren = paren_pos + 1;
            int depth = 1;
            while (close_paren < expr.size() && depth > 0) {
                if (expr[close_paren] == '(') ++depth;
                else if (expr[close_paren] == ')') --depth;
                ++close_paren;
            }
            if (depth != 0) {
                ++pos;
                continue;
            }

            // Look for '=' after params
            size_t eq_pos = close_paren;
            while (eq_pos < expr.size() && ::std::isspace(static_cast<unsigned char>(expr[eq_pos]))) {
                ++eq_pos;
            }
            if (eq_pos >= expr.size() || expr[eq_pos] != '=') {
                ++pos;
                continue;
            }

            // Extract params and body
            ::std::string params = expr.substr(paren_pos + 1, close_paren - paren_pos - 2);

            // Find body bounds - could be expression or {...} block
            size_t body_start = eq_pos + 1;
            while (body_start < expr.size() && ::std::isspace(static_cast<unsigned char>(expr[body_start]))) {
                ++body_start;
            }

            size_t body_end = body_start;
            if (body_start < expr.size() && expr[body_start] == '{') {
                // Block body
                int brace_depth = 0;
                while (body_end < expr.size()) {
                    if (expr[body_end] == '{') ++brace_depth;
                    else if (expr[body_end] == '}') {
                        --brace_depth;
                        if (brace_depth == 0) {
                            ++body_end;
                            break;
                        }
                    }
                    ++body_end;
                }
            } else {
                // Expression body - find end (comma, semicolon, or closing paren/bracket)
                int paren_depth = 0;
                int bracket_depth = 0;
                while (body_end < expr.size()) {
                    char c = expr[body_end];
                    if (c == '(') ++paren_depth;
                    else if (c == ')') {
                        if (paren_depth == 0) break;
                        --paren_depth;
                    }
                    else if (c == '[') ++bracket_depth;
                    else if (c == ']') {
                        if (bracket_depth == 0) break;
                        --bracket_depth;
                    }
                    else if ((c == ',' || c == ';') && paren_depth == 0 && bracket_depth == 0) {
                        break;
                    }
                    ++body_end;
                }
            }

            ::std::string body = trim_copy(expr.substr(body_start, body_end - body_start));

            // Process params - handle 'in', 'out', 'inout', 'move' qualifiers
            ::std::string processed_params;
            if (!params.empty()) {
                // Split params by comma
                ::std::vector<::std::string> param_list;
                size_t param_start = 0;
                int depth = 0;
                for (size_t i = 0; i < params.size(); ++i) {
                    if (params[i] == '(' || params[i] == '<') ++depth;
                    else if (params[i] == ')' || params[i] == '>') --depth;
                    else if (params[i] == ',' && depth == 0) {
                        param_list.push_back(params.substr(param_start, i - param_start));
                        param_start = i + 1;
                    }
                }
                param_list.push_back(params.substr(param_start));

                for (size_t i = 0; i < param_list.size(); ++i) {
                    ::std::string param = trim_copy(param_list[i]);
                    // Handle 'in', 'out', 'inout', 'move' qualifiers
                    // Each modifier affects how the parameter is captured/passed
                    if (param.starts_with("in ")) {
                        param = "const auto& " + param.substr(3);  // in = pass by const ref
                    } else if (param.starts_with("out ")) {
                        param = "auto& " + param.substr(4);        // out = pass by ref
                    } else if (param.starts_with("inout ")) {
                        param = "auto& " + param.substr(6);        // inout = pass by ref
                    } else if (param.starts_with("move ")) {
                        param = "auto&& " + param.substr(5);       // move = pass by rvalue ref
                    } else if (param.find(' ') == ::std::string::npos) {
                        // No type specified, just a name
                        param = "auto " + param;
                    }
                    // else: param already has a type specified, leave as-is

                    if (i > 0) processed_params += ", ";
                    processed_params += param;
                }
            }

            // Build C++ lambda
            ::std::string lambda = "[](";
            lambda += processed_params;
            lambda += ")";

            if (body[0] == '{') {
                // Block body
                lambda += " " + body;
            } else {
                // Expression body - check if it's a void expression (e.g., output statement)
                // Simple heuristic: if it contains << or starts with std::cout, it's likely void
                if (body.find("<<") != ::std::string::npos || body.find("std::cout") != ::std::string::npos ||
                    body.find("printf") != ::std::string::npos || body.find("+=") != ::std::string::npos ||
                    body.find("-=") != ::std::string::npos || body.find("*=") != ::std::string::npos ||
                    body.find("/=") != ::std::string::npos) {
                    // Void expression - no return
                    lambda += " { " + body + "; }";
                } else {
                    // Value expression - add return
                    lambda += " { return " + body + "; }";
                }
            }

            // Replace CPP2 lambda with C++ lambda
            expr.replace(pos, body_end - pos, lambda);
            pos += lambda.size();
        }
    };

    convert_lambdas();

    auto replace_type_literal_initializers = [&]() {
        size_t pos = 0;
        while ((pos = expr.find(':', pos)) != ::std::string::npos) {
            size_t type_start = pos + 1;
            while (type_start < expr.size() && ::std::isspace(static_cast<unsigned char>(expr[type_start]))) {
                ++type_start;
            }
            if (type_start >= expr.size()) {
                break;
            }

            size_t type_end = type_start;
            while (type_end < expr.size()) {
                char c = expr[type_end];
                if (::std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == ':' || c == '<' || c == '>') {
                    ++type_end;
                    continue;
                }
                break;
            }

            ::std::string type_name = trim_copy(expr.substr(type_start, type_end - type_start));
            if (type_name.empty()) {
                ++pos;
                continue;
            }

            size_t assign = type_end;
            while (assign < expr.size() && ::std::isspace(static_cast<unsigned char>(expr[assign]))) {
                ++assign;
            }
            if (assign >= expr.size() || expr[assign] != '=') {
                ++pos;
                continue;
            }

            size_t brace_start = assign + 1;
            while (brace_start < expr.size() && ::std::isspace(static_cast<unsigned char>(expr[brace_start]))) {
                ++brace_start;
            }
            if (brace_start >= expr.size() || expr[brace_start] != '{') {
                ++pos;
                continue;
            }

            size_t brace_end = brace_start;
            int depth = 0;
            while (brace_end < expr.size()) {
                char c = expr[brace_end];
                if (c == '{') {
                    ++depth;
                } else if (c == '}') {
                    --depth;
                    if (depth == 0) {
                        break;
                    }
                }
                ++brace_end;
            }

            if (brace_end >= expr.size()) {
                break;
            }

            ::std::string braces = expr.substr(brace_start, brace_end - brace_start + 1);
            ::std::string replacement = type_name + braces;
            expr.replace(pos, brace_end - pos + 1, replacement);
            pos += replacement.size();
        }
    };

    replace_type_literal_initializers();

    auto replace_as_expressions = [&]() {
        size_t search_pos = 0;
        while ((search_pos = expr.find(" as ", search_pos)) != ::std::string::npos) {
            size_t as_pos = search_pos + 1; // position of 'a'

            // Determine left expression bounds
            size_t left_end = search_pos; // space before 'as'
            while (left_end > 0 && ::std::isspace(static_cast<unsigned char>(expr[left_end - 1]))) {
                --left_end;
            }
            if (left_end == 0) {
                search_pos += 4;
                continue;
            }

            size_t left_start = left_end;
            if (expr[left_end - 1] == ')' || expr[left_end - 1] == ']') {
                char closing = expr[left_end - 1];
                char opening = closing == ')' ? '(' : '[';
                int depth = 1;
                size_t i = left_end - 2;
                while (i < expr.size()) {
                    if (expr[i] == closing) {
                        ++depth;
                    } else if (expr[i] == opening) {
                        --depth;
                        if (depth == 0) {
                            left_start = i;
                            break;
                        }
                    }
                    if (i == 0) {
                        break;
                    }
                    --i;
                }
                if (depth != 0) {
                    search_pos += 4;
                    continue;
                }
            } else {
                while (left_start > 0) {
                    char c = expr[left_start - 1];
                    if (::std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == ':' || c == '.') {
                        --left_start;
                    } else {
                        break;
                    }
                }
            }

            ::std::string left_expr = trim_copy(expr.substr(left_start, left_end - left_start));
            if (left_expr.empty()) {
                search_pos += 4;
                continue;
            }

            size_t type_start = search_pos + 4; // skip " as "
            while (type_start < expr.size() && ::std::isspace(static_cast<unsigned char>(expr[type_start]))) {
                ++type_start;
            }
            if (type_start >= expr.size()) {
                break;
            }

            size_t type_end = type_start;
            int angle_depth = 0;
            while (type_end < expr.size()) {
                char c = expr[type_end];
                if (c == '<') {
                    ++angle_depth;
                } else if (c == '>') {
                    if (angle_depth == 0) {
                        ++type_end;
                        break;
                    }
                    --angle_depth;
                } else if (::std::isspace(static_cast<unsigned char>(c)) || c == ')' || c == ',' || c == ';') {
                    if (angle_depth == 0) {
                        break;
                    }
                }
                ++type_end;
            }

            ::std::string type_expr = trim_copy(expr.substr(type_start, type_end - type_start));
            if (type_expr.empty()) {
                search_pos = type_end;
                continue;
            }

            size_t replace_end = type_end;
            ::std::string replacement = "cpp2::impl::as_<" + type_expr + ">(" + left_expr + ")";
            expr.replace(left_start, replace_end - left_start, replacement);
            search_pos = left_start + replacement.size();
        }
    };

    replace_as_expressions();

    // Simple address-of suffix: something& -> &something (only for simple tokens)
    for (size_t pos = 0; pos + 1 < expr.size(); ++pos) {
        if ((::std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_') && expr[pos+1] == '&') {
            // find start of ident
            size_t end = pos;
            size_t start = pos;
            while (start > 0 && (::std::isalnum(static_cast<unsigned char>(expr[start-1])) || expr[start-1] == '_')) --start;
            ::std::string ident = expr.substr(start, end - start + 1);
            // replace ident& with &ident
            expr.replace(start, ident.size() + 1, ::std::string("&") + ident);
            pos = start + 1 + ident.size();
        }
    }

    // UFCS for selected C stdio-like functions: obj.fprintf(args) -> fprintf(obj, args)
    static const ::std::unordered_set<::std::string> k_c_ufcs_names {
        "fprintf", "fclose", "fflush", "fread", "fwrite", "fgets", "fputs",
        "fscanf", "fgetc", "fputc", "fopen", "freopen"
    };

    for (size_t pos = 0; pos + 2 < expr.size(); ++pos) {
        if (!::std::isalnum(static_cast<unsigned char>(expr[pos])) && expr[pos] != '_') {
            continue;
        }

        size_t obj_start = pos;
        while (pos < expr.size() && (::std::isalnum(static_cast<unsigned char>(expr[pos])) || expr[pos] == '_')) {
            ++pos;
        }

        if (pos + 1 >= expr.size() || expr[pos] != '.') {
            continue;
        }

        if (!::std::isalnum(static_cast<unsigned char>(expr[pos + 1])) && expr[pos + 1] != '_') {
            continue;
        }

        size_t method_start = pos + 1;
        size_t method_end = method_start;
        while (method_end < expr.size() && (::std::isalnum(static_cast<unsigned char>(expr[method_end])) || expr[method_end] == '_')) {
            ++method_end;
        }

        if (method_end >= expr.size() || expr[method_end] != '(') {
            continue;
        }

        ::std::string method = expr.substr(method_start, method_end - method_start);
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

        ::std::string obj = expr.substr(obj_start, pos - obj_start);
        ::std::string args = expr.substr(method_end + 1, args_end - method_end - 2);
        ::std::string replacement = method + "(" + obj;
        if (!args.empty()) {
            replacement += ", " + args;
        }
        replacement += ")";

        expr.replace(obj_start, args_end - obj_start, replacement);
        pos = obj_start + replacement.size();
    }

    // Win #2: Transform cpp2 smart pointer syntax to standard C++
    // Pattern: unique.new<T>(...) → ::std::make_unique<T>(...)
    size_t pos = 0;
    while ((pos = expr.find("unique.new<", pos)) != ::std::string::npos) {
        expr.replace(pos, 11, "::std::make_unique<");
        pos += 17;  // length of "::std::make_unique<"
    }

    // Pattern: shared.new<T>(...) → ::std::make_shared<T>(...)
    pos = 0;
    while ((pos = expr.find("shared.new<", pos)) != ::std::string::npos) {
        expr.replace(pos, 11, "::std::make_shared<");
        pos += 17;  // length of "::std::make_shared<"
    }

    // Pattern: new<T>(...) → ::std::make_unique<T>(...) (shorthand for unique.new)
    // Need to be careful not to match operator new or member.new
    ::std::regex new_shorthand(R"(\bnew<)");
    expr = ::std::regex_replace(expr, new_shorthand, "::std::make_unique<");

    // Win #3: Transform cpp2 suffix dereference operator to prefix
    // Pattern: identifier* → *identifier
    // Match identifier followed by * at end or before operators/punctuation
    // But not multiplication (identifier * identifier)
    // TEMPORARILY DISABLED: Only match simple identifiers (not qualified names starting with ::)
    // ::std::regex suffix_deref(R"(([a-zA-Z_]\w+)\*(?=\s*(?:[+\-;,\)\]\}]|$)))");
    // expr = ::std::regex_replace(expr, suffix_deref, "*$1");

    // Handle discard pattern: _ = expr -> (void)expr
    if (expr.size() >= 4 && expr.substr(0, 4) == "_ = ") {
        expr = "(void)(" + expr.substr(4) + ")";
    }

    // Remove unnecessary parentheses around simple expressions
    // Pattern: (simple_expr) -> simple_expr, where simple_expr is a number, identifier, or string literal
    if (expr.size() >= 3 && expr.front() == '(' && expr.back() == ')') {
        ::std::string inner = expr.substr(1, expr.size() - 2);
        // Check if inner is a simple expression (number, identifier, or string literal)
        bool is_simple = true;
        if (inner.empty()) {
            is_simple = false;
        } else if (::std::isdigit(static_cast<unsigned char>(inner[0])) || inner[0] == '"' || inner[0] == '\'') {
            // Starts with digit, quote, or apostrophe - likely a literal
        } else if (::std::isalpha(static_cast<unsigned char>(inner[0])) || inner[0] == '_') {
            // Starts with letter or underscore - check if it's a valid identifier
            for (char c : inner) {
                if (!(::std::isalnum(static_cast<unsigned char>(c)) || c == '_')) {
                    is_simple = false;
                    break;
                }
            }
        } else {
            is_simple = false;
        }
        
        if (is_simple) {
            expr = inner;
        }
    }

    return expr;
}
}

::std::string Emitter::emit(const TranslationUnit& unit, const EmitOptions& options) const {
    ::std::string result;
    switch (options.backend) {
        case EmitBackend::Cpp:
            result = emit_cpp(unit, options);
            break;
        case EmitBackend::Mlir:
            result = emit_mlir(unit);
            break;
        case EmitBackend::IR:
            result = emit_ir(unit);
            break;
        default:
            return {};
    }

    // GLOBAL HACK: Force lambda conversion on the entire output
    // Process the result line by line to convert any remaining CPP2 lambdas
    ::std::istringstream stream(result);
    ::std::string line;
    ::std::string processed_result;
    while (::std::getline(stream, line)) {
        if (line.find(":(") != ::std::string::npos) {
            line = fix_expression_tokens(line);
        }
        // ALSO fix any bad lambda param patterns that slipped through
        size_t pos = 0;
        while ((pos = line.find("](&auto ", pos)) != ::std::string::npos) {
            line.replace(pos + 2, 6, "auto& ");
            pos += 8;
        }
        processed_result += line + "\n";
    }
    return processed_result;
}

::std::string Emitter::emit_cpp(const TranslationUnit& unit, const EmitOptions& options) const {
    ::std::string output;
    
    // Detect required headers by scanning the code
    bool needs_cstdio = false;
    bool needs_cassert = false;
    
    // Scan all expressions for C stdio functions
    auto scan_for_stdio = [&](const ::std::string& expr) {
        if (expr.find("fopen") != ::std::string::npos ||
            expr.find("fprintf") != ::std::string::npos ||
            expr.find("fclose") != ::std::string::npos ||
            expr.find("fread") != ::std::string::npos ||
            expr.find("fwrite") != ::std::string::npos) {
            needs_cstdio = true;
        }
    };
    
    ::std::function<void(const Statement&)> scan_statement;
    scan_statement = [&](const Statement& stmt) {
        ::std::visit([&](const auto& node) {
            using T = ::std::decay_t<decltype(node)>;
            if constexpr (::std::is_same_v<T, VariableDecl>) {
                if (node.initializer) {
                    scan_for_stdio(*node.initializer);
                }
            } else if constexpr (::std::is_same_v<T, ExpressionStmt>) {
                scan_for_stdio(node.expression);
            } else if constexpr (::std::is_same_v<T, ReturnStmt>) {
                if (node.expression) {
                    scan_for_stdio(*node.expression);
                }
            } else if constexpr (::std::is_same_v<T, ForChainStmt>) {
                scan_for_stdio(node.range_expression);
                if (node.next_expression) {
                    scan_for_stdio(*node.next_expression);
                }
                for (const auto& inner : node.body.statements) {
                    scan_statement(inner);
                }
            } else if constexpr (::std::is_same_v<T, AssertStmt>) {
                needs_cassert = true;
                scan_for_stdio(node.condition);
                if (node.category) {
                    scan_for_stdio(*node.category);
                }
            } else if constexpr (::std::is_same_v<T, RawStmt>) {
                scan_for_stdio(node.text);
            }
        }, stmt);
    };

    // Scan function bodies
    for (const auto& fn : unit.functions) {
        if (::std::holds_alternative<Block>(fn.body)) {
            const auto& block = ::std::get<Block>(fn.body);
            for (const auto& stmt : block.statements) {
                scan_statement(stmt);
            }
        } else if (::std::holds_alternative<ExpressionBody>(fn.body)) {
            const auto& expr_body = ::std::get<ExpressionBody>(fn.body);
            scan_for_stdio(expr_body.expression);
        }
    }
    
    if (options.include_headers) {
        output += "// Generated by cppfort stage0 transpiler\n";
        output += "#include <cstdint>\n";
        output += "#include <iostream>\n";
        output += "#include <memory>\n";
        output += "#include <string>\n";
        output += "#include <string_view>\n";
        output += "#include <vector>\n";
        output += "#include <utility>\n";
        output += "#include \"cpp2.h\"\n";
        if (needs_cassert) {
            output += "#include <cassert>\n";
        }
        if (needs_cstdio) {
            output += "#include <cstdio>\n";
        }
        output += "\n";
        // Cpp2 type aliases
        output += "using i8 = ::std::int8_t;\n";
        output += "using i16 = ::std::int16_t;\n";
        output += "using i32 = ::std::int32_t;\n";
        output += "using i64 = ::std::int64_t;\n";
        output += "using u8 = ::std::uint8_t;\n";
        output += "using u16 = ::std::uint16_t;\n";
        output += "using u32 = ::std::uint32_t;\n";
        output += "using u64 = ::std::uint64_t;\n\n";
    }

    bool first = true;
    for (const auto& type : unit.types) {
        if (!first) {
            output += '\n';
        }
        emit_type(type, output, 0);
        first = false;
    }

    // Win #1: Emit forward declarations for all functions to fix forward reference errors
    // This allows functions to call other functions defined later in the source
    if (!unit.functions.empty()) {
        if (!first) {
            output += '\n';
        }
        output += "// Forward declarations\n";
        for (const auto& fn : unit.functions) {
            emit_forward_declaration(fn, output, 0);
        }
        output += '\n';
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
}

::std::string Emitter::get_param_type(const Parameter& param) const {
    auto type = normalize_space(param.type);
    if (type.empty() || type == "_") {
        type = "auto";
    }
    // canonicalize prefix pointer styles like '*T'
    type = fix_prefix_type(type);
    switch (param.kind) {
        case ParameterKind::In:
            return ::std::string("cpp2::impl::in<") + type + ">";
        case ParameterKind::InOut:
        case ParameterKind::Out:
            return type + "&";
        case ParameterKind::Copy:
            return ::std::string("cpp2::impl::copy<") + type + ">";
        case ParameterKind::Move:
            return ::std::string("cpp2::impl::move<") + type + ">";
        case ParameterKind::Forward:
            return type + "&&";
        case ParameterKind::Default:
        default:
            return type;
    }
}

void Emitter::emit_forward_declaration(const FunctionDecl& fn, ::std::string& out, int indent) const {
    auto return_type = fn.return_type.has_value() && !fn.return_type->empty()
        ? normalize_space(*fn.return_type)
        : ::std::string{"void"};
    if (return_type == "_") {
        return_type = "auto";
    }
    return_type = fix_prefix_type(return_type);

    // Special handling for main: must return int
    bool is_main = fn.name == "main";
    if (is_main && return_type == "void") {
        return_type = "int";
    }

    ::std::ostringstream fwd;
    fwd << "auto " << fn.name << "(";
    for (::std::size_t i = 0; i < fn.parameters.size(); ++i) {
        if (i != 0) {
            fwd << ", ";
        }
        const auto& param = fn.parameters[i];
        auto ptype = get_param_type(param);
        fwd << ptype << ' ' << param.name;
    }
    fwd << ") -> " << return_type << ";";

    Emitter::append_line(out, fwd.str(), indent);
}

void Emitter::emit_function(const FunctionDecl& fn, ::std::string& out, int indent) const {
    auto return_type = fn.return_type.has_value() && !fn.return_type->empty()
        ? normalize_space(*fn.return_type)
        : ::std::string {"void"};
    if (return_type == "_") {
        return_type = "auto";
    }
    return_type = fix_prefix_type(return_type);

    // Special handling for main: must return int
    bool is_main = fn.name == "main";
    if (is_main && return_type == "void") {
        return_type = "int";
    }

    ::std::ostringstream signature;
    signature << "auto " << fn.name << "(";
    for (::std::size_t i = 0; i < fn.parameters.size(); ++i) {
        if (i != 0) {
            signature << ", ";
        }
        const auto& param = fn.parameters[i];
        auto ptype = get_param_type(param);
        signature << ptype << ' ' << param.name;
    }
    signature << ") -> " << return_type << ' ';

    // Emit a debug comment with parameter kinds (helps validate parser wiring)
    signature << "/*kinds:";
    for (::std::size_t i = 0; i < fn.parameters.size(); ++i) {
        if (i) signature << ",";
        switch (fn.parameters[i].kind) {
            case ParameterKind::In: signature << "In"; break;
            case ParameterKind::InOut: signature << "InOut"; break;
            case ParameterKind::Out: signature << "Out"; break;
            case ParameterKind::Copy: signature << "Copy"; break;
            case ParameterKind::Move: signature << "Move"; break;
            case ParameterKind::Forward: signature << "Forward"; break;
            default: signature << "Default"; break;
        }
    }
    signature << "*/";

    Emitter::append_line(out, signature.str() + "{", indent);

        if (::std::holds_alternative<Block>(fn.body)) {
        emit_block(::std::get<Block>(fn.body), out, indent + 1, is_main && return_type == "int");
    } else {
        const auto& expr = ::std::get<ExpressionBody>(fn.body);
        ::std::string line;
        auto fixed = fix_expression_tokens(normalize_space(expr.expression));
        if (return_type == "void") {
            line = fixed + ';';
        } else {
            line = "return " + fixed + ';';
        }
        Emitter::append_line(out, line, indent + 1);
    }

    Emitter::append_line(out, "}", indent);
}

void Emitter::emit_type(const TypeDecl& type, ::std::string& out, int indent) const {
    ::std::ostringstream signature;
    signature << "class " << type.name << " {";
    Emitter::append_line(out, signature.str(), indent);

    // For now, emit the body as-is, assuming it's C++ code
    // TODO: Parse and emit proper Cpp2 type members
    ::std::string body = normalize_space(type.body);
    if (!body.empty()) {
        Emitter::append_line(out, body, indent + 1);
    }

    Emitter::append_line(out, "};", indent);
}

void Emitter::emit_block(const Block& block, ::std::string& out, int indent, bool add_return_0) const {
    for (const auto& stmt : block.statements) {
        emit_statement(stmt, out, indent);
    }
    if (add_return_0) {
        Emitter::append_line(out, "return 0;", indent);
    }
}

void Emitter::emit_statement(const Statement& stmt, ::std::string& out, int indent) const {
    ::std::visit([
        &out,
        indent,
        this
    ](const auto& node) {
        using T = ::std::decay_t<decltype(node)>;
        if constexpr (::std::is_same_v<T, VariableDecl>) {
            auto type = normalize_space(node.type);
            if (type.empty()) {
                type = "auto";
            }
            type = fix_prefix_type(type);

            if (type == "type") {
                ::std::string alias = node.initializer.has_value()
                    ? normalize_space(*node.initializer)
                    : ::std::string{};
                while (!alias.empty() && alias.front() == '=') {
                    alias.erase(alias.begin());
                }
                alias = trim_copy(alias);
                alias = fix_prefix_type(alias);

                if (alias.empty()) {
                    alias = "auto"; // Fallback to keep generated code valid
                }

                ::std::string line = "using " + node.name + " = " + alias + ';';
                Emitter::append_line(out, line, indent);
                return;
            }

            ::std::string line = type + ' ' + node.name;
            if (node.initializer.has_value() && !node.initializer->empty()) {
                auto raw_init = normalize_space(*node.initializer);
                bool paren_wrapped = raw_init.size() >= 2 && raw_init.front() == '(' && raw_init.back() == ')';
                if (paren_wrapped) {
                    raw_init.front() = '{';
                    raw_init.back() = '}';
                }

                auto init = fix_expression_tokens(raw_init);

                if (!paren_wrapped && looks_like_initializer_list(init)) {
                    init.front() = '{';
                    init.back() = '}';
                }

                line += " = " + init;
            }
            line += ';';
            Emitter::append_line(out, line, indent);
        } else if constexpr (::std::is_same_v<T, ExpressionStmt>) {
            Emitter::append_line(out, fix_expression_tokens(normalize_space(node.expression)) + ';', indent);
        } else if constexpr (::std::is_same_v<T, ReturnStmt>) {
            if (node.expression.has_value() && !node.expression->empty()) {
                Emitter::append_line(out, "return " + fix_expression_tokens(normalize_space(*node.expression)) + ';', indent);
            } else {
                Emitter::append_line(out, "return;", indent);
            }
        } else if constexpr (::std::is_same_v<T, AssertStmt>) {
            auto condition = fix_expression_tokens(normalize_space(node.condition));
            ::std::string line = "assert((" + condition + "))";
            line.push_back(';');
            if (node.category && !node.category->empty()) {
                line += " // " + *node.category;
            }
            Emitter::append_line(out, line, indent);
        } else if constexpr (::std::is_same_v<T, ForChainStmt>) {
            auto param_type = node.loop_parameter.type;
            const auto kind = node.loop_parameter.kind;

            auto resolve_type = [&]() {
                if (!param_type.empty()) {
                    return normalize_space(param_type);
                }
                switch (kind) {
                    case ParameterKind::In:
                        return ::std::string("const auto&");
                    case ParameterKind::InOut:
                    case ParameterKind::Out:
                        return ::std::string("auto&");
                    case ParameterKind::Move:
                    case ParameterKind::Forward:
                        return ::std::string("auto&&");
                    default:
                        return ::std::string("auto");
                }
            };

            const auto loop_type = resolve_type();
            const auto loop_var = node.loop_parameter.name.empty() ? ::std::string("item") : node.loop_parameter.name;
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
        } else if constexpr (::std::is_same_v<T, RawStmt>) {
            // Process raw statements through fix_expression_tokens if they might contain lambdas
            ::std::string processed = node.text;
            if (processed.find(":(") != ::std::string::npos) {
                // This is likely an expression with lambdas - process it
                processed = fix_expression_tokens(processed);
            }
            Emitter::append_line(out, processed, indent);
        }
    }, stmt);
}

void Emitter::append_line(::std::string& out, ::std::string_view text, int indent) {
    out.append(indent_string(indent));
    out.append(text);
    out.push_back('\n');
}

::std::string Emitter::indent_string(int indent) {
    constexpr int spaces_per_indent = 4;
    return ::std::string(static_cast<::std::size_t>(indent * spaces_per_indent), ' ');
}

::std::string Emitter::emit_mlir(const TranslationUnit& unit) const {
    // Chapter 19: MLIR emission via Sea of Nodes
    // Full implementation requires GraphBuilder and InstructionSelection from IR layer
    // For now, emit simple MLIR module structure

    ::std::ostringstream os;
    os << "module {\n";

    for (const auto& fn : unit.functions) {
        const ::std::string functionName = fn.name.empty() ? "anon" : fn.name;
        os << "  func.func @" << functionName << "() -> i32 {\n";
        os << "    // TODO: Sea of Nodes IR emission\n";
        os << "    %0 = arith.constant 0 : i32\n";
        os << "    func.return %0 : i32\n";
        os << "  }\n";
    }

    os << "}\n";
    return os.str();
}

::std::string Emitter::emit_ir(const TranslationUnit& unit) const {
    // Emit the AST in a serialized IR format that can be read by stage1
    // This allows stage0 -> stage1 communication via IR instead of C++
    return IRSerializer::serialize(unit);
}

} // namespace cppfort::stage0

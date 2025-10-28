#include "cpp2_emitter.h"
#include "tblgen_pattern_matcher.h"
#include "confix_tracker.h"
#include "depth_pattern_matcher.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <sstream>
#include <string_view>
#include <vector>

#include "confix_orbit.h"
#include "pattern_loader.h"

namespace cppfort::stage0 {

namespace {

// Forward declarations
std::string apply_recursive_transformations(const std::string& input, const std::vector<PatternData>& patterns, int nesting_depth = 0);
std::string transform_parameter(const std::string& param, const std::vector<PatternData>& patterns);

std::string trim_copy(std::string_view text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return std::string{text.substr(begin, end - begin)};
}

void trim_in_place(std::string& text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    text.assign(text, begin, end - begin);
}

size_t find_top_level_colon(std::string_view text) {
    int paren = 0;
    int brace = 0;
    int bracket = 0;
    int angle = 0;

    for (size_t i = 0; i < text.size(); ++i) {
        char ch = text[i];
        switch (ch) {
            case '(': ++paren; break;
            case ')': if (paren > 0) --paren; break;
            case '{': ++brace; break;
            case '}': if (brace > 0) --brace; break;
            case '[': ++bracket; break;
            case ']': if (bracket > 0) --bracket; break;
            case '<': ++angle; break;
            case '>': if (angle > 0) --angle; break;
            default: break;
        }

        if (paren || brace || bracket || angle) {
            continue;
        }

        if (ch == ':' && !(i > 0 && text[i - 1] == ':') && !(i + 1 < text.size() && text[i + 1] == ':')) {
            return i;
        }
    }

    return std::string::npos;
}

size_t find_top_level_equal(std::string_view text) {
    int paren = 0;
    int brace = 0;
    int bracket = 0;
    int angle = 0;

    for (size_t i = 0; i < text.size(); ++i) {
        char ch = text[i];
        switch (ch) {
            case '(': ++paren; break;
            case ')': if (paren > 0) --paren; break;
            case '{': ++brace; break;
            case '}': if (brace > 0) --brace; break;
            case '[': ++bracket; break;
            case ']': if (bracket > 0) --bracket; break;
            case '<': ++angle; break;
            case '>': if (angle > 0) --angle; break;
            default: break;
        }

        if (paren || brace || bracket || angle) {
            continue;
        }

        if (ch == '=') {
            char prev = (i > 0) ? text[i - 1] : '\0';
            char next = (i + 1 < text.size()) ? text[i + 1] : '\0';
            if (prev == '=' || prev == '!' || prev == '<' || prev == '>') {
                continue;
            }
            if (next == '=') {
                continue;
            }
            return i;
        }
    }

    return std::string::npos;
}

bool contains_const_qualifier(std::string_view text) {
    size_t pos = text.find("const");
    while (pos != std::string::npos) {
        bool valid_prefix = (pos == 0) || (!std::isalnum(static_cast<unsigned char>(text[pos - 1])) && text[pos - 1] != '_');
        bool valid_suffix = (pos + 5 >= text.size()) || (!std::isalnum(static_cast<unsigned char>(text[pos + 5])) && text[pos + 5] != '_');
        if (valid_prefix && valid_suffix) {
            return true;
        }
        pos = text.find("const", pos + 5);
    }
    return false;
}

void remove_const_qualifier(std::string& text) {
    size_t pos = text.find("const");
    while (pos != std::string::npos) {
        bool valid_prefix = (pos == 0) || (!std::isalnum(static_cast<unsigned char>(text[pos - 1])) && text[pos - 1] != '_');
        bool valid_suffix = (pos + 5 >= text.size()) || (!std::isalnum(static_cast<unsigned char>(text[pos + 5])) && text[pos + 5] != '_');
        if (valid_prefix && valid_suffix) {
            text.erase(pos, 5);
            // Remove any doubled whitespace created by erasing "const"
            while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
                text.erase(pos, 1);
            }
            if (pos > 0 && pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos - 1])) && std::isspace(static_cast<unsigned char>(text[pos]))) {
                text.erase(pos, 1);
            }
            // Restart search to catch multiple const qualifiers
            pos = text.find("const");
            continue;
        }
        pos = text.find("const", pos + 5);
    }
    trim_in_place(text);
}

bool ends_with_reference(std::string_view text) {
    size_t end = text.size();
    while (end > 0 && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    if (end == 0) {
        return false;
    }
    if (text[end - 1] == '&') {
        if (end >= 2 && text[end - 2] == '&') {
            return true;
        }
        return true;
    }
    return false;
}

bool ends_with_rvalue_reference(std::string_view text) {
    size_t end = text.size();
    while (end > 0 && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    if (end < 2) {
        return false;
    }
    return text[end - 1] == '&' && text[end - 2] == '&';
}

void ensure_reference(std::string& text) {
    trim_in_place(text);
    if (!ends_with_reference(text)) {
        text += "&";
    }
}

void ensure_rvalue_reference(std::string& text) {
    trim_in_place(text);
    if (ends_with_rvalue_reference(text)) {
        return;
    }

    // Drop trailing single '&' if present before adding &&
    if (ends_with_reference(text) && !ends_with_rvalue_reference(text)) {
        // Remove trailing '&'
        size_t end = text.size();
        while (end > 0 && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
            --end;
        }
        if (end > 0) {
            --end;
            if (end > 0 && text[end - 1] == '&') {
                --end;
            }
            text.erase(end);
        }
        trim_in_place(text);
    }

    text += "&&";
}

void remove_reference_suffix(std::string& text) {
    trim_in_place(text);
    size_t end = text.size();
    while (end > 0 && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    if (end == 0) {
        text.clear();
        return;
    }
    if (text[end - 1] == '&') {
        size_t erase_start = end - 1;
        if (erase_start > 0 && text[erase_start - 1] == '&') {
            --erase_start;
        }
        text.erase(erase_start);
        trim_in_place(text);
    }
}

std::vector<std::string> split_whitespace_tokens(std::string_view text) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
        size_t start = i;
        while (i < text.size() && !std::isspace(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
        if (start < i) {
            tokens.emplace_back(text.substr(start, i - start));
        }
    }
    return tokens;
}

bool is_parameter_kind_token(std::string_view token) {
    return token == "in"
        || token == "in_ref"
        || token == "copy"
        || token == "inout"
        || token == "out"
        || token == "move"
        || token == "forward"
        || token == "forward_ref";
}

std::string normalize_type_token(std::string_view type) {
    std::string normalized = trim_copy(type);
    if (normalized.empty()) {
        return normalized;
    }
    if (normalized == "_") {
        return "auto";
    }
    return normalized;
}

bool has_pack_suffix(std::string_view text) {
    size_t end = text.size();
    while (end > 0 && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return end >= 3 && text.substr(end - 3, 3) == "...";
}

void remove_pack_suffix(std::string& text) {
    size_t end = text.size();
    while (end > 0 && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    if (end >= 3 && text.compare(end - 3, 3, "...") == 0) {
        text.erase(end - 3);
        trim_in_place(text);
    }
}

void append_pack_suffix(std::string& text) {
    trim_in_place(text);
    if (!has_pack_suffix(text)) {
        text += "...";
    }
}

std::string join_tokens(const std::vector<std::string>& tokens, size_t start_index = 0) {
    std::string result;
    for (size_t i = start_index; i < tokens.size(); ++i) {
        if (i > start_index) {
            result += " ";
        }
        result += tokens[i];
    }
    return result;
}

enum class ParameterKind {
    In,
    InRef,
    Copy,
    Inout,
    Out,
    Move,
    Forward,
    ForwardRef,
    Unknown
};

ParameterKind to_parameter_kind(std::string_view kind) {
    if (kind == "in") return ParameterKind::In;
    if (kind == "in_ref") return ParameterKind::InRef;
    if (kind == "copy") return ParameterKind::Copy;
    if (kind == "inout") return ParameterKind::Inout;
    if (kind == "out") return ParameterKind::Out;
    if (kind == "move") return ParameterKind::Move;
    if (kind == "forward") return ParameterKind::Forward;
    if (kind == "forward_ref") return ParameterKind::ForwardRef;
    if (kind.empty()) return ParameterKind::In;
    return ParameterKind::Unknown;
}

bool transform_cpp2_parameter_direct(const std::string& original, std::string& output) {
    std::string trimmed = trim_copy(original);
    if (trimmed.empty()) {
        output = trimmed;
        return true;
    }

    size_t colon_pos = find_top_level_colon(trimmed);
    if (colon_pos == std::string::npos) {
        return false; // Not a colon-form parameter (e.g., "forward x" or "this")
    }

    std::string before = trim_copy(trimmed.substr(0, colon_pos));
    std::string after = trim_copy(trimmed.substr(colon_pos + 1));
    if (before.empty() || after.empty()) {
        return false;
    }

    std::string default_value;
    size_t equals_pos = find_top_level_equal(after);
    if (equals_pos != std::string::npos) {
        default_value = trim_copy(after.substr(equals_pos + 1));
        after = trim_copy(after.substr(0, equals_pos));
    }

    auto tokens = split_whitespace_tokens(before);
    if (tokens.empty()) {
        return false;
    }

    std::string kind_token;
    std::string name_token;

    for (const auto& token : tokens) {
        if (is_parameter_kind_token(token)) {
            kind_token = std::string(token);
            continue;
        }
        if (token == "virtual" || token == "constexpr") {
            // Function qualifiers belong outside the parameter list; ignore them here.
            continue;
        }
        if (name_token.empty()) {
            name_token = std::string(token);
        } else {
            name_token += " " + std::string(token);
        }
    }

    if (name_token.empty()) {
        return false;
    }

    if (name_token == "this") {
        // Method-style parameters require additional context; fall back to legacy handling.
        return false;
    }

    bool name_variadic = false;
    if (name_token.size() > 3 && name_token.substr(name_token.size() - 3) == "...") {
        name_variadic = true;
        name_token.erase(name_token.size() - 3);
        trim_in_place(name_token);
        if (name_token.empty()) {
            return false;
        }
    }

    std::string type_token = normalize_type_token(after);
    bool type_variadic = false;
    if (!type_token.empty() && has_pack_suffix(type_token)) {
        type_variadic = true;
        remove_pack_suffix(type_token);
    }

    ParameterKind kind = to_parameter_kind(kind_token);

    if (type_token.empty()) {
        type_token = "auto";
    }

    std::string type_builder = type_token;
    switch (kind) {
        case ParameterKind::In:
        case ParameterKind::InRef: {
            if (!contains_const_qualifier(type_builder)) {
                type_builder = "const " + type_builder;
            }
            ensure_reference(type_builder);
            break;
        }
        case ParameterKind::Copy: {
            remove_const_qualifier(type_builder);
            remove_reference_suffix(type_builder);
            break;
        }
        case ParameterKind::Inout: {
            remove_const_qualifier(type_builder);
            ensure_reference(type_builder);
            break;
        }
        case ParameterKind::Out: {
            remove_const_qualifier(type_builder);
            remove_reference_suffix(type_builder);
            trim_in_place(type_builder);
            if (type_builder.empty()) {
                type_builder = "auto";
            }
            if (type_builder.rfind("cpp2::impl::out", 0) != 0) {
                type_builder = "cpp2::impl::out<" + type_builder + ">";
            }
            break;
        }
        case ParameterKind::Move: {
            remove_const_qualifier(type_builder);
            remove_reference_suffix(type_builder);
            ensure_rvalue_reference(type_builder);
            break;
        }
        case ParameterKind::Forward:
        case ParameterKind::ForwardRef: {
            remove_const_qualifier(type_builder);
            remove_reference_suffix(type_builder);
            ensure_rvalue_reference(type_builder);
            break;
        }
        case ParameterKind::Unknown:
            return false;
    }

    bool is_variadic = name_variadic || type_variadic;
    if (is_variadic) {
        append_pack_suffix(type_builder);
    }

    trim_in_place(type_builder);
    std::string result = type_builder;
    if (!result.empty() && !name_token.empty()) {
        result += " ";
    }
    result += name_token;
    if (!default_value.empty()) {
        result += " = " + default_value;
    }

    output = result;
    return true;
}

std::string transform_return_spec(std::string_view spec) {
    std::string trimmed = trim_copy(spec);
    if (trimmed.empty()) {
        return "auto";
    }

    auto tokens = split_whitespace_tokens(trimmed);
    if (tokens.empty()) {
        return "auto";
    }

    std::string kind_token;
    if (!tokens.empty() && (tokens[0] == "forward" || tokens[0] == "forward_ref" || tokens[0] == "move")) {
        kind_token = tokens[0];
        tokens.erase(tokens.begin());
    }

    std::string type_part = join_tokens(tokens);
    if (type_part.empty() || type_part == "_") {
        type_part = "auto";
    }

    std::string type_builder = type_part;
    if (kind_token == "forward" || kind_token == "forward_ref") {
        remove_const_qualifier(type_builder);
        remove_reference_suffix(type_builder);
        ensure_rvalue_reference(type_builder);
    } else if (kind_token == "move") {
        remove_const_qualifier(type_builder);
        remove_reference_suffix(type_builder);
    }

    trim_in_place(type_builder);
    if (type_builder.empty()) {
        type_builder = "auto";
    }
    return type_builder;
}

std::string transform_function_signature_line(const std::string& line, const std::vector<PatternData>& patterns) {
    size_t indent_end = 0;
    while (indent_end < line.size() && (line[indent_end] == ' ' || line[indent_end] == '\t')) {
        ++indent_end;
    }

    std::string indent = line.substr(0, indent_end);
    std::string content = line.substr(indent_end);
    std::string trimmed_content = trim_copy(content);
    if (trimmed_content.empty()) {
        return {};
    }

    size_t colon_pos = content.find(": (");
    if (colon_pos == std::string::npos) {
        colon_pos = content.find(":(");
        if (colon_pos == std::string::npos) {
            return {};
        }
    }

    size_t open_paren = content.find('(', colon_pos);
    if (open_paren == std::string::npos) {
        return {};
    }

    size_t close_paren = std::string::npos;
    int depth = 0;
    for (size_t i = open_paren; i < content.size(); ++i) {
        char ch = content[i];
        if (ch == '(') {
            depth++;
        } else if (ch == ')') {
            depth--;
            if (depth == 0) {
                close_paren = i;
                break;
            }
        }
    }
    if (close_paren == std::string::npos) {
        return {};
    }

    size_t eq_pos = content.find('=', close_paren);
    if (eq_pos == std::string::npos) {
        return {};
    }

    std::string name = trim_copy(content.substr(0, colon_pos));
    if (name.empty()) {
        return {};
    }

    std::string params_section = content.substr(open_paren + 1, close_paren - open_paren - 1);
    std::string between = content.substr(close_paren + 1, eq_pos - close_paren - 1);
    std::string between_trim = trim_copy(between);

    std::string body_fragment = content.substr(eq_pos + 1);
    std::string body_trim = trim_copy(body_fragment);
    if (body_trim.empty()) {
        return {};
    }

    bool is_expression_body = false;
    bool has_arrow = false;
    std::string cpp_return_type = "void";
    if (!between_trim.empty()) {
        if (between_trim.rfind("->", 0) == 0) {
            has_arrow = true;
            std::string return_spec = trim_copy(between_trim.substr(2));
            cpp_return_type = transform_return_spec(return_spec);
        } else {
            is_expression_body = true;
        }
    }

    if (body_trim.front() != '{') {
        is_expression_body = true;
    }

    if (is_expression_body && !has_arrow) {
        cpp_return_type = "auto";
    }

    std::string transformed_params = transform_parameter(params_section, patterns);

    std::string result = indent;
    result += cpp_return_type + " " + name + "(" + transformed_params + ")";

    if (is_expression_body) {
        std::string expr = body_trim;
        if (expr.front() == '=') {
            expr.erase(expr.begin());
            trim_in_place(expr);
        }
        if (!expr.empty() && expr.back() == ';') {
            expr.pop_back();
        }
        trim_in_place(expr);
        if (expr.empty()) {
            return {};
        }
        result += " { return " + expr + "; }";
    } else {
        std::string body_text = body_trim;
        size_t open_brace = body_text.find('{');
        size_t close_brace = body_text.rfind('}');
        if (open_brace != std::string::npos && close_brace != std::string::npos && close_brace > open_brace) {
            std::string inner = body_text.substr(open_brace + 1, close_brace - open_brace - 1);
            std::string transformed_inner = apply_recursive_transformations(inner, patterns, 1);
            trim_in_place(transformed_inner);
            if (transformed_inner.empty()) {
                result += " { }";
            } else {
                result += " { " + transformed_inner + " }";
            }
        } else {
            result += " " + body_trim;
        }
    }

    return result;
}

// PATTERN-DRIVEN TRANSFORMATION ENGINE (anchor-based substitution)

// Extract segment from text using anchor position and delimiters
std::string extract_segment(std::string_view text, size_t anchor_pos, const AnchorSegment& seg) {
    size_t start = anchor_pos;

    // Apply offset
    if (seg.offset_from_anchor < 0) {
        // Before anchor - find identifier before anchor
        size_t scan_end = anchor_pos;
        while (scan_end > 0 && std::isspace(text[scan_end - 1])) {
            --scan_end;
        }
        size_t ident_start = scan_end;
        while (ident_start > 0 && (std::isalnum(text[ident_start - 1]) || text[ident_start - 1] == '_')) {
            --ident_start;
        }
        if (ident_start < scan_end) {
            return std::string(text.substr(ident_start, scan_end - ident_start));
        }
        return "";
    }

    // Find delimiter start
    if (!seg.delimiter_start.empty()) {
        size_t delim_pos = text.find(seg.delimiter_start, start);
        if (delim_pos == std::string::npos) return "";
        start = delim_pos + seg.delimiter_start.length();
    }

    // Find delimiter end
    if (!seg.delimiter_end.empty()) {
        // For multi-char delimiters or non-nesting delimiters, use find
        if (seg.delimiter_end.size() > 1 || (seg.delimiter_start != "(" && seg.delimiter_start != "[" && seg.delimiter_start != "{")) {
            size_t end_pos = text.find(seg.delimiter_end, start);
            if (end_pos != std::string::npos) {
                return std::string(text.substr(start, end_pos - start));
            }
        } else {
            // Original nesting logic for brackets
            int depth = 1;
            size_t end = start;
            char open_char = seg.delimiter_start.empty() ? '\0' : seg.delimiter_start[0];
            char close_char = seg.delimiter_end[0];

            while (end < text.size() && depth > 0) {
                if (open_char != '\0' && text[end] == open_char) depth++;
                else if (text[end] == close_char) {
                    depth--;
                    if (depth == 0) break;
                }
                end++;
            }
            if (depth == 0) {
                return std::string(text.substr(start, end - start));
            }
        }
    }

    return std::string(text.substr(start));
}

// Apply substitution template with extracted segments
std::string apply_substitution(const std::string& template_str, const std::vector<std::string>& segments) {
    std::string result = template_str;

    // Pre-process segments
    std::vector<std::string> processed_segments = segments;
    for (size_t i = 0; i < processed_segments.size(); ++i) {
        auto& seg = processed_segments[i];

        // Don't trim body segment (i==3) - it needs spaces for formatting
        if (i != 3) {
            // Trim whitespace
            seg.erase(seg.begin(), std::find_if(seg.begin(), seg.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            seg.erase(std::find_if(seg.rbegin(), seg.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), seg.end());
        }

        // Transform return type: "-> int" -> "int", empty -> "auto" (only for return type segment)
        if (i == 2) {  // return_type segment
            if (seg.starts_with("->")) {
                seg = seg.substr(2); // Remove "->"
                seg.erase(seg.begin(), std::find_if(seg.begin(), seg.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            }
            if (seg.empty()) {
                seg = "auto"; // Default return type
            }
        }
    }

    // Replace placeholders with processed segments
    // Use 0-indexed ($0, $1, $2, $3) to match tblgen patterns
    for (size_t i = 0; i < processed_segments.size(); ++i) {
        std::string placeholder = "$" + std::to_string(i);
        size_t pos = 0;
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.length(), processed_segments[i]);
            pos += processed_segments[i].length();
        }
    }

    return result;
}

// Extract template parameters: "Pair<A,B>" → {"Pair", "A, B"}
std::pair<std::string, std::string> extract_template_params(const std::string& text) {
    size_t open = text.find('<');
    if (open == std::string::npos) {
        return {text, ""};
    }

    size_t close = text.rfind('>');
    if (close == std::string::npos || close <= open) {
        return {text, ""};
    }

    std::string name = text.substr(0, open);
    std::string params = text.substr(open + 1, close - open - 1);
    return {name, params};
}

// Generate template prefix: "A, B" → "template<typename A, typename B> "
std::string generate_template_prefix(const std::string& params) {
    if (params.empty()) return "";

    std::string result = "template<";
    size_t start = 0;
    for (size_t i = 0; i <= params.size(); ++i) {
        if (i == params.size() || params[i] == ',') {
            std::string param = params.substr(start, i - start);
            // Trim whitespace
            param.erase(0, param.find_first_not_of(" \t\n\r"));
            param.erase(param.find_last_not_of(" \t\n\r") + 1);
            if (!param.empty()) {
                if (result.back() != '<') result += ", ";
                result += "typename " + param;
            }
            start = i + 1;
        }
    }
    result += "> ";
    return result;
}

std::string apply_substitution_with_offset(const std::string& template_str, const std::vector<std::string>& segments, int placeholder_offset) {
    std::string result = template_str;

    // Replace placeholders with segments, accounting for offset
    for (size_t i = 0; i < segments.size(); ++i) {
        std::string placeholder = "$" + std::to_string(i + placeholder_offset);
        size_t pos = 0;
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.length(), segments[i]);
            pos += segments[i].length();
        }
    }

    return result;
}

// Template-aware substitution for template alias patterns
std::string apply_template_substitution(const std::string& template_str, const std::vector<std::string>& segments, int placeholder_offset) {
    std::vector<std::string> processed = segments;

    // Extract template params from segment[0] if present
    if (!processed.empty()) {
        auto [name, params] = extract_template_params(processed[0]);
        if (!params.empty()) {
            // Generate template prefix and modify segment[0] to just the name
            std::string prefix = generate_template_prefix(params);
            processed[0] = name;
            std::string result = prefix + template_str;

            // Now do substitution
            for (size_t i = 0; i < processed.size(); ++i) {
                std::string placeholder = "$" + std::to_string(i + placeholder_offset);
                size_t pos = 0;
                while ((pos = result.find(placeholder, pos)) != std::string::npos) {
                    result.replace(pos, placeholder.length(), processed[i]);
                    pos += processed[i].length();
                }
            }
            return result;
        }
    }

    // No template params, use normal substitution
    return apply_substitution_with_offset(template_str, segments, placeholder_offset);
}

// Transform parameter segment: "inout s: std::string" -> "std::string& s"
std::string transform_parameter(const std::string& param, const std::vector<PatternData>& patterns) {
    // Split by comma to handle multiple parameters
    std::vector<std::string> params;
    size_t start = 0;
    int depth = 0;
    for (size_t i = 0; i < param.size(); ++i) {
        if (param[i] == '(' || param[i] == '[' || param[i] == '{' || param[i] == '<') depth++;
        else if (param[i] == ')' || param[i] == ']' || param[i] == '}' || param[i] == '>') depth--;
        else if (param[i] == ',' && depth == 0) {
            params.push_back(param.substr(start, i - start));
            start = i + 1;
        }
    }
    if (start < param.size()) {
        params.push_back(param.substr(start));
    }

    // Transform each parameter individually
    std::vector<std::string> transformed;
    for (auto& p : params) {
        // Trim whitespace
        size_t s = 0;
        while (s < p.size() && std::isspace(static_cast<unsigned char>(p[s]))) ++s;
        size_t e = p.size();
        while (e > s && std::isspace(static_cast<unsigned char>(p[e - 1]))) --e;
        p = p.substr(s, e - s);

        std::string direct_result;
        if (transform_cpp2_parameter_direct(p, direct_result)) {
            transformed.push_back(direct_result);
            continue;
        }

        std::string result = p;
        // Find parameter patterns
        for (const auto& pattern : patterns) {
            if (pattern.name.find("parameter") == std::string::npos) continue;
            if (!pattern.use_alternating || pattern.alternating_anchors.empty()) continue;

            const std::string& anchor = pattern.alternating_anchors[0];
            size_t anchor_pos = p.find(anchor);
            if (anchor_pos != std::string::npos) {
                // Extract segments for this parameter pattern
                CPP2Emitter emitter;
                auto segments = emitter.extract_alternating_segments(p, pattern);

                if (segments.size() == pattern.evidence_types.size()) {
                    auto template_it = pattern.substitution_templates.find(2);
                    if (template_it != pattern.substitution_templates.end()) {
                        result = apply_substitution_with_offset(template_it->second, segments, 1);
                        break;
                    }
                }
            }
        }
        transformed.push_back(result);
    }

    // Join transformed parameters
    std::string output;
    for (size_t i = 0; i < transformed.size(); ++i) {
        if (i > 0) output += ", ";
        output += transformed[i];
    }
    return output;
}

// Transform nested function to lambda expression
std::string transform_to_lambda(const std::vector<std::string>& segments, const std::vector<PatternData>& patterns, int nesting_depth) {
    // segments can be:
    // [name, params, body] for void functions
    // [name, params, return_type, body] for functions with return type

    if (segments.size() < 3) {
        return ""; // Invalid
    }

    std::string name = segments[0];
    std::string params = segments[1];
    std::string return_type;
    std::string body;

    if (segments.size() == 3) {
        // Void function: [name, params, body]
        body = segments[2];
    } else {
        // Function with return type: [name, params, return_type, body]
        return_type = segments[2];
        body = segments[3];
    }

    std::cerr << "DEBUG transform_to_lambda: name='" << name << "' params='" << params << "' return_type='" << return_type << "' body='" << body << "' nesting_depth=" << nesting_depth << "\n";

    // Recursively transform the body with increased nesting depth
    std::string transformed_body = apply_recursive_transformations(body, patterns, nesting_depth + 1);

    // Transform parameters
    std::string transformed_params = transform_parameter(params, patterns);

    // Build lambda: auto name = [](params) -> return_type { body }; or auto name = [](params) { body };
    std::string lambda = "auto " + name + " = [](" + transformed_params + ")";
    if (!return_type.empty()) {
        lambda += " -> " + return_type;
    }
    lambda += " " + transformed_body + ";";

    return lambda;
}

bool is_enclosed_in_parens(const std::string& text) {
    if (text.size() < 2 || text.front() != '(' || text.back() != ')') {
        return false;
    }
    int depth = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        char ch = text[i];
        if (ch == '(') {
            depth++;
        } else if (ch == ')') {
            depth--;
            if (depth == 0 && i != text.size() - 1) {
                return false;
            }
            if (depth < 0) {
                return false;
            }
        }
    }
    return depth == 0;
}

bool transform_for_next_do_blocks(std::string& text, const std::vector<PatternData>& patterns, int nesting_depth) {
    std::vector<std::string> lines;
    size_t start = 0;
    bool ends_with_newline = false;
    while (start <= text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) {
            lines.emplace_back(text.substr(start));
            ends_with_newline = false;
            break;
        } else {
            lines.emplace_back(text.substr(start, end - start));
            start = end + 1;
            if (start == text.size()) {
                ends_with_newline = true;
                break;
            }
        }
    }

    bool changed = false;
    for (size_t i = 0; i + 2 < lines.size(); ++i) {
        const std::string& for_line = lines[i];
        std::string trimmed_for = trim_copy(for_line);
        if (trimmed_for.rfind("for ", 0) != 0) {
            continue;
        }

        size_t indent_len = for_line.find_first_not_of(" \t");
        if (indent_len == std::string::npos) {
            indent_len = 0;
        }
        std::string indent_prefix = for_line.substr(0, indent_len);
        std::string range_expr = trim_copy(trimmed_for.substr(3));
        if (range_expr.empty()) {
            continue;
        }

        const std::string& next_line = lines[i + 1];
        std::string trimmed_next = trim_copy(next_line);
        if (trimmed_next.rfind("next ", 0) != 0) {
            continue;
        }
        std::string next_expr = trim_copy(trimmed_next.substr(5));
        if (next_expr.empty()) {
            continue;
        }

        const std::string& do_line = lines[i + 2];
        std::string trimmed_do = trim_copy(do_line);
        if (trimmed_do.rfind("do", 0) != 0) {
            continue;
        }

        size_t paren_open = do_line.find('(');
        size_t paren_close = do_line.rfind(')');
        if (paren_open == std::string::npos || paren_close == std::string::npos || paren_close <= paren_open) {
            continue;
        }

        std::string raw_param = do_line.substr(paren_open + 1, paren_close - paren_open - 1);
        std::string param_expr = trim_copy(raw_param);
        std::string loop_param = transform_parameter(param_expr, patterns);
        if (loop_param.empty()) {
            loop_param = "auto&& item";
        }

        size_t body_start = i + 3;
        size_t body_end = body_start;
        while (body_end < lines.size()) {
            std::string trimmed_body = trim_copy(lines[body_end]);
            size_t indent_body = lines[body_end].find_first_not_of(" \t");
            if (indent_body == std::string::npos) {
                indent_body = lines[body_end].size();
            }
            if (!trimmed_body.empty() && indent_body <= indent_len) {
                break;
            }
            ++body_end;
        }

        if (body_start >= body_end) {
            continue;
        }

        std::ostringstream raw_body;
        for (size_t k = body_start; k < body_end; ++k) {
            raw_body << lines[k];
            if (k + 1 < body_end) {
                raw_body << '\n';
            }
        }

        std::string transformed_body = apply_recursive_transformations(raw_body.str(), patterns, nesting_depth + 1);

        std::vector<std::string> body_lines;
        {
            std::istringstream body_stream(transformed_body);
            std::string line;
            while (std::getline(body_stream, line)) {
                body_lines.push_back(line);
            }
        }

        size_t min_body_indent = std::numeric_limits<size_t>::max();
        for (const auto& body_line : body_lines) {
            std::string trimmed = trim_copy(body_line);
            if (trimmed.empty()) {
                continue;
            }
            size_t indent = body_line.find_first_not_of(" \t");
            if (indent == std::string::npos) {
                indent = body_line.size();
            }
            if (indent < min_body_indent) {
                min_body_indent = indent;
            }
        }
        if (min_body_indent == std::numeric_limits<size_t>::max()) {
            min_body_indent = 0;
        }

        std::vector<std::string> new_lines;
        new_lines.push_back(indent_prefix + "for (" + loop_param + " : " + range_expr + ") {");

        std::string child_indent = indent_prefix + "    ";
        for (const auto& body_line : body_lines) {
            std::string trimmed = trim_copy(body_line);
            if (trimmed.empty()) {
                new_lines.push_back(child_indent);
                continue;
            }
            std::string adjusted = body_line;
            if (min_body_indent > 0 && adjusted.size() >= min_body_indent) {
                adjusted.erase(0, min_body_indent);
            }
            adjusted = trim_copy(adjusted);
            new_lines.push_back(child_indent + adjusted);
        }

        if (!next_expr.empty()) {
            std::string normalized_next = next_expr;
            if (normalized_next.back() != ';') {
                normalized_next += ';';
            }
            new_lines.push_back(child_indent + normalized_next);
        }

        new_lines.push_back(indent_prefix + "}");

        lines.erase(lines.begin() + static_cast<long>(i), lines.begin() + static_cast<long>(body_end));
        lines.insert(lines.begin() + static_cast<long>(i), new_lines.begin(), new_lines.end());
        changed = true;
        i += new_lines.size() - 1;
    }

    if (!changed) {
        return false;
    }

    std::ostringstream rebuilt;
    for (size_t idx = 0; idx < lines.size(); ++idx) {
        rebuilt << lines[idx];
        if (idx + 1 < lines.size() || ends_with_newline) {
            rebuilt << '\n';
        }
    }
    text = rebuilt.str();
    if (!ends_with_newline && !text.empty() && text.back() == '\n') {
        text.pop_back();
    }

    return true;
}

// Recursive orbit transformation for nested segments
std::string apply_recursive_transformations(const std::string& input, const std::vector<PatternData>& patterns, int nesting_depth) {
    std::cerr << "DEBUG apply_recursive_transformations: input='" << input << "' nesting_depth=" << nesting_depth << "\n";
    std::string result = input;

    // If input is wrapped in braces, recursively transform the contents
    std::string working = result;
    bool has_braces = false;
    if (!working.empty() && working.front() == '{' && working.back() == '}') {
        working = working.substr(1, working.size() - 2);
        has_braces = true;
        std::cerr << "DEBUG apply_recursive_transformations: stripped braces, working='" << working << "'\n";
    }

    // Try each pattern to see if it matches (try longer anchors first to avoid substring conflicts)
    std::vector<const PatternData*> sorted_patterns;
    for (const auto& pattern : patterns) {
        if (pattern.use_alternating && !pattern.alternating_anchors.empty()) {
            sorted_patterns.push_back(&pattern);
        }
    }
    std::sort(sorted_patterns.begin(), sorted_patterns.end(), [](const PatternData* a, const PatternData* b) {
        // First, prefer patterns with more anchors (more specific)
        if (a->alternating_anchors.size() != b->alternating_anchors.size()) {
            return a->alternating_anchors.size() > b->alternating_anchors.size();
        }
        // Then prefer longer first anchors (to avoid substring conflicts like := vs :)
        return a->alternating_anchors[0].length() > b->alternating_anchors[0].length();
    });

    // Process ALL matches, not just the first one
    bool made_change = true;
    while (made_change) {
        made_change = false;
        if (transform_for_next_do_blocks(working, patterns, nesting_depth)) {
            made_change = true;
            continue;
        }
        for (const auto* pattern_ptr : sorted_patterns) {
            const auto& pattern = *pattern_ptr;
            if (pattern.use_alternating && !pattern.alternating_anchors.empty()) {
                // Skip parameter patterns in recursive transformation
                if (pattern.name.find("parameter") != std::string::npos) {
                    continue;
                }

                // Try alternating pattern
                const std::string& anchor = pattern.alternating_anchors[0];
                size_t anchor_pos = working.find(anchor);
                if (anchor_pos != std::string::npos) {
                    // Check if this is a real match or a substring of a longer anchor
                    // e.g., don't match ":" if it's part of ":="
                    bool is_substring = false;
                    if (anchor_pos + anchor.length() < working.length()) {
                        char next_ch = working[anchor_pos + anchor.length()];
                        // If anchor is ":" and next char is "=" or "type", it's part of a longer anchor
                        if (anchor == ":" && (next_ch == '=' || working.substr(anchor_pos).find(": type") == 0)) {
                            is_substring = true;
                        }
                    }

                    // Also check if this is the C++ :: scope operator
                    if (anchor == ":" && anchor_pos > 0 && working[anchor_pos - 1] == ':') {
                        is_substring = true; // Skip :: in std::string etc
                    }
                    if (anchor == ":" && anchor_pos + 1 < working.length() && working[anchor_pos + 1] == ':') {
                        is_substring = true; // Skip :: in std::string etc
                    }

                    if (!is_substring) {
                        // Pattern might match - extract segments using depth matcher (FIXED)
                        auto [segments, end_pos] = DepthPatternMatcher::extract_segments(working, pattern, anchor_pos);

                        if (segments.size() == pattern.evidence_types.size()) {
                            // Check if this is a nested function declaration - transform to lambda instead
                            bool is_function_pattern = (pattern.name.find("function") != std::string::npos);
                            bool is_main_pattern = (pattern.name.find("main") != std::string::npos);
                            // CRITICAL: Also check actual function NAME in segments, not just pattern name
                            bool is_main_function = (segments.size() >= 1 && segments[0].find("main") != std::string::npos);

                            std::string transformed;
                            // Only transform to lambda if: (1) it's a function pattern, (2) not main, (3) nesting_depth > 0 (we're inside a function)
                            if (is_function_pattern && !is_main_pattern && !is_main_function && nesting_depth > 0 && segments.size() >= 3) {
                                // Nested function: transform to lambda assignment
                                // segments: [name, parameters, body] or [name, parameters, return_type, body]
                                std::cerr << "DEBUG: Transforming nested function '" << segments[0] << "' to lambda (depth=" << nesting_depth << "), segments.size()=" << segments.size() << "\n";
                                for (size_t i = 0; i < segments.size(); ++i) {
                                    std::cerr << "  segment[" << i << "]='" << segments[i] << "'\n";
                                }
                                transformed = transform_to_lambda(segments, patterns, nesting_depth);
                            } else {
                                // Found a match - apply normal transformation
                                auto template_it = pattern.substitution_templates.find(2); // CPP target
                                if (template_it != pattern.substitution_templates.end()) {
                                    transformed = apply_substitution_with_offset(template_it->second, segments, 1);
                                    // Add semicolon only for cpp2_variable (walrus operator), not typed_variable
                                    // typed_variable already has the semicolon in original source
                                    if (pattern.name == "cpp2_variable" || pattern.name.find("alias") != std::string::npos) {
                                        transformed += ";";
                                    }
                                } else {
                                    continue; // No template for this pattern
                                }
                            }
                            if (!transformed.empty()) {
                                // Replace the matched portion with transformed text
                                size_t match_start = anchor_pos;
                                // Find the start of the statement (look backward for statement boundaries)
                                while (match_start > 0 && std::isspace(working[match_start - 1])) match_start--;
                                while (match_start > 0) {
                                    char ch = working[match_start - 1];
                                    if (ch == '{' || ch == ';' || ch == ',') break;
                                    match_start--;
                                }
                                // Skip leading whitespace
                                while (match_start < anchor_pos && std::isspace(working[match_start])) match_start++;

                                // Check if we need to skip trailing semicolon to avoid duplication
                                size_t actual_end = end_pos;
                                while (actual_end < working.length() && std::isspace(working[actual_end])) actual_end++;
                                if (actual_end < working.length() && working[actual_end] == ';') {
                                    actual_end++; // Skip the semicolon since we're adding it
                                }

                                // Preserve spacing after transformed statement
                                std::string suffix = working.substr(actual_end);
                                // Trim leading whitespace from suffix to normalize
                                size_t suffix_start = 0;
                                while (suffix_start < suffix.length() && std::isspace(suffix[suffix_start])) suffix_start++;
                                suffix = suffix.substr(suffix_start);

                                // Add space before next token if needed (but not for function patterns, they already have semicolon)
                                if (!is_function_pattern && !suffix.empty() && !std::isspace(suffix[0])) {
                                    transformed += " "; // Add space before next statement/closing brace
                                }

                                working = working.substr(0, match_start) + transformed + suffix;
                                made_change = true;
                                break; // Restart pattern search after transformation
                            }
                        }
                    }
                }
            }
            if (made_change) break; // Restart from first pattern
        }
    }

    // Restore braces if we had them
    if (has_braces) {
        if (working.empty()) {
            result = "{}"; // Empty braces, no spaces
        } else {
            // Normalize whitespace: single space after { and before }
            std::string trimmed = working;
            // Trim leading whitespace
            size_t start = 0;
            while (start < trimmed.length() && std::isspace(trimmed[start])) start++;
            // Trim trailing whitespace
            size_t end = trimmed.length();
            while (end > start && std::isspace(trimmed[end - 1])) end--;
            trimmed = trimmed.substr(start, end - start);
            result = "{ " + trimmed + " }";
        }
    } else {
        result = working;
    }

    return result;
}

} // anonymous namespace

// Extract segments for alternating anchor/evidence patterns
std::vector<std::string> CPP2Emitter::extract_alternating_segments(std::string_view text, const PatternData& pattern) const {
    std::vector<std::string> segments;

    if (pattern.alternating_anchors.empty()) {
        return segments;
    }

    // Build confix depth map to validate pattern boundaries
    auto depth_map = build_depth_map(text);

    // Find the first anchor
    const std::string& first_anchor = pattern.alternating_anchors[0];
    size_t anchor_pos = text.find(first_anchor);
    if (anchor_pos == std::string::npos) {
        return segments;
    }

    // Record the confix depth at pattern start
    int pattern_start_depth = (anchor_pos < depth_map.size()) ? depth_map[anchor_pos] : 0;

    // Special case: if only one anchor and 2 evidence spans, extract before AND after
    if (pattern.alternating_anchors.size() == 1 && pattern.evidence_types.size() == 2) {
        // Evidence before anchor
        std::string before = std::string(text.substr(0, anchor_pos));
        before.erase(before.begin(), std::find_if(before.begin(), before.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        before.erase(std::find_if(before.rbegin(), before.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), before.end());
        segments.push_back(before);

        // Evidence after anchor
        size_t after_start = anchor_pos + first_anchor.length();
        std::string after = std::string(text.substr(after_start));
        after.erase(after.begin(), std::find_if(after.begin(), after.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        after.erase(std::find_if(after.rbegin(), after.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), after.end());
        if (!after.empty() && after.back() == ';') {
            after.pop_back();
        }
        segments.push_back(after);

        return segments;
    }

    // Check if we need evidence BEFORE first anchor
    // This happens when we have N+1 evidence types for N anchors
    size_t evidence_start_idx = 0;
    if (pattern.evidence_types.size() > pattern.alternating_anchors.size()) {
        // Extract evidence before first anchor
        std::string before = std::string(text.substr(0, anchor_pos));
        before.erase(before.begin(), std::find_if(before.begin(), before.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        before.erase(std::find_if(before.rbegin(), before.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), before.end());
        segments.push_back(before);
        evidence_start_idx = 1;
    }

    size_t current_pos = anchor_pos + first_anchor.length();

    // Extract evidence spans between anchors
    for (size_t i = evidence_start_idx; i < pattern.evidence_types.size(); ++i) {
        // Find next anchor or end
        size_t next_anchor_pos = std::string::npos;
        size_t anchor_idx = i - evidence_start_idx + 1; // Next anchor index
        if (anchor_idx < pattern.alternating_anchors.size()) {
            const std::string& next_anchor = pattern.alternating_anchors[anchor_idx];
            next_anchor_pos = text.find(next_anchor, current_pos);

            // If this anchor is required but not found, fail the match
            if (next_anchor_pos == std::string::npos) {
                return {}; // Anchor not found - pattern doesn't match
            }
        }

        size_t evidence_end = (next_anchor_pos != std::string::npos) ? next_anchor_pos : text.length();
        std::string evidence = std::string(text.substr(current_pos, evidence_end - current_pos));

        // Trim whitespace
        evidence.erase(evidence.begin(), std::find_if(evidence.begin(), evidence.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        evidence.erase(std::find_if(evidence.rbegin(), evidence.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), evidence.end());

        // Strip trailing semicolon if present (template will add it back)
        if (!evidence.empty() && evidence.back() == ';') {
            evidence.pop_back();
        }

        // Validate confix balance - if close > open, we crossed a boundary
        int balance = 0;
        for (char ch : evidence) {
            if (ch == '(' || ch == '[' || ch == '{' || ch == '<') balance++;
            if (ch == ')' || ch == ']' || ch == '}' || ch == '>') {
                balance--;
                if (balance < 0) {
                    return {}; // Crossed scope boundary - invalid span
                }
            }
        }

        segments.push_back(evidence);
        current_pos = evidence_end;

        if (next_anchor_pos != std::string::npos && anchor_idx < pattern.alternating_anchors.size()) {
            current_pos += pattern.alternating_anchors[anchor_idx].length();
        }
    }

    return segments;
}

// Extract std:: types and generate includes
std::vector<std::string> extract_std_includes(std::string_view source) {
    std::vector<std::string> includes;

    auto add_include = [&](const std::string& include_line) {
        if (std::find(includes.begin(), includes.end(), include_line) == includes.end()) {
            includes.push_back(include_line);
        }
    };

    auto has_out_parameter = [](std::string_view text) {
        auto is_prefix = [](char ch) {
            return std::isspace(static_cast<unsigned char>(ch)) || ch == '(' || ch == ',';
        };
        auto is_suffix = [](char ch) {
            return std::isspace(static_cast<unsigned char>(ch)) || ch == ':';
        };

        size_t pos = text.find("out");
        while (pos != std::string::npos) {
            bool prefix_ok = (pos == 0) ? false : is_prefix(text[pos - 1]);
            bool suffix_ok = (pos + 3 >= text.size()) ? true : is_suffix(text[pos + 3]);
            if (prefix_ok && suffix_ok) {
                return true;
            }
            pos = text.find("out", pos + 3);
        }
        return false;
    };

    // Simple mapping of std::Type to <header>
    struct StdMapping {
        const char* type;
        const char* header;
    };

    static const StdMapping mappings[] = {
        {"std::cout", "iostream"},
        {"std::cin", "iostream"},
        {"std::cerr", "iostream"},
        {"std::string", "string"},
        {"std::vector", "vector"},
        {"std::map", "map"},
        {"std::set", "set"},
        {"std::unordered_map", "unordered_map"},
        {"std::unordered_set", "unordered_set"},
        {"std::list", "list"},
        {"std::deque", "deque"},
        {"std::array", "array"},
    };

    for (const auto& mapping : mappings) {
        if (source.find(mapping.type) != std::string::npos) {
            add_include(std::string("#include <") + mapping.header + ">" );
        }
    }

    if (source.find("cpp2::impl::out<") != std::string::npos || has_out_parameter(source)) {
        add_include("#include \"cpp2_inline.h\"");
    }

    return includes;
}

// Depth-based emit: deterministic pattern matching without speculation
void CPP2Emitter::emit_depth_based(std::string_view source, std::ostream& out, const std::vector<PatternData>& patterns) const {
    // Emit includes first
    auto includes = extract_std_includes(source);
    for (const auto& include : includes) {
        out << include << "\n";
    }
    // Split on newlines for line-by-line processing
    std::vector<std::string_view> lines;
    size_t line_start = 0;
    for (size_t i = 0; i <= source.size(); ++i) {
        if (i == source.size() || source[i] == '\n') {
            if (i > line_start) {
                lines.push_back(source.substr(line_start, i - line_start));
            }
            line_start = i + 1;
        }
    }

    // Process each line separately
    bool first = true;
    for (const auto& line : lines) {
        std::cerr << "DEBUG emit_depth_based: Processing line: '" << line << "'\n";
        if (!first) out << "\n";
        first = false;

        auto matches = DepthPatternMatcher::find_matches(line, patterns);
        std::cerr << "DEBUG emit_depth_based: Found " << matches.size() << " matches\n";

        if (matches.empty()) {
            std::string transformed_line = transform_function_signature_line(std::string(line), patterns);
            if (!transformed_line.empty()) {
                out << transformed_line;
            } else {
                out << line;
            }
            continue;
        }

        size_t last_pos = 0;

        // Apply transformations in order (innermost first, as returned by find_matches)
        for (const auto& match : matches) {
            // Emit text before this match
            if (last_pos < match.start_pos) {
                out << line.substr(last_pos, match.start_pos - last_pos);
            }

            // Apply substitution for this match
            int target_grammar = 2; // CPP target
            auto template_it = match.pattern->substitution_templates.find(target_grammar);

            if (template_it != match.pattern->substitution_templates.end()) {
                // Recursively transform segments (skip for template patterns - they have type expressions)
                std::vector<std::string> transformed_segments = match.segments;
                bool is_template = (match.pattern->name.find("template") != std::string::npos);
                bool is_function = (match.pattern->name.find("function") != std::string::npos);

                if (!is_template) {
                    for (size_t i = 0; i < transformed_segments.size(); ++i) {
                        auto& seg = transformed_segments[i];

                        // Check if this segment is parameters based on evidence type
                        bool is_parameters = (i < match.pattern->evidence_types.size() &&
                                            match.pattern->evidence_types[i] == "parameters");

                        if (is_function && is_parameters) {
                            seg = transform_parameter(seg, patterns);
                        } else {
                            // Top-level function body gets nesting_depth=1 (inside first function)
                            seg = apply_recursive_transformations(seg, patterns, 1);
                        }
                    }
                }

                // Apply substitution template (use template-aware version for template patterns)
                int placeholder_offset = match.pattern->use_alternating ? 1 : 0;
                std::string result;
                if (is_template) {
                    result = apply_template_substitution(template_it->second, transformed_segments, placeholder_offset);
                } else {
                    result = apply_substitution_with_offset(template_it->second, transformed_segments, placeholder_offset);
                }
                out << result;
            } else {
                // No template for target grammar - emit original
                out << line.substr(match.start_pos, match.end_pos - match.start_pos);
        }

            last_pos = match.end_pos;
        }

        // Emit remaining text from this line
        if (last_pos < line.size()) {
            out << line.substr(last_pos);
        }
    }
}

void CPP2Emitter::emit(OrbitIterator& iterator, std::string_view source, std::ostream& out, const std::vector<PatternData>& patterns) const {
    // Emit includes first
    auto includes = extract_std_includes(source);
    for (const auto& include : includes) {
        out << include << "\n";
    }

    // Reset iterator to beginning
    iterator.reset();

    size_t last_pos = 0;

    // Iterate through all orbits and emit their results
    size_t orbit_count = 0;
    for (cppfort::stage0::Orbit* orbit = iterator.next(); orbit; orbit = iterator.next()) {
        if (auto* confix = dynamic_cast<cppfort::stage0::ConfixOrbit*>(orbit)) {
            orbit_count++;
            std::cerr << "DEBUG emit: Orbit #" << orbit_count << " [" << confix->start_pos << ", " << confix->end_pos << ") pattern=" << confix->selected_pattern() << "\n";
            std::cerr << "DEBUG emit: Orbit text: '" << source.substr(confix->start_pos, std::min(size_t(50), confix->end_pos - confix->start_pos)) << "...'\n";

            // Fill gap between last position and current orbit
            if (last_pos < confix->start_pos) {
                std::string gap = std::string(source.substr(last_pos, confix->start_pos - last_pos));
                std::cerr << "DEBUG emit: Gap [" << last_pos << ", " << confix->start_pos << "): '" << gap << "'\n";
                out << source.substr(last_pos, confix->start_pos - last_pos);
            }

            // Find pattern for this orbit
            const PatternData* pattern = nullptr;
            std::string pattern_name = confix->selected_pattern();

            for (const auto& p : patterns) {
                if (p.name == pattern_name) {
                    pattern = &p;
                }
            }

            // Emit the orbit
            emit_orbit(*confix, source, out, pattern, patterns);
            last_pos = confix->end_pos;
        }
    }
    std::cerr << "DEBUG emit: Total orbits processed: " << orbit_count << "\n";

    // Fill remaining text after last orbit
    if (last_pos < source.size()) {
        out << source.substr(last_pos);
    }
}

void CPP2Emitter::emit_fragment(const OrbitFragment& fragment, std::string_view source, std::ostream& out) const {
    // Deprecated - use emit_orbit instead
    out << extract_fragment_text(fragment, source);
}

void CPP2Emitter::emit_orbit(const ConfixOrbit& orbit, std::string_view source, std::ostream& out, const PatternData* pattern, const std::vector<PatternData>& all_patterns) const {
    if (orbit.start_pos >= source.size() || orbit.end_pos > source.size() || orbit.start_pos >= orbit.end_pos) {
        return;
    }

    std::string_view text = source.substr(orbit.start_pos, orbit.end_pos - orbit.start_pos);

    // Filter out whitespace-only orbits (garbage from over-speculation)
    bool is_whitespace_only = true;
    for (char c : text) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            is_whitespace_only = false;
            break;
        }
    }
    if (is_whitespace_only) {
        out << text; // Emit as-is, don't try to transform
        return;
    }

    // ORBIT RECURSION MUST TERMINATE
    if (orbit.confidence == 0.0) {
        // For orbits with zero confidence, emit original text unchanged
        out << text;
        return;
    }

    // PATTERN-DRIVEN TRANSFORMATION
    if (!pattern) {
        // Pattern wasn't found for this orbit; emit original text and continue.
        out << text;
        return;
    }

    if (pattern->use_alternating) {
        if (pattern->alternating_anchors.empty() || pattern->evidence_types.empty() || pattern->substitution_templates.empty()) {
            std::cerr << "FATAL: Alternating pattern missing required fields\n";
            std::exit(1);
        }
    } else {
        if (pattern->segments.empty() || pattern->substitution_templates.empty()) {
            std::cerr << "FATAL: Segment-based pattern has no transformation data\n";
            std::exit(1);
        }
    }

    std::vector<std::string> segments;

    if (pattern->use_alternating) {
        // Extract evidence spans for alternating anchor/evidence pattern
        segments = extract_alternating_segments(text, *pattern);
    } else {
        // Try pattern matching for function_declaration pattern
        if (pattern->name == "function_declaration" && !pattern->signature_patterns.empty()) {
            // Use tblgen pattern matcher for CPP2 pattern
            std::string cpp2_pattern = "$0: ($1) -> $2 = $3";
            auto match_result = TblgenPatternMatcher::match(cpp2_pattern, std::string(text));

            if (match_result) {
                segments = *match_result;
            } else {
                // Pattern match failed, emit original
                out << text;
                return;
            }
        } else {
            // Fallback to old segment extraction for other patterns
            size_t anchor_pos = std::string::npos;
            for (const auto& sig : pattern->signature_patterns) {
                anchor_pos = text.find(sig);
                if (anchor_pos != std::string::npos) break;
            }

            if (anchor_pos == std::string::npos) {
                out << text;
                return;
            }

            for (const auto& seg_def : pattern->segments) {
                std::string seg = extract_segment(text, anchor_pos, seg_def);
                segments.push_back(seg);
            }
        }
    }

    // Apply recursive transformations to segments BEFORE substitution
    // This handles nested CPP2 syntax like variable declarations in function bodies
    bool is_function = (pattern->name.find("function") != std::string::npos);
    for (size_t i = 0; i < segments.size(); ++i) {
        auto& seg = segments[i];
        // Check if this segment is parameters based on evidence type
        bool is_parameters = (i < pattern->evidence_types.size() &&
                            pattern->evidence_types[i] == "parameters");

        if (is_function && is_parameters) {
            seg = transform_parameter(seg, all_patterns);
        } else {
            // Top-level function body gets nesting_depth=1 (inside first function)
            seg = apply_recursive_transformations(seg, all_patterns, 1);
        }
    }

    // Get target grammar (CPP=2 for now)
    int target_grammar = 2;
    auto template_it = pattern->substitution_templates.find(target_grammar);
    if (template_it == pattern->substitution_templates.end()) {
        std::cerr << "FATAL: No substitution template for grammar " << target_grammar << "\n";
        std::exit(1);
    }

    // Apply substitution (segments already transformed above)
    // Alternating patterns use 1-indexed placeholders ($1, $2), function patterns use 0-indexed ($0, $1)
    int placeholder_offset = pattern->use_alternating ? 1 : 0;
    std::string result = apply_substitution_with_offset(template_it->second, segments, placeholder_offset);
    out << result;
}

std::string_view CPP2Emitter::extract_fragment_text(const OrbitFragment& fragment, std::string_view source) const {
    if (fragment.start_pos >= source.size() || fragment.end_pos > source.size() ||
        fragment.start_pos >= fragment.end_pos) {
        return {};
    }
    return source.substr(fragment.start_pos, fragment.end_pos - fragment.start_pos);
}

namespace testing {

std::string transform_parameter_for_testing(std::string_view param) {
    std::string original{param};
    std::string transformed;
    if (transform_cpp2_parameter_direct(original, transformed)) {
        return transformed;
    }
    return std::string{param};
}

} // namespace testing

} // namespace cppfort::stage0

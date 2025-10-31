#include "cpp2_emitter.h"
#include "unified_pattern_matcher.h"
#include "confix_tracker.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "confix_orbit.h"
#include "pattern_loader.h"

namespace cppfort::stage0 {

namespace {

#ifdef CPP2_EMITTER_ENABLE_DEBUG
constexpr bool kEmitterDebugEnabled = true;
#else
constexpr bool kEmitterDebugEnabled = false;
#endif

#define CPP2_EMITTER_DEBUG(statement) \
    do { if (kEmitterDebugEnabled) { statement; } } while (false)

// Forward declarations
std::string apply_recursive_transformations(const std::string& input, const std::vector<PatternData>& patterns, int nesting_depth = 0);
std::string transform_parameter(const std::string& param, const std::vector<PatternData>& patterns, bool prefer_value_for_in = false);
std::pair<std::string, std::string> extract_template_params(const std::string& text);
std::string generate_template_prefix(const std::string& params);
std::string transform_statement(const std::string& statement, const std::vector<PatternData>& patterns, int nesting_depth);
std::string rewrite_statement_from_match(const std::string& statement, const UnifiedPatternMatch& match, const std::vector<PatternData>& patterns, int nesting_depth);
std::string map_cpp2_type(std::string type);

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

bool is_control_keyword(std::string_view word) {
    static const std::unordered_set<std::string> control_keywords = {
        "if", "for", "while", "switch", "catch"
    };
    return control_keywords.count(std::string(word)) > 0;
}

std::string extract_function_name(std::string_view signature_prefix) {
    size_t end = signature_prefix.size();
    while (end > 0 && std::isspace(static_cast<unsigned char>(signature_prefix[end - 1]))) {
        --end;
    }
    size_t start = end;
    while (start > 0) {
        char ch = signature_prefix[start - 1];
        if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '~') {
            --start;
        } else {
            break;
        }
    }
    if (start == end) {
        return {};
    }
    return std::string(signature_prefix.substr(start, end - start));
}

size_t find_insertion_after_includes(const std::string& text) {
    size_t search_pos = 0;
    size_t last_include_end = 0;
    bool found_include = false;
    while (true) {
        size_t include_pos = text.find("#include", search_pos);
        if (include_pos == std::string::npos) {
            break;
        }
        found_include = true;
        size_t line_end = text.find('\n', include_pos);
        if (line_end == std::string::npos) {
            last_include_end = text.size();
            return last_include_end;
        }
        last_include_end = line_end + 1;
        search_pos = line_end + 1;
    }
    if (!found_include) {
        return 0;
    }
    // Skip any blank lines immediately following the include block
    size_t insertion = last_include_end;
    while (insertion < text.size() && (text[insertion] == '\n' || text[insertion] == '\r')) {
        ++insertion;
    }
    return insertion;
}

std::vector<std::string> collect_top_level_function_prototypes(const std::string& text) {
    std::vector<std::string> prototypes;
    std::unordered_set<std::string> seen_signatures;

    size_t current = 0;
    int depth = 0;

    while (current < text.size()) {
        size_t line_end = text.find('\n', current);
        if (line_end == std::string::npos) {
            line_end = text.size();
        }
        std::string line = text.substr(current, line_end - current);
        std::string trimmed = trim_copy(line);
        int depth_before_line = depth;

        if (depth_before_line == 0 && !trimmed.empty() && trimmed[0] != '#') {
            size_t brace_pos = line.find('{');
            if (brace_pos != std::string::npos) {
                std::string signature = trim_copy(line.substr(0, brace_pos));
                size_t paren_pos = signature.find('(');
                size_t equals_pos = signature.find('=');
                if (paren_pos != std::string::npos &&
                    (equals_pos == std::string::npos || equals_pos > paren_pos) &&
                    signature.find("operator") == std::string::npos &&
                    signature.find("class") == std::string::npos &&
                    signature.find("struct") == std::string::npos &&
                    signature.find("enum") == std::string::npos &&
                    signature.find("namespace") == std::string::npos) {

                    std::string name = extract_function_name(signature.substr(0, paren_pos));
                    if (!name.empty() && !is_control_keyword(name) && name != "main") {
                        if (seen_signatures.insert(signature).second) {
                            std::string prototype = signature;
                            if (!prototype.empty() && prototype.back() == '\r') {
                                prototype.pop_back();
                            }
                            prototype.append(";");
                            prototypes.push_back(std::move(prototype));
                        }
                    }
                }
            }
        }

        for (char ch : line) {
            if (ch == '{') {
                ++depth;
            } else if (ch == '}') {
                if (depth > 0) {
                    --depth;
                }
            }
        }

        if (line_end == text.size()) {
            break;
        }
        current = line_end + 1;
    }

    return prototypes;
}

std::string finalize_cpp2_output(std::string text) {
    constexpr std::string_view cpp2_header = "#include \"cpp2util.h\"\n";

    if (text.find("cpp2util.h") == std::string::npos) {
        size_t insert_pos = find_insertion_after_includes(text);
        std::string header_line(cpp2_header);
        if (insert_pos == 0) {
            text.insert(0, header_line + "\n");
        } else {
            text.insert(insert_pos, header_line);
        }
    }

    auto prototypes = collect_top_level_function_prototypes(text);
    if (!prototypes.empty()) {
        size_t insert_pos = find_insertion_after_includes(text);
        std::string block;
        if (insert_pos > 0 && insert_pos <= text.size() && text[insert_pos - 1] != '\n') {
            block.push_back('\n');
        }
        for (const auto& proto : prototypes) {
            block.append(proto);
            block.push_back('\n');
        }
        block.push_back('\n');
        text.insert(insert_pos, block);
    }

    return text;
}

size_t find_keyword_outside(std::string_view text, std::string_view keyword, size_t start_pos) {
    bool in_string = false;
    bool in_char = false;
    bool escape = false;

    for (size_t i = start_pos; i + keyword.size() <= text.size(); ++i) {
        char ch = text[i];

        if (escape) {
            escape = false;
            continue;
        }

        if (in_string) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (in_char) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '\'') {
                in_char = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
            continue;
        }

        if (ch == '\'') {
            in_char = true;
            continue;
        }

        if (text.compare(i, keyword.size(), keyword) == 0) {
            bool boundary_before = (i == 0) || (!std::isalnum(static_cast<unsigned char>(text[i - 1])) && text[i - 1] != '_');
            char after_char = (i + keyword.size() < text.size()) ? text[i + keyword.size()] : ' ';
            bool boundary_after = (i + keyword.size() >= text.size()) || (!std::isalnum(static_cast<unsigned char>(after_char)) && after_char != '_');
            if (boundary_before && boundary_after) {
                return i;
            }
        }
    }

    return std::string::npos;
}

std::string rewrite_using_namespace_placeholder(std::string statement) {
    std::string trimmed = trim_copy(statement);
    if (!trimmed.starts_with("using ")) {
        return statement;
    }

    const std::string target_prefix = "using std::";
    if (!trimmed.starts_with(target_prefix)) {
        return statement;
    }

    if (!trimmed.ends_with("_ ;") && !trimmed.ends_with("_ ;") && !trimmed.ends_with("_;") ) {
        return statement;
    }

    size_t literal_pos = trimmed.find("::_");
    if (literal_pos == std::string::npos) {
        return statement;
    }

    std::string ns = trimmed.substr(6, literal_pos - 6);
    ns = trim_copy(ns);
    if (ns.empty()) {
        return statement;
    }

    std::string indent;
    size_t first_non_ws = statement.find_first_not_of(" \t");
    if (first_non_ws != std::string::npos) {
        indent = statement.substr(0, first_non_ws);
    }

    std::string rebuilt = indent + "using namespace " + ns + ";";
    return rebuilt;
}

std::string rewrite_default_placeholders(std::string statement) {
    std::string result;
    result.reserve(statement.size());

    bool changed = false;
    bool in_string = false;
    bool in_char = false;
    bool escape = false;
    int paren_depth = 0;
    int brace_depth = 0;
    int bracket_depth = 0;

    for (size_t i = 0; i < statement.size(); ++i) {
        char ch = statement[i];

        result.push_back(ch);

        if (escape) {
            escape = false;
            continue;
        }

        if (in_string) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (in_char) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '\'') {
                in_char = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
            continue;
        }

        if (ch == '\'') {
            in_char = true;
            continue;
        }

        if (ch == '(') {
            ++paren_depth;

            if (paren_depth > 1 && i + 1 < statement.size() && statement[i + 1] == ')') {
                result.back() = '{';
                result.push_back('}');
                ++i; // skip ')'
                changed = true;
                continue;
            }
            continue;
        }

        if (ch == ')') {
            if (paren_depth > 0) {
                --paren_depth;
            }
            continue;
        }

        if (ch == '{') {
            ++brace_depth;
            continue;
        }

        if (ch == '}') {
            if (brace_depth > 0) {
                --brace_depth;
            }
            continue;
        }

        if (ch == '[') {
            ++bracket_depth;
            continue;
        }

        if (ch == ']') {
            if (bracket_depth > 0) {
                --bracket_depth;
            }
            continue;
        }
    }

    if (changed) {
        return result;
    }
    return statement;
}

bool is_literal_template_candidate(std::string_view text) {
    auto trimmed = trim_copy(text);
    if (trimmed.empty()) {
        return false;
    }

    if (trimmed.front() == '\'' && trimmed.back() == '\'' && trimmed.size() >= 3) {
        return true;
    }

    size_t idx = 0;
    bool has_digits = false;

    auto consume_digit_like = [&](char c) {
        if (c == '\'') {
            return true;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            has_digits = true;
            return true;
        }
        return false;
    };

    if (trimmed[idx] == '0' && idx + 1 < trimmed.size() && (trimmed[idx + 1] == 'x' || trimmed[idx + 1] == 'X')) {
        idx += 2;
        for (; idx < trimmed.size(); ++idx) {
            char c = trimmed[idx];
            if (c == '\'') {
                continue;
            }
            if (std::isxdigit(static_cast<unsigned char>(c))) {
                has_digits = true;
                continue;
            }
            break;
        }
    } else if (trimmed[idx] == '0' && idx + 1 < trimmed.size() && (trimmed[idx + 1] == 'b' || trimmed[idx + 1] == 'B')) {
        idx += 2;
        for (; idx < trimmed.size(); ++idx) {
            char c = trimmed[idx];
            if (c == '\'') {
                continue;
            }
            if (c == '0' || c == '1') {
                has_digits = true;
                continue;
            }
            break;
        }
    } else {
        for (; idx < trimmed.size(); ++idx) {
            char c = trimmed[idx];
            if (!consume_digit_like(c)) {
                break;
            }
        }
    }

    if (!has_digits) {
        return false;
    }

    static constexpr std::string_view allowed_suffix_chars = "uUlLzZ";
    for (; idx < trimmed.size(); ++idx) {
        char c = trimmed[idx];
        if (std::isspace(static_cast<unsigned char>(c))) {
            continue;
        }
        if (allowed_suffix_chars.find(c) == std::string_view::npos) {
            return false;
        }
    }

    return true;
}

size_t find_matching_delimiter(const std::string& text, size_t open_pos, char open_ch, char close_ch) {
    bool in_string = false;
    bool in_char = false;
    bool escape = false;
    int depth = 0;

    for (size_t i = open_pos; i < text.size(); ++i) {
        char ch = text[i];

        if (escape) {
            escape = false;
            continue;
        }

        if (in_string) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (in_char) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '\'') {
                in_char = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
            continue;
        }

        if (ch == '\'') {
            in_char = true;
            continue;
        }

        if (ch == open_ch) {
            ++depth;
            continue;
        }

        if (ch == close_ch) {
            if (--depth == 0) {
                return i;
            }
        }
    }

    return std::string::npos;
}

std::vector<std::string> split_top_level(std::string_view text) {
    std::vector<std::string> parts;
    size_t last = 0;
    int angle = 0;
    int paren = 0;
    int brace = 0;
    int bracket = 0;
    bool in_string = false;
    bool in_char = false;
    bool escape = false;

    for (size_t i = 0; i < text.size(); ++i) {
        char ch = text[i];

        if (escape) {
            escape = false;
            continue;
        }

        if (in_string) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (in_char) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '\'') {
                in_char = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
            continue;
        }

        if (ch == '\'') {
            in_char = true;
            continue;
        }

        switch (ch) {
            case '<': ++angle; break;
            case '>': if (angle > 0) --angle; break;
            case '(': ++paren; break;
            case ')': if (paren > 0) --paren; break;
            case '{': ++brace; break;
            case '}': if (brace > 0) --brace; break;
            case '[': ++bracket; break;
            case ']': if (bracket > 0) --bracket; break;
            case ',':
                if (angle == 0 && paren == 0 && brace == 0 && bracket == 0) {
                    parts.push_back(trim_copy(text.substr(last, i - last)));
                    last = i + 1;
                }
                break;
            default:
                break;
        }
    }

    if (last <= text.size()) {
        parts.push_back(trim_copy(text.substr(last)));
    }
    return parts;
}

std::string rewrite_contract_statement(const std::string& statement) {
    std::string trimmed = trim_copy(statement);
    if (!trimmed.starts_with("pre<") && !trimmed.starts_with("post<")) {
        return {};
    }

    bool is_post = trimmed.starts_with("post<");
    size_t template_start = trimmed.find('<');
    size_t template_end = find_matching_delimiter(trimmed, template_start, '<', '>');
    if (template_start == std::string::npos || template_end == std::string::npos) {
        return {};
    }

    size_t args_start = trimmed.find('(', template_end + 1);
    size_t args_end = find_matching_delimiter(trimmed, args_start, '(', ')');
    if (args_start == std::string::npos || args_end == std::string::npos) {
        return {};
    }

    std::string contract_targets = trimmed.substr(template_start + 1, template_end - template_start - 1);
    std::vector<std::string> targets = split_top_level(contract_targets);
    if (targets.empty()) {
        return {};
    }

    std::string args_text = trimmed.substr(args_start + 1, args_end - args_start - 1);
    std::vector<std::string> args = split_top_level(args_text);
    if (args.empty()) {
        return {};
    }

    std::string condition = trim_copy(args[0]);
    std::string message = (args.size() > 1) ? trim_copy(args[1]) : std::string{""""};
    if (message.empty()) {
        message = "\"\"";
    }

    std::string indent;
    size_t first_non_ws = statement.find_first_not_of(" \t");
    if (first_non_ws != std::string::npos) {
        indent = statement.substr(0, first_non_ws);
    }

    std::string guard_chain;
    if (targets.size() > 1) {
        for (size_t i = 1; i < targets.size(); ++i) {
            if (!targets[i].empty()) {
                guard_chain += trim_copy(targets[i]);
                guard_chain += " && ";
            }
        }
    }

    std::string system = trim_copy(targets[0]);
    if (system.empty()) {
        return {};
    }

    std::string result;
    result.reserve(statement.size() * 2);

    result += indent;
    result += "if (";
    if (!guard_chain.empty()) {
        result += guard_chain;
    }
    result += "cpp2::" + system + ".is_active() && !(" + condition + ") ) { ";
    result += "cpp2::" + system + ".report_violation(CPP2_CONTRACT_MSG(" + message + "));";
    result += " }";

    if (is_post) {
        result += "\n";
    }

    return result;
}

std::string rewrite_as_expressions(std::string statement) {
    bool changed = false;
    size_t search_pos = 0;

    while (true) {
        size_t pos = find_keyword_outside(statement, "as", search_pos);
        if (pos == std::string::npos) {
            break;
        }

        if (pos == 0 || pos + 2 >= statement.size()) {
            search_pos = pos + 2;
            continue;
        }

        if (!std::isspace(static_cast<unsigned char>(statement[pos - 1])) && statement[pos - 1] != '(') {
            search_pos = pos + 2;
            continue;
        }

        size_t rhs_start = pos + 2;
        while (rhs_start < statement.size() && std::isspace(static_cast<unsigned char>(statement[rhs_start]))) {
            ++rhs_start;
        }
        if (rhs_start >= statement.size()) {
            break;
        }

        size_t lhs_end = pos;
        while (lhs_end > 0 && std::isspace(static_cast<unsigned char>(statement[lhs_end - 1]))) {
            --lhs_end;
        }
        if (lhs_end == 0) {
            search_pos = pos + 2;
            continue;
        }

        size_t lhs_start = 0;
        int paren = 0;
        int brace = 0;
        int bracket = 0;
        for (size_t i = lhs_end; i > 0; --i) {
            char ch = statement[i - 1];
            switch (ch) {
                case ')': ++paren; break;
                case '(': if (paren > 0) { --paren; } else { lhs_start = i - 1; goto lhs_done; }
                          break;
                case '}': ++brace; break;
                case '{': if (brace > 0) { --brace; } else { lhs_start = i - 1; goto lhs_done; }
                          break;
                case ']': ++bracket; break;
                case '[': if (bracket > 0) { --bracket; } else { lhs_start = i - 1; goto lhs_done; }
                          break;
                default:
                    if (paren == 0 && brace == 0 && bracket == 0) {
                        if (ch == ',' || ch == ';' || ch == '=' || ch == '+' || ch == '-' || ch == '*' || ch == '/' ||
                            ch == '%' || ch == '|' || ch == '&' || ch == '^' || ch == '?' || ch == ':' || ch == '<' || ch == '>') {
                            lhs_start = i;
                            goto lhs_done;
                        }
                    }
                    break;
            }
        }
        lhs_start = 0;
lhs_done:
        while (lhs_start < lhs_end && std::isspace(static_cast<unsigned char>(statement[lhs_start]))) {
            ++lhs_start;
        }
        if (lhs_start >= lhs_end) {
            search_pos = pos + 2;
            continue;
        }

        size_t rhs_end = rhs_start;
        int angle = 0;
        int paren_rhs = 0;
        int brace_rhs = 0;
        for (size_t i = rhs_start; i < statement.size(); ++i) {
            char ch = statement[i];
            if (ch == '<') {
                ++angle;
            } else if (ch == '>') {
                if (angle > 0) {
                    --angle;
                }
            } else if (ch == '(') {
                ++paren_rhs;
            } else if (ch == ')') {
                if (paren_rhs > 0) {
                    --paren_rhs;
                } else {
                    rhs_end = i;
                    break;
                }
            }

            if (angle == 0 && paren_rhs == 0) {
                if (ch == ',' || ch == ';' || ch == ')' || ch == ']' || ch == '}') {
                    rhs_end = i;
                    break;
                }
                if (ch == '=' || ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '%' ||
                    ch == '|' || ch == '&' || ch == '^' || ch == '?' || ch == ':') {
                    rhs_end = i;
                    break;
                }
            }
            rhs_end = i + 1;
        }

        std::string lhs = trim_copy(statement.substr(lhs_start, lhs_end - lhs_start));
        std::string rhs = trim_copy(statement.substr(rhs_start, rhs_end - rhs_start));
        if (lhs.empty() || rhs.empty()) {
            search_pos = pos + 2;
            continue;
        }

        std::string replacement;
        if (is_literal_template_candidate(lhs)) {
            replacement = "cpp2::impl::as_<" + rhs + ", " + lhs + ">()";
        } else {
            replacement = "cpp2::impl::as_<" + rhs + ">(" + lhs + ")";
        }

        statement.replace(lhs_start, rhs_end - lhs_start, replacement);
        search_pos = lhs_start + replacement.size();
        changed = true;
    }

    return statement;
}

std::optional<std::string> rewrite_known_ufcs_call(std::string_view expr_view) {
    std::string trimmed = trim_copy(expr_view);
    size_t dot_pos = trimmed.find('.');
    if (dot_pos == std::string::npos) {
        return std::nullopt;
    }

    size_t open_paren = trimmed.find('(', dot_pos + 1);
    if (open_paren == std::string::npos) {
        return std::nullopt;
    }

    std::string lhs = trim_copy(trimmed.substr(0, dot_pos));
    std::string function_name = trim_copy(trimmed.substr(dot_pos + 1, open_paren - dot_pos - 1));
    if (lhs.empty() || function_name.empty()) {
        return std::nullopt;
    }

    static const std::unordered_set<std::string> cstdio_functions = {
        "fprintf", "fclose", "fscanf", "fgets", "fputs", "fread", "fwrite", "fflush"
    };

    if (cstdio_functions.find(function_name) == cstdio_functions.end()) {
        return std::nullopt;
    }

    int depth = 0;
    size_t close_paren = std::string::npos;
    for (size_t idx = open_paren; idx < trimmed.size(); ++idx) {
        char ch = trimmed[idx];
        if (ch == '(') {
            ++depth;
        } else if (ch == ')') {
            --depth;
            if (depth == 0) {
                close_paren = idx;
                break;
            }
        }
    }
    if (close_paren == std::string::npos) {
        return std::nullopt;
    }

    std::string args = trim_copy(trimmed.substr(open_paren + 1, close_paren - open_paren - 1));
    std::string rewritten = "std::" + function_name + "(" + lhs;
    if (!args.empty()) {
        rewritten += ", " + args;
    }
    rewritten += ")";

    if (close_paren + 1 < trimmed.size()) {
        rewritten += trimmed.substr(close_paren + 1);
    }

    return rewritten;
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

std::string ensure_parenthesized_conditions(const std::string& text) {
    std::string result;
    result.reserve(text.size() + text.size() / 4);

    size_t pos = 0;
    while (pos < text.size()) {
        size_t if_pos = text.find("if", pos);
        if (if_pos == std::string::npos) {
            result.append(text.substr(pos));
            break;
        }

        auto is_identifier_char = [](char ch) {
            return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
        };

        if ((if_pos > 0 && is_identifier_char(text[if_pos - 1])) ||
            (if_pos + 2 < text.size() && is_identifier_char(text[if_pos + 2]))) {
            result.append(text.substr(pos, if_pos - pos + 2));
            pos = if_pos + 2;
            continue;
        }

        size_t cursor = if_pos + 2;
        std::string spaces;
        while (cursor < text.size() && (text[cursor] == ' ' || text[cursor] == '\t')) {
            spaces.push_back(text[cursor]);
            ++cursor;
        }

        if (cursor >= text.size()) {
            result.append(text.substr(pos));
            break;
        }

        if (text[cursor] == '(') {
            result.append(text.substr(pos, cursor - pos));
            pos = cursor;
            continue;
        }

        size_t expr_start = cursor;
        size_t expr_end = expr_start;
        int paren_depth = 0;
        bool saw_expression_char = false;
        for (; expr_end < text.size(); ++expr_end) {
            char ch = text[expr_end];
            if (ch == '(') {
                ++paren_depth;
            } else if (ch == ')') {
                if (paren_depth == 0) {
                    break;
                }
                --paren_depth;
            } else if ((ch == '{' || ch == '\n' || ch == '\r' || ch == ';') && paren_depth == 0) {
                break;
            }
            saw_expression_char = true;
        }

        if (!saw_expression_char) {
            result.append(text.substr(pos));
            break;
        }

        size_t trimmed_end = expr_end;
        while (trimmed_end > expr_start && (text[trimmed_end - 1] == ' ' || text[trimmed_end - 1] == '\t')) {
            --trimmed_end;
        }

        std::string expr = text.substr(expr_start, trimmed_end - expr_start);

        result.append(text.substr(pos, if_pos - pos));
        result += "if";
        if (spaces.empty()) {
            result.push_back(' ');
        } else {
            result += spaces;
        }
        result.push_back('(');
        result += expr;
        result.push_back(')');
        result += text.substr(trimmed_end, expr_end - trimmed_end);

        pos = expr_end;
    }

    return result;
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

int brace_depth_before(std::string_view source, size_t pos) {
    int depth = 0;
    for (size_t i = 0; i < pos; ++i) {
        char ch = source[i];
        if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            if (depth > 0) {
                --depth;
            }
        }
    }
    return depth;
}

size_t find_statement_end(std::string_view text, size_t start_pos) {
    if (start_pos >= text.size()) {
        return text.size();
    }

    int brace_depth = 0;
    int paren_depth = 0;
    bool in_string = false;
    bool in_char = false;
    bool escape = false;
    int contract_level = 0;
    int contract_brace_depth = 0;

    for (size_t pos = start_pos; pos < text.size(); ++pos) {
        char ch = text[pos];

        if (escape) {
            escape = false;
            continue;
        }

        if (in_string) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (in_char) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '\'') {
                in_char = false;
            }
            continue;
        }

        // Check for raw string literal: R"delimiter(content)delimiter"
        if (ch == 'R' && pos + 1 < text.size() && text[pos + 1] == '"') {
            // Extract delimiter
            size_t delim_start = pos + 2;
            size_t delim_end = text.find('(', delim_start);
            if (delim_end != std::string::npos) {
                std::string delimiter(text.substr(delim_start, delim_end - delim_start));
                std::string end_marker = ")" + delimiter + "\"";
                size_t content_start = delim_end + 1;
                size_t raw_end = text.find(end_marker, content_start);
                if (raw_end != std::string::npos) {
                    pos = raw_end + end_marker.length() - 1;
                    continue;
                }
            }
        }

        // Check for prefix raw string literals: uR"...", u8R"...", UR"...", LR"..."
        if (pos + 2 < text.size() && text[pos + 1] == 'R' && text[pos + 2] == '"') {
            char prefix = ch;
            if (prefix == 'u' || prefix == 'U' || prefix == 'L') {
                size_t delim_start = pos + 3;
                size_t delim_end = text.find('(', delim_start);
                if (delim_end != std::string::npos) {
                    std::string delimiter(text.substr(delim_start, delim_end - delim_start));
                    std::string end_marker = ")" + delimiter + "\"";
                    size_t content_start = delim_end + 1;
                    size_t raw_end = text.find(end_marker, content_start);
                    if (raw_end != std::string::npos) {
                        pos = raw_end + end_marker.length() - 1;
                        continue;
                    }
                }
            }
        }

        // Check for u8R"..." prefix
        if (pos + 3 < text.size() && ch == 'u' && text[pos + 1] == '8' && text[pos + 2] == 'R' && text[pos + 3] == '"') {
            size_t delim_start = pos + 4;
            size_t delim_end = text.find('(', delim_start);
            if (delim_end != std::string::npos) {
                std::string delimiter(text.substr(delim_start, delim_end - delim_start));
                std::string end_marker = ")" + delimiter + "\"";
                size_t content_start = delim_end + 1;
                size_t raw_end = text.find(end_marker, content_start);
                if (raw_end != std::string::npos) {
                    pos = raw_end + end_marker.length() - 1;
                    continue;
                }
            }
        }

        if (ch == '"') {
            in_string = true;
            continue;
        }

        if (ch == '\'') {
            in_char = true;
            continue;
        }

        if (!in_string && !in_char && ch == '<') {
            bool is_contract = false;
            if (pos >= 3 && text.substr(pos - 3, 3) == "pre") {
                is_contract = true;
            } else if (pos >= 4 && text.substr(pos - 4, 4) == "post") {
                is_contract = true;
            }
            if (is_contract) {
                ++contract_level;
                contract_brace_depth = 0;
                continue;
            }
        }

        if (contract_level > 0) {
            if (ch == '{') {
                ++contract_brace_depth;
                continue;
            }
            if (ch == '}') {
                if (contract_brace_depth > 0) {
                    --contract_brace_depth;
                }
                continue;
            }
            if (ch == '>') {
                if (contract_brace_depth == 0) {
                    if (contract_level > 0) {
                        --contract_level;
                    }
                }
                continue;
            }
        }

        if (ch == '{') {
            ++brace_depth;
            continue;
        }

        if (ch == '}') {
            if (brace_depth > 0) {
                --brace_depth;
                if (brace_depth == 0) {
                    size_t end = pos + 1;
                    size_t probe = end;
                    while (probe < text.size() && std::isspace(static_cast<unsigned char>(text[probe]))) {
                        ++probe;
                    }
                    if (probe < text.size() && text[probe] == ';') {
                        end = probe + 1;
                    }
                    return end;
                }
                continue;
            }
            return pos + 1;
        }

        if (ch == '(') {
            ++paren_depth;
            continue;
        }

        if (ch == ')') {
            if (paren_depth > 0) {
                --paren_depth;
            }
            continue;
        }

        if (ch == ';' && brace_depth == 0 && paren_depth == 0) {
            return pos + 1;
        }
    }

    return text.size();
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

struct SignatureModifiers {
    bool has_return = false;
    std::string return_type;
    std::vector<std::pair<std::string, std::string>> contracts;
    std::string residual;
};

SignatureModifiers parse_signature_modifiers(std::string_view text) {
    SignatureModifiers modifiers;
    std::string working = trim_copy(text);
    if (working.empty()) {
        return modifiers;
    }

    if (working.rfind("->", 0) == 0) {
        std::string after_arrow = working.substr(2);
        trim_in_place(after_arrow);

        size_t contract_pos = std::string::npos;
        for (size_t i = 0; i < after_arrow.size(); ++i) {
            if (after_arrow.compare(i, 4, "pre<") == 0 || after_arrow.compare(i, 5, "post<") == 0) {
                contract_pos = i;
                break;
            }
        }

        modifiers.has_return = true;
        if (contract_pos == std::string::npos) {
            modifiers.return_type = after_arrow;
            trim_in_place(modifiers.return_type);
            working.clear();
        } else {
            modifiers.return_type = after_arrow.substr(0, contract_pos);
            trim_in_place(modifiers.return_type);
            working = after_arrow.substr(contract_pos);
        }
    }

    while (!working.empty()) {
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG parse_signature_modifiers: entering with working='" << working << "'\n");
        trim_in_place(working);
        if (!(working.rfind("pre<", 0) == 0 || working.rfind("post<", 0) == 0)) {
            modifiers.residual = working;
            break;
        }

        size_t kind_end = working.find('<');
        if (kind_end == std::string::npos) {
            modifiers.residual = working;
            break;
        }

        std::string kind = working.substr(0, kind_end);
        trim_in_place(kind);

        size_t content_start = kind_end + 1;
        size_t pos = content_start;
        int depth = 1;
        int brace_depth = 0;
        int paren_depth = 0;
        int bracket_depth = 0;
        bool in_string = false;
        bool in_char = false;
        bool escape = false;

        for (; pos < working.size(); ++pos) {
            char ch = working[pos];
            if (escape) {
                escape = false;
                continue;
            }
            if (ch == '\\') {
                escape = true;
                continue;
            }
            if (in_string) {
                if (ch == '"') {
                    in_string = false;
                }
                continue;
            }
            if (in_char) {
                if (ch == '\\') {
                    escape = true;
                    continue;
                }
                if (ch == '\'') {
                    in_char = false;
                }
                continue;
            }
            if (ch == '"') {
                in_string = true;
                continue;
            }
            if (ch == '\'') {
                in_char = true;
                continue;
            }
            if (ch == '{') {
                ++brace_depth;
                continue;
            }
            if (ch == '}') {
                if (brace_depth > 0) {
                    --brace_depth;
                }
                continue;
            }
            if (ch == '(') {
                ++paren_depth;
                continue;
            }
            if (ch == ')') {
                if (paren_depth > 0) {
                    --paren_depth;
                }
                continue;
            }
            if (ch == '[') {
                ++bracket_depth;
                continue;
            }
            if (ch == ']') {
                if (bracket_depth > 0) {
                    --bracket_depth;
                }
                continue;
            }
            if (ch == '<' && brace_depth == 0 && paren_depth == 0 && bracket_depth == 0) {
                ++depth;
                continue;
            }
            if (ch == '>' && brace_depth == 0 && paren_depth == 0 && bracket_depth == 0) {
                --depth;
                if (depth == 0) {
                    std::string content = working.substr(content_start, pos - content_start);
                    trim_in_place(content);
                    if (content.size() >= 2 && content.front() == '{' && content.back() == '}') {
                        content = content.substr(1, content.size() - 2);
                        trim_in_place(content);
                    }
                    modifiers.contracts.emplace_back(kind, content);
                    working = (pos + 1 < working.size()) ? working.substr(pos + 1) : std::string{};
                    break;
                }
                continue;
            }
        }

        if (depth != 0) {
            modifiers.residual = working;
            CPP2_EMITTER_DEBUG(std::cerr << "DEBUG parse_signature_modifiers: unmatched depth, working='" << working << "'\n");
            break;
        }
    }

    if (!working.empty()) {
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG parse_signature_modifiers: leftover working='" << working << "'\n");
    }
    return modifiers;
}

std::string build_contract_suffix(const std::vector<std::pair<std::string, std::string>>& contracts) {
    std::string suffix;
    for (const auto& [kind, expr] : contracts) {
        if (expr.empty()) {
            continue;
        }
        suffix += " [[";
        suffix += kind;
        suffix += ": ";
        suffix += expr;
        suffix += "]]";
    }
    return suffix;
}

bool transform_cpp2_parameter_direct(const std::string& original, std::string& output, bool prefer_value_for_in = false) {
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
        case ParameterKind::In: {
            if (prefer_value_for_in) {
                remove_const_qualifier(type_builder);
                remove_reference_suffix(type_builder);
            } else {
                if (!contains_const_qualifier(type_builder)) {
                    type_builder = "const " + type_builder;
                }
                ensure_reference(type_builder);
            }

            break;
        }
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

    size_t semicolon_pos = content.find(';', close_paren);
    size_t eq_pos = std::string::npos;
    {
        std::string_view after_paren = std::string_view{content}.substr(close_paren + 1);
        size_t eq_relative = find_top_level_equal(after_paren);
        if (eq_relative != std::string::npos) {
            eq_pos = close_paren + 1 + eq_relative;
        }
    }

    std::string name = trim_copy(content.substr(0, colon_pos));
    if (name.empty()) {
        return {};
    }

    std::string params_section = content.substr(open_paren + 1, close_paren - open_paren - 1);

    if (eq_pos == std::string::npos && semicolon_pos != std::string::npos) {
        std::string between = content.substr(close_paren + 1, semicolon_pos - close_paren - 1);
        SignatureModifiers modifiers = parse_signature_modifiers(between);
        if (!modifiers.residual.empty()) {
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_function_signature_line: forward decl residual '" << modifiers.residual
                                      << "' for line '" << content << "'\n");
            return {};
        }

        std::string cpp_return_type = modifiers.has_return ? transform_return_spec(modifiers.return_type) : std::string{"void"};
    std::string transformed_params = transform_parameter(params_section, patterns, /*prefer_value_for_in=*/true);
        std::string contract_suffix = build_contract_suffix(modifiers.contracts);

        std::string forward_decl = indent + cpp_return_type + " " + name + "(" + transformed_params + ")";
        if (!contract_suffix.empty()) {
            forward_decl += contract_suffix;
        }
        forward_decl += ";";

        std::string trailing = content.substr(semicolon_pos + 1);
        if (!trailing.empty()) {
            forward_decl += trailing;
        }

        return forward_decl;
    }

    if (eq_pos == std::string::npos) {
        return {};
    }

    std::string between = content.substr(close_paren + 1, eq_pos - close_paren - 1);
    std::string body_fragment = content.substr(eq_pos + 1);
    std::string body_trim = trim_copy(body_fragment);
    if (body_trim.empty()) {
        return {};
    }

    SignatureModifiers modifiers = parse_signature_modifiers(between);
    if (!modifiers.residual.empty()) {
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_function_signature_line: residual '" << modifiers.residual
                                      << "' for line '" << content << "'\n");
        return {};
    }

    bool is_expression_body = body_trim.front() != '{';
    std::string cpp_return_type;
    if (modifiers.has_return) {
        cpp_return_type = transform_return_spec(modifiers.return_type);
    } else if (is_expression_body) {
        cpp_return_type = "auto";
    } else {
        cpp_return_type = "void";
    }

    std::string transformed_params = transform_parameter(params_section, patterns, /*prefer_value_for_in=*/true);
    std::string contract_suffix = build_contract_suffix(modifiers.contracts);

    std::string result = indent;
    result += cpp_return_type + " " + name + "(" + transformed_params + ")";
    if (!contract_suffix.empty()) {
        result += contract_suffix;
    }

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
            CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_function_signature_line: empty expression body for line '"
                                          << content << "'\n");
            return {};
        }
        result += " { return " + expr + "; }";
    } else {
        std::string body_text = body_trim;
        size_t open_brace = body_text.find('{');
        size_t close_brace = body_text.rfind('}');
        if (open_brace != std::string::npos && close_brace != std::string::npos && close_brace > open_brace) {
            std::string inner = body_text.substr(open_brace + 1, close_brace - open_brace - 1);
            bool inner_whitespace_only = !inner.empty();
            if (inner_whitespace_only) {
                for (char ch : inner) {
                    if (!std::isspace(static_cast<unsigned char>(ch))) {
                        inner_whitespace_only = false;
                        break;
                    }
                }
            }

            std::string transformed_inner = apply_recursive_transformations(inner, patterns, 1);
            trim_in_place(transformed_inner);
            if (transformed_inner.empty()) {
                if (inner_whitespace_only) {
                    result += " { }";
                } else {
                    result += " {}";
                }
            } else {
                result += " { " + transformed_inner + " }";
            }
        } else {
            result += " " + body_trim;
        }
    }

    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_function_signature_line: produced '" << result << "'\n");
    return result;
}

std::string transform_variable_declaration(const std::string& line) {
    std::string trimmed = trim_copy(line);
    if (trimmed.empty()) {
        return line;
    }
    
    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_variable_declaration: input='" << trimmed << "'\n");
    
    // Check for := pattern (auto variable)
    size_t assign_pos = trimmed.find(":=");
    if (assign_pos != std::string::npos) {
        std::string name = trim_copy(trimmed.substr(0, assign_pos));
        std::string value = trim_copy(trimmed.substr(assign_pos + 2));
        if (!value.empty() && value.back() == ';') {
            value.pop_back();
            std::string result = "auto " + name + " = " + trim_copy(value) + ";";
            CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_variable_declaration: := pattern -> '" << result << "'\n");
            return result;
        }
        std::string result = "auto " + name + " = " + value;
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_variable_declaration: := pattern -> '" << result << "'\n");
        return result;
    }
    
    // Check for : type = value pattern
    size_t colon_pos = find_top_level_colon(trimmed);
    if (colon_pos == std::string::npos) {
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_variable_declaration: no colon found\n");
        return line;
    }
    
    size_t equal_pos = find_top_level_equal(std::string_view{trimmed}.substr(colon_pos));
    if (equal_pos == std::string::npos) {
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_variable_declaration: no equal after colon\n");
        return line;
    }
    equal_pos += colon_pos; // adjust to absolute position
    
    std::string name = trim_copy(trimmed.substr(0, colon_pos));
    std::string type = trim_copy(trimmed.substr(colon_pos + 1, equal_pos - colon_pos - 1));
    std::string value = trim_copy(trimmed.substr(equal_pos + 1));
    
    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_variable_declaration: name='" << name << "' type='" << type << "' value='" << value << "'\n");

    if (name.empty() || type.empty()) {
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_variable_declaration: name or type empty\n");
        return line;
    }
    
    if (type == "type") {
        std::string alias_target = value;
        bool had_semicolon = false;
        if (!alias_target.empty() && alias_target.back() == ';') {
            alias_target.pop_back();
            had_semicolon = true;
        }
        alias_target = trim_copy(alias_target);
        if (alias_target.empty()) {
            CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_variable_declaration: alias target empty\n");
            return line;
        }

        std::string result = "using " + name + " = " + alias_target + ";";
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_variable_declaration: type alias -> '" << result << "'\n");
        return result;
    }

    // Check if this is actually a function or type declaration
    if (type.find("type") != std::string::npos || trimmed.find("->") != std::string::npos || trimmed.find(" = {") != std::string::npos) {
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_variable_declaration: looks like function/type, skipping\n");
        return line;
    }
    
    type = map_cpp2_type(type);
    std::string result;
    if (!value.empty() && value.back() == ';') {
        value.pop_back();
        result = type + " " + name + " = " + trim_copy(value) + ";";
    } else {
        result = type + " " + name + " = " + value;
    }
    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_variable_declaration: result='" << result << "'\n");
    return result;
}

std::string rewrite_free_text_segment(std::string_view text, const std::vector<PatternData>& patterns) {
    if (text.empty()) {
        return {};
    }

    std::string result;
    result.reserve(text.size());
    size_t line_start = 0;
    bool changed = false;

    for (size_t i = 0; i <= text.size(); ++i) {
        bool at_end = (i == text.size());
        bool is_newline = (!at_end && text[i] == '\n');

        if (at_end || is_newline) {
            std::string line{text.substr(line_start, i - line_start)};
            
            // Try function transformation first
            std::string transformed = transform_function_signature_line(line, patterns);
            
            // If function transform didn't work, try variable transform
            if (transformed.empty() || transformed == line) {
                std::string var_transformed = transform_variable_declaration(line);
                if (!var_transformed.empty() && var_transformed != line) {
                    transformed = var_transformed;
                    changed = true;
                }
            }

            if (!transformed.empty()) {
                if (transformed != line) {
                    changed = true;
                }
                result += transformed;
            } else {
                result += line;
            }

            if (is_newline) {
                result.push_back('\n');
            }

            line_start = i + 1;
        }
    }

    if (!changed) {
        return std::string{text};
    }

    return result;
}

bool transform_template_alias_core(std::string_view core, std::string& output) {
    size_t semicolon_pos = core.find(';');
    if (semicolon_pos == std::string::npos) {
        return false;
    }

    std::string_view statement = core.substr(0, semicolon_pos);
    size_t pos = 0;
    while (pos < statement.size() && (statement[pos] == ' ' || statement[pos] == '\t')) {
        ++pos;
    }

    std::string indent{statement.substr(0, pos)};

    if (pos + 4 > statement.size() || statement.compare(pos, 4, "type") != 0) {
        return false;
    }

    pos += 4;
    while (pos < statement.size() && std::isspace(static_cast<unsigned char>(statement[pos]))) {
        ++pos;
    }

    if (pos >= statement.size()) {
        return false;
    }

    std::string_view remainder = statement.substr(pos);
    size_t equal_rel = std::string::npos;
    int paren_depth = 0;
    int bracket_depth = 0;
    int brace_depth = 0;
    int angle_depth = 0;
    for (size_t i = 0; i < remainder.size(); ++i) {
        char ch = remainder[i];
        switch (ch) {
            case '(': ++paren_depth; break;
            case ')': if (paren_depth > 0) --paren_depth; break;
            case '[': ++bracket_depth; break;
            case ']': if (bracket_depth > 0) --bracket_depth; break;
            case '{': ++brace_depth; break;
            case '}': if (brace_depth > 0) --brace_depth; break;
            case '<': ++angle_depth; break;
            case '>': if (angle_depth > 0) --angle_depth; break;
            case '=':
                if (paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 && angle_depth == 0) {
                    equal_rel = i;
                    i = remainder.size();
                }
                break;
            default:
                break;
        }
    }

    if (equal_rel == std::string::npos) {
        return false;
    }

    std::string alias = trim_copy(remainder.substr(0, equal_rel));
    std::string value = trim_copy(remainder.substr(equal_rel + 1));
    if (alias.empty() || value.empty()) {
        return false;
    }

    auto [alias_name, template_params] = extract_template_params(alias);
    std::string result = indent;

    if (!template_params.empty()) {
        result += generate_template_prefix(template_params);
    }

    result += "using " + alias_name + " = " + value + ";";

    output = std::move(result);
    return true;
}

bool try_template_alias_rewrite(size_t start_pos, std::string_view source, const std::vector<PatternData>& patterns, size_t& consumed_end, std::string& rewritten) {
    (void)patterns;
    if (start_pos >= source.size()) {
        return false;
    }

    size_t pos = start_pos;
    while (pos < source.size() && (source[pos] == ' ' || source[pos] == '\t')) {
        ++pos;
    }

    if (pos >= source.size()) {
        return false;
    }

    if (source.compare(pos, 4, "type") != 0) {
        return false;
    }

    size_t semicolon_pos = source.find(';', pos);
    if (semicolon_pos == std::string::npos) {
        return false;
    }

    size_t line_end = semicolon_pos + 1;
    while (line_end < source.size() && source[line_end] != '\n' && source[line_end] != '\r') {
        ++line_end;
    }

    size_t newline_end = line_end;
    if (newline_end < source.size() && source[newline_end] == '\r') {
        ++newline_end;
    }
    if (newline_end < source.size() && source[newline_end] == '\n') {
        ++newline_end;
    }

    std::string core = std::string(source.substr(start_pos, semicolon_pos + 1 - start_pos));
    std::string trailing_comment;
    if (line_end > semicolon_pos + 1) {
        trailing_comment = std::string(source.substr(semicolon_pos + 1, line_end - (semicolon_pos + 1)));
    }

    std::string transformed_core;
    if (!transform_template_alias_core(core, transformed_core)) {
        return false;
    }

    rewritten = transformed_core;
    if (!trailing_comment.empty()) {
        rewritten += trailing_comment;
    }
    if (newline_end > line_end) {
        rewritten.append(source.substr(line_end, newline_end - line_end));
    }

    consumed_end = newline_end;
    return true;
}

bool try_function_signature_rewrite(size_t start_pos, std::string_view source, const std::vector<PatternData>& patterns, size_t& consumed_end, std::string& rewritten) {
    if (start_pos >= source.size()) {
        return false;
    }

    size_t statement_end = find_statement_end(source, start_pos);
    if (statement_end <= start_pos) {
        return false;
    }

    std::string_view statement_view = source.substr(start_pos, statement_end - start_pos);
    std::string transformed = transform_function_signature_line(std::string(statement_view), patterns);
    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG try_function_signature_rewrite: statement='" << statement_view
                                  << "' transformed='" << transformed << "'\n");
    if (transformed.empty() || transformed == statement_view) {
        return false;
    }

    consumed_end = statement_end;
    rewritten = std::move(transformed);

    size_t trailing_start = consumed_end;
    size_t trailing = trailing_start;
    while (trailing < source.size() && (source[trailing] == ' ' || source[trailing] == '\t')) {
        ++trailing;
    }
    if (trailing > trailing_start) {
        rewritten.append(source.substr(trailing_start, trailing - trailing_start));
    }

    size_t newline_end = trailing;
    if (newline_end < source.size() && source[newline_end] == '\r') {
        ++newline_end;
    }
    if (newline_end < source.size() && source[newline_end] == '\n') {
        ++newline_end;
    }
    if (newline_end > trailing) {
        rewritten.append(source.substr(trailing, newline_end - trailing));
    }

    consumed_end = newline_end;
    return true;
}

std::string map_cpp2_type(std::string type) {
    type = trim_copy(type);

    if (type.empty()) {
        return type;
    }

    if (type == "i8") return "int8_t";
    if (type == "i16") return "int16_t";
    if (type == "i32") return "int32_t";
    if (type == "i64") return "int64_t";
    if (type == "u8") return "uint8_t";
    if (type == "u16") return "uint16_t";
    if (type == "u32") return "uint32_t";
    if (type == "u64") return "uint64_t";
    if (type == "bool") return "bool";
    if (type == "char") return "char";
    if (type == "string" || type == "std::string") return "std::string";

    return type;
}

std::string to_cpp_template_params(std::string_view cpp2_params) {
    if (cpp2_params.size() < 2 || cpp2_params.front() != '<' || cpp2_params.back() != '>') {
        return std::string{};
    }

    std::string result = "template<";
    bool first = true;
    size_t pos = 1; // skip leading '<'
    const size_t end = cpp2_params.size() - 1; // exclude closing '>'

    while (pos < end) {
        size_t comma = cpp2_params.find(',', pos);
        if (comma == std::string::npos || comma > end) {
            comma = end;
        }

        std::string param = trim_copy(cpp2_params.substr(pos, comma - pos));
        if (!param.empty()) {
            std::string cpp_param;
            size_t colon = param.find(':');
            if (colon != std::string::npos) {
                std::string name = trim_copy(param.substr(0, colon));
                std::string constraint = trim_copy(param.substr(colon + 1));

                if (!name.empty() && constraint == "type") {
                    cpp_param = "typename " + name;
                } else if (!name.empty() && constraint == "value") {
                    cpp_param = "auto " + name;
                } else {
                    cpp_param = param;
                }
            } else {
                cpp_param = "typename " + param;
            }

            if (!cpp_param.empty()) {
                if (!first) {
                    result += ", ";
                }
                first = false;
                result += cpp_param;
            }
        }

        if (comma >= end) {
            break;
        }
        pos = comma + 1;
    }

    result += ">\n";
    return result;
}

std::string indent_block(const std::string& text, std::string_view indent) {
    if (text.empty()) {
        return {};
    }

    std::string result;
    result.reserve(text.size() + indent.size() * 4); // rough guess

    bool at_line_start = true;
    for (char ch : text) {
        if (at_line_start && ch != '\n') {
            result.append(indent);
        }

        result.push_back(ch);
        at_line_start = (ch == '\n');
    }

    if (!result.empty() && result.back() != '\n') {
        result.push_back('\n');
    }

    return result;
}

std::vector<std::string> split_top_level_parameters(std::string_view params_section) {
    std::vector<std::string> params;
    if (trim_copy(params_section).empty()) {
        return params;
    }

    size_t start = 0;
    int paren_depth = 0;
    int bracket_depth = 0;
    int brace_depth = 0;
    int angle_depth = 0;
    bool in_string = false;
    bool in_char = false;
    bool escape = false;

    for (size_t i = 0; i < params_section.size(); ++i) {
        char ch = params_section[i];

        if (escape) {
            escape = false;
            continue;
        }

        if (in_string) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (in_char) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '\'') {
                in_char = false;
            }
            continue;
        }

        switch (ch) {
            case '"':
                in_string = true;
                break;
            case '\'':
                in_char = true;
                break;
            case '(': ++paren_depth; break;
            case ')': if (paren_depth > 0) --paren_depth; break;
            case '[': ++bracket_depth; break;
            case ']': if (bracket_depth > 0) --bracket_depth; break;
            case '{': ++brace_depth; break;
            case '}': if (brace_depth > 0) --brace_depth; break;
            case '<': ++angle_depth; break;
            case '>': if (angle_depth > 0) --angle_depth; break;
            case ',':
                if (paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 && angle_depth == 0) {
                    std::string param = std::string(params_section.substr(start, i - start));
                    params.push_back(trim_copy(param));
                    start = i + 1;
                }
                break;
            default:
                break;
        }
    }

    if (start < params_section.size()) {
        std::string param = std::string(params_section.substr(start));
        params.push_back(trim_copy(param));
    }

    return params;
}

std::string join_comma_separated(const std::vector<std::string>& values) {
    std::string joined;
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i].empty()) {
            continue;
        }
        if (!joined.empty()) {
            joined += ", ";
        }
        joined += values[i];
    }
    return joined;
}

struct ThisParameterInfo {
    bool found = false;
    bool is_const = true;
    std::string ref_qualifier;
};

ThisParameterInfo analyze_this_parameter(std::string_view param_text) {
    ThisParameterInfo info;

    std::string descriptor = trim_copy(param_text);
    if (descriptor.empty()) {
        return info;
    }

    size_t colon_pos = find_top_level_colon(descriptor);
    if (colon_pos != std::string::npos) {
        descriptor = trim_copy(descriptor.substr(0, colon_pos));
    }

    auto tokens = split_whitespace_tokens(descriptor);
    if (tokens.empty()) {
        return info;
    }

    std::optional<size_t> this_index;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == "this") {
            this_index = i;
            break;
        }
    }

    if (!this_index) {
        return info;
    }

    info.found = true;
    info.is_const = true;

    for (size_t i = 0; i < *this_index; ++i) {
        const std::string& token = tokens[i];
        if (token == "inout" || token == "out" || token == "move" || token == "forward" || token == "forward_ref" || token == "copy") {
            info.is_const = false;
            if (token == "move" || token == "forward" || token == "forward_ref") {
                info.ref_qualifier = "&&";
            }
        } else if (token == "in") {
            info.is_const = true;
        } else if (token == "const") {
            info.is_const = true;
        } else if (token == "&") {
            info.ref_qualifier = "&";
        } else if (token == "&&") {
            info.ref_qualifier = "&&";
        }
    }

    if (info.ref_qualifier.empty() && !info.is_const) {
        // For mutating methods default to lvalue qualifier
        info.ref_qualifier.clear();
    }

    return info;
}

std::string transform_union_method(const std::string& method_source, const std::vector<PatternData>& patterns) {
    std::string content = trim_copy(method_source);
    if (content.empty()) {
        return {};
    }

    size_t colon_pos = find_top_level_colon(content);
    if (colon_pos == std::string::npos) {
        return transform_function_signature_line(content, patterns);
    }

    std::string name = trim_copy(content.substr(0, colon_pos));
    if (name.empty()) {
        return {};
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
            ++depth;
        } else if (ch == ')') {
            if (depth == 0) {
                return {};
            }
            --depth;
            if (depth == 0) {
                close_paren = i;
                break;
            }
        }
    }
    if (close_paren == std::string::npos) {
        return {};
    }

    std::string params_section = content.substr(open_paren + 1, close_paren - open_paren - 1);
    auto params = split_top_level_parameters(params_section);

    ThisParameterInfo this_info;
    if (!params.empty()) {
        this_info = analyze_this_parameter(params.front());
        if (this_info.found) {
            params.erase(params.begin());
        }
    }

    if (!this_info.found) {
        return transform_function_signature_line(content, patterns);
    }

    std::string remaining_params = join_comma_separated(params);

    std::string after_paren = content.substr(close_paren + 1);
    trim_in_place(after_paren);
    if (after_paren.empty()) {
        return {};
    }

    size_t eq_pos_rel = find_top_level_equal(after_paren);
    if (eq_pos_rel == std::string::npos) {
        return {};
    }

    std::string signature_suffix = trim_copy(after_paren.substr(0, eq_pos_rel));
    std::string body_fragment = trim_copy(after_paren.substr(eq_pos_rel + 1));
    if (body_fragment.empty()) {
        return {};
    }

    bool expression_body = body_fragment.front() != '{';

    SignatureModifiers modifiers = parse_signature_modifiers(signature_suffix);
    if (!modifiers.residual.empty()) {
        return {};
    }

    std::string return_type;
    if (modifiers.has_return) {
        return_type = transform_return_spec(modifiers.return_type);
    } else if (expression_body) {
        return_type = "auto";
    } else {
        return_type = "void";
    }

    std::string transformed_params;
    if (!remaining_params.empty()) {
        transformed_params = transform_parameter(remaining_params, patterns, /*prefer_value_for_in=*/true);
    }

    std::string contract_suffix = build_contract_suffix(modifiers.contracts);

    std::string signature = return_type + " " + name + "(";
    signature += transformed_params;
    signature += ")";
    if (!contract_suffix.empty()) {
        signature += contract_suffix;
    }
    if (this_info.is_const) {
        signature += " const";
    }
    if (!this_info.ref_qualifier.empty()) {
        signature += " ";
        signature += this_info.ref_qualifier;
    }

    std::string result = signature;

    if (expression_body) {
        std::string expr = body_fragment;
        if (!expr.empty() && expr.front() == '=') {
            expr.erase(expr.begin());
            trim_in_place(expr);
        }
        if (!expr.empty() && expr.back() == ';') {
            expr.pop_back();
        }
        trim_in_place(expr);
        std::string transformed_expr = apply_recursive_transformations(expr, patterns, 1);
        trim_in_place(transformed_expr);
        if (transformed_expr.empty()) {
            return {};
        }
        transformed_expr = ensure_parenthesized_conditions(transformed_expr);
        result += " { return " + transformed_expr + "; }";
        return result;
    }

    // Block body
    if (body_fragment.front() == '{' && body_fragment.back() == '}') {
        std::string inner = body_fragment.substr(1, body_fragment.size() - 2);
        std::string transformed_inner = apply_recursive_transformations(inner, patterns, 1);
        trim_in_place(transformed_inner);
        transformed_inner = ensure_parenthesized_conditions(transformed_inner);
        if (transformed_inner.empty()) {
            result += " { }";
        } else {
            result += " { " + transformed_inner + " }";
        }
    } else {
        result += " " + body_fragment;
    }

    return result;
}

bool transform_union_core(std::string_view core, const std::vector<PatternData>& patterns, std::string& output) {
    // Find the colon
    size_t colon_pos = core.find(':');
    if (colon_pos == std::string::npos) {
        return false;
    }

    // Extract the name
    std::string name = trim_copy(core.substr(0, colon_pos));
    if (name.empty()) {
        return false;
    }

    size_t after_name = colon_pos + 1;
    while (after_name < core.size() && std::isspace(static_cast<unsigned char>(core[after_name]))) {
        ++after_name;
    }

    // Locate the union keyword and the body
    size_t union_pos = core.find("@union", after_name);
    if (union_pos == std::string::npos) {
        return false;
    }

    // Detect optional template parameters immediately after @union
    std::string template_prefix;
    size_t after_union = union_pos + 6; // length of "@union"
    while (after_union < core.size() && std::isspace(static_cast<unsigned char>(core[after_union]))) {
        ++after_union;
    }

    if (after_union < core.size() && core[after_union] == '<') {
        size_t template_end = after_union + 1;
        int angle_depth = 1;
        while (template_end < core.size() && angle_depth > 0) {
            char ch = core[template_end];
            if (ch == '<') {
                ++angle_depth;
            } else if (ch == '>') {
                --angle_depth;
            }
            ++template_end;
        }
        if (angle_depth != 0) {
            return false;
        }

        std::string_view template_part = core.substr(after_union, template_end - after_union);
        template_prefix = to_cpp_template_params(template_part);

        after_union = template_end;
        while (after_union < core.size() && std::isspace(static_cast<unsigned char>(core[after_union]))) {
            ++after_union;
        }
    }

    size_t brace_start = core.find('{', union_pos);
    if (brace_start == std::string::npos) {
        return false;
    }

    size_t brace_iter = brace_start + 1;
    int brace_depth = 1;
    while (brace_iter < core.size() && brace_depth > 0) {
        char ch = core[brace_iter];
        if (ch == '{') {
            ++brace_depth;
        } else if (ch == '}') {
            --brace_depth;
        }
        ++brace_iter;
    }
    if (brace_depth != 0) {
        return false;
    }

    size_t brace_end = brace_iter - 1;
    std::string_view body = core.substr(brace_start + 1, brace_end - brace_start - 1);

    struct UnionField {
        std::string name;
        std::string type;
    };

    std::vector<UnionField> fields;
    std::vector<std::string> method_sources;

    size_t pos = 0;
    while (pos < body.size()) {
        while (pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos]))) {
            ++pos;
        }
        if (pos >= body.size()) {
            break;
        }

        size_t entry_start = pos;
        size_t colon = body.find(':', pos);
        if (colon == std::string::npos) {
            break;
        }

        std::string entry_name = trim_copy(body.substr(entry_start, colon - entry_start));
        if (entry_name.empty()) {
            pos = colon + 1;
            continue;
        }

        size_t scan = colon + 1;
        bool in_string = false;
        bool in_char = false;
        bool escape = false;
        int paren_depth = 0;
        int bracket_depth = 0;
        size_t semicolon_pos = std::string::npos;
        size_t equal_pos = std::string::npos;

        for (; scan < body.size(); ++scan) {
            char ch = body[scan];

            if (escape) {
                escape = false;
                continue;
            }

            if (in_string) {
                if (ch == '\\') {
                    escape = true;
                } else if (ch == '"') {
                    in_string = false;
                }
                continue;
            }

            if (in_char) {
                if (ch == '\\') {
                    escape = true;
                } else if (ch == '\'') {
                    in_char = false;
                }
                continue;
            }

            switch (ch) {
                case '"':
                    in_string = true;
                    break;
                case '\'':
                    in_char = true;
                    break;
                case '(':
                    ++paren_depth;
                    break;
                case ')':
                    if (paren_depth > 0) --paren_depth;
                    break;
                case '[':
                    ++bracket_depth;
                    break;
                case ']':
                    if (bracket_depth > 0) --bracket_depth;
                    break;
                case ';':
                    if (paren_depth == 0 && bracket_depth == 0) {
                        semicolon_pos = scan;
                        scan = body.size();
                    }
                    break;
                case '=':
                    if (paren_depth == 0 && bracket_depth == 0) {
                        equal_pos = scan;
                        scan = body.size();
                    }
                    break;
                default:
                    break;
            }
        }

        if (equal_pos != std::string::npos && (semicolon_pos == std::string::npos || equal_pos < semicolon_pos)) {
            // Method: capture until matching closing brace
            size_t brace_open = body.find('{', equal_pos);
            if (brace_open == std::string::npos) {
                pos = equal_pos + 1;
                continue;
            }

            size_t method_iter = brace_open + 1;
            int method_depth = 1;
            bool method_in_string = false;
            bool method_in_char = false;
            bool method_escape = false;
            for (; method_iter < body.size() && method_depth > 0; ++method_iter) {
                char ch = body[method_iter];
                if (method_escape) {
                    method_escape = false;
                    continue;
                }
                if (method_in_string) {
                    if (ch == '\\') method_escape = true;
                    else if (ch == '"') method_in_string = false;
                    continue;
                }
                if (method_in_char) {
                    if (ch == '\\') method_escape = true;
                    else if (ch == '\'') method_in_char = false;
                    continue;
                }

                if (ch == '"') {
                    method_in_string = true;
                } else if (ch == '\'') {
                    method_in_char = true;
                } else if (ch == '{') {
                    ++method_depth;
                } else if (ch == '}') {
                    --method_depth;
                }
            }

            size_t method_end = method_iter;
            while (method_end < body.size() && (body[method_end] == '\n' || body[method_end] == '\r')) {
                ++method_end;
            }

            method_sources.emplace_back(trim_copy(body.substr(entry_start, method_end - entry_start)));
            pos = method_end;
            continue;
        }

        if (semicolon_pos == std::string::npos) {
            break;
        }

        std::string type = trim_copy(body.substr(colon + 1, semicolon_pos - colon - 1));
        if (!type.empty()) {
            fields.push_back({entry_name, map_cpp2_type(type)});
        }

        pos = semicolon_pos + 1;
    }

    if (fields.empty()) {
        return false;
    }

    std::string result;
    if (!template_prefix.empty()) {
        result += template_prefix;
    }

    result += "struct " + name + " {\n";

    result += "    using storage_t = std::variant<std::monostate";
    for (const auto& field : fields) {
        result += ", " + field.type;
    }
    result += ">;\n";
    result += "    storage_t data{};\n\n";

    result += "    bool empty() const { return std::holds_alternative<std::monostate>(data); }\n";

    for (const auto& field : fields) {
        std::string lowered_name = field.name;
        std::replace(lowered_name.begin(), lowered_name.end(), ' ', '_');

        result += "    bool is_" + lowered_name + "() const { return std::holds_alternative<" + field.type + ">(data); }\n";
        result += "    const " + field.type + "& " + lowered_name + "() const { return std::get<" + field.type + ">(data); }\n";
        result += "    " + field.type + "& " + lowered_name + "() { return std::get<" + field.type + ">(data); }\n";
        result += "    template <typename... Args> void set_" + lowered_name + "(Args&&... args) { data.emplace<" + field.type + ">(std::forward<Args>(args)...); }\n\n";
    }

    for (const auto& method_source : method_sources) {
        std::string transformed = transform_union_method(method_source, patterns);
        if (!transformed.empty()) {
            result += indent_block(transformed, "    ");
            result += "\n";
        }
    }

    result += "};\n";

    output = std::move(result);
    return true;
}

bool try_union_rewrite(size_t start_pos, std::string_view source, const std::vector<PatternData>& patterns, size_t& consumed_end, std::string& rewritten) {
    (void)patterns;
    if (start_pos >= source.size()) {
        return false;
    }

    size_t union_pos = source.find("@union", start_pos);
    if (union_pos == std::string::npos) {
        return false;
    }

    size_t colon_pos = std::string::npos;
    size_t search_pos = union_pos;
    while (search_pos > 0) {
        --search_pos;
        char ch = source[search_pos];
        if (ch == ':') {
            bool part_of_scope = (search_pos > 0 && source[search_pos - 1] == ':') ||
                                 (search_pos + 1 < source.size() && source[search_pos + 1] == ':');
            if (!part_of_scope) {
                colon_pos = search_pos;
                break;
            }
        }
        if (ch == ';' || ch == '}') {
            break;
        }
    }

    if (colon_pos == std::string::npos) {
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG try_union_rewrite: failed to locate colon before union starting at " << start_pos
                                     << ", snippet='" << source.substr(start_pos, std::min<size_t>(40, source.size() - start_pos)) << "'\n");
        return false;
    }

    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG try_union_rewrite: evaluating union at start_pos=" << start_pos
                                 << " union_pos=" << union_pos << " colon_pos=" << colon_pos << "\n");

    size_t line_start = colon_pos;
    while (line_start > 0 && source[line_start - 1] != '\n' && source[line_start - 1] != '\r') {
        --line_start;
    }

    size_t scan = colon_pos + 1;
    while (scan < source.size() && std::isspace(static_cast<unsigned char>(source[scan]))) {
        ++scan;
    }

    size_t brace_start = source.find('{', union_pos);
    if (brace_start == std::string::npos) {
        return false;
    }

    size_t brace_end = brace_start;
    int depth = 1;
    while (brace_end < source.size() && depth > 0) {
        ++brace_end;
        if (source[brace_end] == '{') ++depth;
        else if (source[brace_end] == '}') --depth;
    }
    if (depth != 0) {
        return false;
    }

    // Find the end of the line/statement
    size_t end_pos = brace_end + 1;
    while (end_pos < source.size() && source[end_pos] != '\n' && source[end_pos] != '\r') {
        ++end_pos;
    }
    if (end_pos < source.size() && source[end_pos] == '\r') {
        ++end_pos;
    }
    if (end_pos < source.size() && source[end_pos] == '\n') {
        ++end_pos;
    }

    std::string core = std::string(source.substr(line_start, brace_end + 1 - line_start));

    std::string transformed_core;
    if (!transform_union_core(core, patterns, transformed_core)) {
        return false;
    }

    std::string trailing;
    if (brace_end + 1 < end_pos) {
        trailing.assign(source.substr(brace_end + 1, end_pos - (brace_end + 1)));
    }

    rewritten = transformed_core;
    rewritten += trailing;

    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG try_union_rewrite: succeeded for span [" << line_start << ", " << end_pos << ")\n");

    consumed_end = end_pos;
    return true;
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
std::string transform_parameter(const std::string& param, const std::vector<PatternData>& patterns, bool prefer_value_for_in) {
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
    if (transform_cpp2_parameter_direct(p, direct_result, prefer_value_for_in)) {
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

    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG transform_to_lambda: name='" << name << "' params='" << params << "' return_type='" << return_type << "' body='" << body << "' nesting_depth=" << nesting_depth << "\n");

    // Recursively transform the body with increased nesting depth
    std::string transformed_body = apply_recursive_transformations(body, patterns, nesting_depth + 1);

    // Transform parameters
    std::string transformed_params = transform_parameter(params, patterns, /*prefer_value_for_in=*/true);

    // Build lambda: auto name = [](params) -> return_type { body }; or auto name = [](params) { body };
    std::string lambda = "auto " + name + " = [](" + transformed_params + ")";
    if (!return_type.empty()) {
        lambda += " -> " + return_type;
    }
    lambda += " " + transformed_body + ";";

    return lambda;
}

std::string transform_nested_function_statement(std::string_view statement, const std::vector<PatternData>& patterns, int nesting_depth) {
    std::string trimmed = trim_copy(statement);
    if (trimmed.empty()) {
        return {};
    }

    size_t colon_pos = find_top_level_colon(trimmed);
    if (colon_pos == std::string::npos) {
        return {};
    }

    std::string name = trim_copy(trimmed.substr(0, colon_pos));
    if (name.empty()) {
        return {};
    }

    size_t open_paren = trimmed.find('(', colon_pos);
    if (open_paren == std::string::npos) {
        return {};
    }

    size_t close_paren = std::string::npos;
    int depth = 0;
    for (size_t i = open_paren; i < trimmed.size(); ++i) {
        char ch = trimmed[i];
        if (ch == '(') {
            ++depth;
        } else if (ch == ')') {
            if (depth == 0) {
                return {};
            }
            --depth;
            if (depth == 0) {
                close_paren = i;
                break;
            }
        }
    }
    if (close_paren == std::string::npos) {
        return {};
    }

    std::string params_section = trimmed.substr(open_paren + 1, close_paren - open_paren - 1);

    std::string after_paren = trimmed.substr(close_paren + 1);
    trim_in_place(after_paren);
    if (after_paren.empty()) {
        return {};
    }

    size_t equal_pos = find_top_level_equal(after_paren);
    if (equal_pos == std::string::npos) {
        return {};
    }

    std::string signature_suffix = trim_copy(after_paren.substr(0, equal_pos));
    std::string body = trim_copy(after_paren.substr(equal_pos + 1));
    if (body.empty()) {
        return {};
    }

    if (!body.empty() && body.back() == ';') {
        body.pop_back();
        trim_in_place(body);
    }

    SignatureModifiers modifiers = parse_signature_modifiers(signature_suffix);
    if (!modifiers.residual.empty()) {
        return {};
    }

    std::string transformed_params = transform_parameter(params_section, patterns, /*prefer_value_for_in=*/true);

    std::string lambda_return;
    if (modifiers.has_return) {
        lambda_return = transform_return_spec(modifiers.return_type);
    }

    bool body_has_braces = !body.empty() && body.front() == '{';
    std::string transformed_body;
    if (body_has_braces) {
        transformed_body = apply_recursive_transformations(body, patterns, nesting_depth + 1);
        trim_in_place(transformed_body);
        if (transformed_body.empty()) {
            transformed_body = "{}";
        }
    } else {
        std::string expr = body;
        if (!expr.empty() && expr.front() == '=') {
            expr.erase(expr.begin());
            trim_in_place(expr);
        }
        if (expr.empty()) {
            return {};
        }
        transformed_body = "{ return " + expr + "; }";
        if (!modifiers.has_return) {
            lambda_return.clear();
        }
    }

    std::string lambda = "auto " + name + " = [](" + transformed_params + ")";
    if (!lambda_return.empty()) {
        lambda += " -> " + lambda_return;
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

std::string rewrite_statement_from_match(const std::string& statement, const UnifiedPatternMatch& match, const std::vector<PatternData>& patterns, int nesting_depth) {
    if (!match.pattern) {
        return {};
    }

    const PatternData& pattern = *match.pattern;
    auto segments = match.segments;

    CPP2_EMITTER_DEBUG({
        std::cerr << "DEBUG rewrite_statement_from_match: pattern='" << pattern.name << "' segments=";
        for (size_t i = 0; i < segments.size(); ++i) {
            if (i != 0) {
                std::cerr << ", ";
            }
            std::cerr << "['" << segments[i] << "']";
        }
        std::cerr << "\n";
    });

    bool is_function_pattern = pattern.name.find("function") != std::string::npos;
    bool is_main_pattern = pattern.name.find("main") != std::string::npos;
    bool is_main_function = (!segments.empty() && segments[0].find("main") != std::string::npos);

    if (is_function_pattern && !is_main_pattern && !is_main_function && nesting_depth > 0 && segments.size() >= 3) {
        std::string lambda = transform_nested_function_statement(statement, patterns, nesting_depth);
        if (!lambda.empty()) {
            return lambda;
        }
        lambda = transform_to_lambda(segments, patterns, nesting_depth);
        if (!lambda.empty()) {
            return lambda;
        }
    }

    for (size_t i = 0; i < segments.size(); ++i) {
        std::string evidence = (pattern.evidence_types.size() > i) ? pattern.evidence_types[i] : std::string{};
        if (is_function_pattern && evidence == "parameters") {
            segments[i] = transform_parameter(segments[i], patterns);
        } else if (evidence == "body" || evidence == "expression" || evidence == "initializer") {
            segments[i] = apply_recursive_transformations(segments[i], patterns, nesting_depth + 1);
        }
    }

    auto template_it = pattern.substitution_templates.find(2);
    if (template_it == pattern.substitution_templates.end()) {
        return {};
    }

    int placeholder_offset = pattern.use_alternating ? 1 : 0;
    std::string transformed = apply_substitution_with_offset(template_it->second, segments, placeholder_offset);

    if (pattern.name == "cpp2_typed_variable" && segments.size() >= 3) {
        std::string alias_name = trim_copy(segments[0]);
        std::string alias_type = trim_copy(segments[1]);
        std::string alias_target = trim_copy(segments[2]);
        if (!alias_name.empty() && alias_type == "type" && !alias_target.empty()) {
            transformed = "using " + alias_name + " = " + alias_target + ";";
        }
    }

    auto ensure_semicolon = [](std::string& text) {
        size_t last_non_space = text.find_last_not_of(" \t\n\r");
        if (last_non_space == std::string::npos || text[last_non_space] != ';') {
            text += ";";
        }
    };

    std::string post_match = statement.substr(match.end_pos);
    size_t idx = 0;
    while (idx < post_match.size() && std::isspace(static_cast<unsigned char>(post_match[idx]))) {
        ++idx;
    }
    bool has_semicolon = (idx < post_match.size() && post_match[idx] == ';');
    if (has_semicolon) {
        ensure_semicolon(transformed);
        ++idx;
    }
    std::string spacing_after_semicolon;
    while (idx < post_match.size() && std::isspace(static_cast<unsigned char>(post_match[idx]))) {
        spacing_after_semicolon.push_back(post_match[idx]);
        ++idx;
    }
    std::string trailing_text = post_match.substr(idx);

    if (!has_semicolon) {
        transformed += post_match;
        return transformed;
    }

    transformed += spacing_after_semicolon;
    transformed += trailing_text;
    return transformed;
}

std::string transform_statement(const std::string& statement, const std::vector<PatternData>& patterns, int nesting_depth) {
    std::string working = statement;
    if (working.empty()) {
        return working;
    }

    if (auto contract = rewrite_contract_statement(working); !contract.empty()) {
        return contract;
    }

    std::string namespace_rewrite = rewrite_using_namespace_placeholder(working);
    if (namespace_rewrite != working) {
        return namespace_rewrite;
    }

    std::string placeholder_rewrite = rewrite_default_placeholders(working);
    if (placeholder_rewrite != working) {
        working = std::move(placeholder_rewrite);
    }

    std::string as_rewrite = rewrite_as_expressions(working);
    if (as_rewrite != working) {
        working = std::move(as_rewrite);
    }

    if (working.empty()) {
        return working;
    }

    std::string trimmed_statement = trim_copy(working);

    size_t leading_ws_pos = working.find_first_not_of(" \t");
    if (leading_ws_pos != std::string::npos && working[leading_ws_pos] == '_' &&
        trimmed_statement.size() > 2 && trimmed_statement[1] == ' ' && trimmed_statement[2] == '=') {
        size_t eq_pos = working.find('=', leading_ws_pos);
        if (eq_pos != std::string::npos) {
            // Skip compound operators like '==' or '+='
            if (eq_pos + 1 < working.size() && working[eq_pos + 1] == '=') {
                // Not a simple assignment, skip transformation
            } else {
                size_t semi_pos = working.find(';', eq_pos);
                if (semi_pos != std::string::npos) {
                    std::string expr = working.substr(eq_pos + 1, semi_pos - eq_pos - 1);
                    std::string sanitized_expr = trim_copy(expr);
                    if (!sanitized_expr.empty()) {
                        if (auto rewritten = rewrite_known_ufcs_call(sanitized_expr)) {
                            sanitized_expr = *rewritten;
                        }
                        std::string prefix = working.substr(0, leading_ws_pos);
                        std::string suffix = working.substr(semi_pos + 1);
                        return prefix + "static_cast<void>(" + sanitized_expr + ");" + suffix;
                    }
                }
            }
        }
    }

    if (working.front() == '{') {
        size_t first_non_ws = working.find_first_not_of(" \t\n\r");
        size_t last_non_ws = working.find_last_not_of(" \t\n\r");
        if (first_non_ws == std::string::npos || last_non_ws == std::string::npos) {
            return working;
        }

        size_t open_brace = working.find('{', first_non_ws);
        if (open_brace == std::string::npos) {
            return working;
        }
        if (working[last_non_ws] != '}' || last_non_ws <= open_brace) {
            return working;
        }

        std::string prefix = working.substr(0, open_brace);
        std::string suffix = working.substr(last_non_ws + 1);
        std::string inner = working.substr(open_brace + 1, last_non_ws - open_brace - 1);
        std::string transformed_inner = apply_recursive_transformations(inner, patterns, nesting_depth + 1);
        std::string trimmed_inner = trim_copy(transformed_inner);
        std::string rebuilt = trimmed_inner.empty() ? std::string{"{}"}
                                                    : std::string{"{ "} + trimmed_inner + " }";
        return prefix + rebuilt + suffix;
    }

    auto matches = UnifiedPatternMatcher::find_matches(working, patterns, true, 0);
    if (matches.empty()) {
        std::string fallback = transform_variable_declaration(working);
        if (!fallback.empty() && fallback != working) {
            return fallback;
        }
        return working;
    }

    const UnifiedPatternMatch* best_match = nullptr;
    int best_priority = std::numeric_limits<int>::min();
    size_t best_span = 0;
    for (const auto& match : matches) {
        if (match.depth != 0) {
            continue;
        }
        if (match.start_pos != 0) {
            continue;
        }
        int priority = match.pattern ? match.pattern->priority : 0;
        size_t span = match.end_pos > match.start_pos ? (match.end_pos - match.start_pos) : 0;
        if (!best_match || priority > best_priority || (priority == best_priority && span > best_span)) {
            best_match = &match;
            best_priority = priority;
            best_span = span;
        }
    }

    if (!best_match) {
        std::string fallback = transform_variable_declaration(working);
        if (!fallback.empty() && fallback != working) {
            return fallback;
        }
        return working;
    }

    std::string rewritten = rewrite_statement_from_match(working, *best_match, patterns, nesting_depth);
    if (rewritten.empty()) {
        return working;
    }
    return rewritten;
}

// Recursive orbit transformation for nested segments
std::string apply_recursive_transformations(const std::string& input, const std::vector<PatternData>& patterns, int nesting_depth) {
    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG apply_recursive_transformations: input='" << input << "' nesting_depth=" << nesting_depth << "\n");

    std::string working = input;
    bool has_braces = false;
    std::string outer_prefix;
    std::string outer_suffix;
    if (!working.empty()) {
        size_t first_non_ws = working.find_first_not_of(" \t\n\r");
        size_t last_non_ws = working.find_last_not_of(" \t\n\r");
        if (first_non_ws != std::string::npos && last_non_ws != std::string::npos &&
            working[first_non_ws] == '{' && working[last_non_ws] == '}' && last_non_ws > first_non_ws) {
            outer_prefix = working.substr(0, first_non_ws);
            outer_suffix = working.substr(last_non_ws + 1);
            working = working.substr(first_non_ws + 1, last_non_ws - first_non_ws - 1);
            has_braces = true;
            CPP2_EMITTER_DEBUG(std::cerr << "DEBUG apply_recursive_transformations: stripped braces, working='" << working << "'\n");
        }
    }

    while (transform_for_next_do_blocks(working, patterns, nesting_depth)) {
        // Keep normalizing until no further next/do patterns remain
    }

    std::string rebuilt;
    size_t pos = 0;
    while (pos < working.size()) {
        size_t whitespace_start = pos;
        while (pos < working.size() && std::isspace(static_cast<unsigned char>(working[pos]))) {
            ++pos;
        }
        rebuilt.append(working, whitespace_start, pos - whitespace_start);
        if (pos >= working.size()) {
            break;
        }

        size_t statement_end = find_statement_end(working, pos);
        if (statement_end <= pos) {
            rebuilt.append(working.substr(pos));
            break;
        }

        std::string statement = working.substr(pos, statement_end - pos);
        std::string transformed = transform_statement(statement, patterns, nesting_depth);
        rebuilt += transformed;
        pos = statement_end;
    }

    if (has_braces) {
        std::string trimmed = trim_copy(rebuilt);
        std::string rebuilt_block;
        if (trimmed.empty()) {
            rebuilt_block = "{}";
        } else {
            rebuilt_block = "{ " + trimmed + " }";
        }
        return outer_prefix + rebuilt_block + outer_suffix;
    }

    return rebuilt;
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
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG extract_alternating_segments: first anchor '" << first_anchor
                                     << "' not found in text='" << text << "'\n");
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
                CPP2_EMITTER_DEBUG(std::cerr << "DEBUG extract_alternating_segments: anchor '" << next_anchor
                                             << "' not found after position " << current_pos << " in text='" << text << "'\n");
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
        int brace_balance = 0;
        for (size_t idx = 0; idx < evidence.size(); ++idx) {
            char ch = evidence[idx];
            if (ch == '{') {
                ++brace_balance;
            } else if (ch == '}') {
                if (brace_balance == 0) {
                    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG extract_alternating_segments: unmatched closing brace in evidence for pattern '"
                                                 << pattern.name << "' text='" << evidence << "'\n");
                    return {};
                }
                --brace_balance;
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
        {"std::span", "span"},
        {"int8_t", "cstdint"},
        {"int16_t", "cstdint"},
        {"int32_t", "cstdint"},
        {"int64_t", "cstdint"},
        {"uint8_t", "cstdint"},
        {"uint16_t", "cstdint"},
        {"uint32_t", "cstdint"},
        {"uint64_t", "cstdint"},
        {"std::int8_t", "cstdint"},
        {"std::int16_t", "cstdint"},
        {"std::int32_t", "cstdint"},
        {"std::int64_t", "cstdint"},
        {"std::uint8_t", "cstdint"},
        {"std::uint16_t", "cstdint"},
        {"std::uint32_t", "cstdint"},
        {"std::uint64_t", "cstdint"},
        {"std::fprintf", "cstdio"},
        {"std::fclose", "cstdio"},
        {"std::fscanf", "cstdio"},
        {"std::fgets", "cstdio"},
        {"std::fputs", "cstdio"},
        {"std::fread", "cstdio"},
        {"std::fwrite", "cstdio"},
        {"std::fflush", "cstdio"},
        {"i8", "cstdint"},
        {"i16", "cstdint"},
        {"i32", "cstdint"},
        {"i64", "cstdint"},
        {"u8", "cstdint"},
        {"u16", "cstdint"},
        {"u32", "cstdint"},
        {"u64", "cstdint"},
    };

    for (const auto& mapping : mappings) {
        if (source.find(mapping.type) != std::string::npos) {
            add_include(std::string("#include <") + mapping.header + ">" );
        }
    }

    auto add_cstdio_if_needed = [&](std::string_view needle) {
        if (source.find(needle) != std::string::npos) {
            add_include("#include <cstdio>");
        }
    };

    add_cstdio_if_needed(".fprintf");
    add_cstdio_if_needed(".fclose");
    add_cstdio_if_needed(".fscanf");
    add_cstdio_if_needed(".fgets");
    add_cstdio_if_needed(".fputs");
    add_cstdio_if_needed(".fread");
    add_cstdio_if_needed(".fwrite");
    add_cstdio_if_needed(".fflush");

    if (source.find("cpp2::impl::out<") != std::string::npos || has_out_parameter(source)) {
        add_include("#include \"cpp2_inline.h\"");
    }

    return includes;
}

class IncludeDeduper {
public:
    IncludeDeduper() = default;

    void note_emitted(std::string_view line) {
        std::string trimmed = trim_copy(line);
        if (!trimmed.empty() && trimmed.front() == '#') {
            seen_.insert(std::move(trimmed));
        }
    }

    std::string filter(std::string_view block) {
        if (block.empty()) {
            return {};
        }

        std::string result;
        result.reserve(block.size());

        size_t pos = 0;
        while (pos <= block.size()) {
            size_t newline = block.find('\n', pos);
            const bool has_newline = (newline != std::string::npos);
            std::string_view line_view = has_newline ? block.substr(pos, newline - pos)
                                                     : block.substr(pos);

            bool drop_line = false;
            std::string trimmed = trim_copy(line_view);
            if (!trimmed.empty() && trimmed.front() == '#') {
                auto [_, inserted] = seen_.insert(trimmed);
                drop_line = !inserted;
            }

            if (!drop_line) {
                result.append(line_view.begin(), line_view.end());
                if (has_newline) {
                    result.push_back('\n');
                }
            }

            if (!has_newline) {
                break;
            }
            pos = newline + 1;
        }

        return result;
    }

private:
    std::unordered_set<std::string> seen_;
};

// Depth-based emit: deterministic pattern matching without speculation
void CPP2Emitter::emit_depth_based(std::string_view source, std::ostream& out, const std::vector<PatternData>& patterns) const {
    IncludeDeduper include_deduper;

    // Emit includes first
    auto includes = extract_std_includes(source);
    for (const auto& include : includes) {
        out << include << "\n";
        include_deduper.note_emitted(include);
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
    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit_depth_based: Processing line: '" << line << "'\n");

        std::string rendered_line;
        std::string trimmed_line = trim_copy(line);

        if (!trimmed_line.empty() && trimmed_line.front() == '#') {
            rendered_line.assign(line.begin(), line.end());
        } else {
            auto matches = UnifiedPatternMatcher::find_matches(line, patterns, true);
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit_depth_based: Found " << matches.size() << " matches\n");

            if (matches.empty()) {
                std::string transformed_line = transform_function_signature_line(std::string(line), patterns);
                if (!transformed_line.empty()) {
                    rendered_line = std::move(transformed_line);
                } else {
                    rendered_line.assign(line.begin(), line.end());
                }
            } else {
                size_t last_pos = 0;
                rendered_line.reserve(line.size() * 2);

                // Apply transformations in order (innermost first, as returned by find_matches)
                for (const auto& match : matches) {
                    // Emit text before this match
                    if (last_pos < match.start_pos) {
                        rendered_line.append(line.substr(last_pos, match.start_pos - last_pos));
                    }

                    // Apply substitution for this match
                    int target_grammar = 2; // CPP target
                    auto template_it = match.pattern->substitution_templates.find(target_grammar);

                    CPP2_EMITTER_DEBUG({
                        std::cerr << "DEBUG emit_depth_based: match pattern='" << match.pattern->name << "' start=" << match.start_pos << " end=" << match.end_pos << " segments=";
                        for (size_t idx = 0; idx < match.segments.size(); ++idx) {
                            if (idx != 0) {
                                std::cerr << ", ";
                            }
                            std::cerr << "['" << match.segments[idx] << "']";
                        }
                        std::cerr << "\n";
                    });

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
                        rendered_line.append(result);
                    } else {
                        // No template for target grammar - emit original
                        rendered_line.append(line.substr(match.start_pos, match.end_pos - match.start_pos));
                    }

                    last_pos = match.end_pos;
                }

                // Emit remaining text from this line
                if (last_pos < line.size()) {
                    rendered_line.append(line.substr(last_pos));
                }
            }
        }

        std::string filtered = include_deduper.filter(rendered_line);
        if (filtered.empty()) {
            continue;
        }

        if (!first) {
            out << "\n";
        }
        first = false;
        out << filtered;
    }
}

void CPP2Emitter::emit(OrbitIterator& iterator, std::string_view source, std::ostream& raw_out, const std::vector<PatternData>& patterns) const {
    std::ostringstream generated_output;
    std::ostream& out = generated_output;
    IncludeDeduper include_deduper;

    // Emit includes first
    auto includes = extract_std_includes(source);
    for (const auto& include : includes) {
        out << include << "\n";
        include_deduper.note_emitted(include);
    }

    auto sanitize_segment = [&](std::string text) {
        return include_deduper.filter(text);
    };

    // Reset iterator to beginning
    iterator.reset();

    size_t last_pos = 0;
    size_t skip_until = 0;
    bool found_first_orbit = false;

    // Iterate through all orbits and emit their results
    size_t orbit_count = 0;
    for (cppfort::stage0::Orbit* orbit = iterator.next(); orbit; orbit = iterator.next()) {
        if (auto* confix = dynamic_cast<cppfort::stage0::ConfixOrbit*>(orbit)) {
            orbit_count++;
            CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Orbit #" << orbit_count << " [" << confix->start_pos << ", " << confix->end_pos << ") pattern=" << confix->selected_pattern() << "\n");
            CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Orbit text: '" << source.substr(confix->start_pos, std::min(size_t(50), confix->end_pos - confix->start_pos)) << "...'\n");

            if (skip_until > last_pos) {
                last_pos = skip_until;
            }

            if (confix->end_pos <= last_pos) {
                CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Skipping orbit entirely within consumed range\n");
                continue;
            }

            // On first orbit, handle any initial text before it
            if (!found_first_orbit && confix->start_pos > 0) {
                std::string initial = std::string(source.substr(0, confix->start_pos));
                CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Initial segment before first orbit [0, " << confix->start_pos << "): '" << initial << "'\n");
                std::string rewritten_initial = rewrite_free_text_segment(initial, patterns);
                if (rewritten_initial != initial) {
                    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Rewritten initial -> '" << rewritten_initial << "'\n");
                }
                std::string sanitized = sanitize_segment(std::move(rewritten_initial));
                out << sanitized;
                last_pos = confix->start_pos;
                found_first_orbit = true;
            }

            // Fill gap between last position and current orbit
            if (last_pos < confix->start_pos) {
                size_t consumed_gap_end = 0;
                std::string rewritten_gap_stmt;
                if (try_union_rewrite(last_pos, source, patterns, consumed_gap_end, rewritten_gap_stmt)) {
                    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Gap union rewrite consumed up to " << consumed_gap_end << "\n");
                    out << rewritten_gap_stmt;
                    skip_until = consumed_gap_end;
                    last_pos = consumed_gap_end;
                    continue;
                }

                std::string gap = std::string(source.substr(last_pos, confix->start_pos - last_pos));
                CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Gap [" << last_pos << ", " << confix->start_pos << "): '" << gap << "'\n");
                std::string rewritten_gap = rewrite_free_text_segment(gap, patterns);
                if (rewritten_gap != gap) {
                    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Rewritten gap -> '" << rewritten_gap << "'\n");
                }
                std::string sanitized = sanitize_segment(std::move(rewritten_gap));
                out << sanitized;
                last_pos = confix->start_pos;
            }
            
            found_first_orbit = true;

            if (confix->start_pos < last_pos) {
                CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Orbit starts inside consumed range, skipping\n");
                continue;
            }

            std::string selected_pattern_name = confix->selected_pattern();
            if (selected_pattern_name.empty() || selected_pattern_name == "default") {
                size_t consumed_end = 0;
                std::string rewritten_stmt;
                if (try_template_alias_rewrite(confix->start_pos, source, patterns, consumed_end, rewritten_stmt)) {
                    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Template alias rewrite consumed up to " << consumed_end << "\n");
                    std::string sanitized = sanitize_segment(std::move(rewritten_stmt));
                    out << sanitized;
                    skip_until = consumed_end;
                    last_pos = consumed_end;
                    continue;
                }
                if (try_union_rewrite(confix->start_pos, source, patterns, consumed_end, rewritten_stmt)) {
                    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Union rewrite consumed up to " << consumed_end << "\n");
                    std::string sanitized = sanitize_segment(std::move(rewritten_stmt));
                    out << sanitized;
                    skip_until = consumed_end;
                    last_pos = consumed_end;
                    continue;
                }
                if (try_function_signature_rewrite(confix->start_pos, source, patterns, consumed_end, rewritten_stmt)) {
                    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Function signature rewrite consumed up to " << consumed_end << "\n");
                    std::string sanitized = sanitize_segment(std::move(rewritten_stmt));
                    out << sanitized;
                    skip_until = consumed_end;
                    last_pos = consumed_end;
                    continue;
                }
                // Attempt variable declaration rewrite at this position
                {
                    size_t stmt_end = find_statement_end(source, confix->start_pos);
                    if (stmt_end > confix->start_pos) {
                        std::string stmt = std::string(source.substr(confix->start_pos, stmt_end - confix->start_pos));
                        std::string var_rewritten = transform_variable_declaration(stmt);
                        if (!var_rewritten.empty() && var_rewritten != stmt) {
                            CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Variable rewrite consumed up to " << stmt_end << "\n");
                            std::string sanitized = sanitize_segment(std::move(var_rewritten));
                            out << sanitized;
                            skip_until = stmt_end;
                            last_pos = stmt_end;
                            continue;
                        }
                    }
                }
            }

            // Find pattern for this orbit
            const PatternData* pattern = nullptr;
            std::string pattern_name = selected_pattern_name;

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
    CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Total orbits processed: " << orbit_count << "\n");

    // Fill remaining text after last orbit
    if (last_pos < source.size()) {
        std::string trailing = std::string(source.substr(last_pos));
        std::string rewritten_trailing = rewrite_free_text_segment(trailing, patterns);
        if (rewritten_trailing != trailing) {
            CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit: Rewritten trailing segment -> '" << rewritten_trailing << "'\n");
        }
        std::string sanitized = sanitize_segment(std::move(rewritten_trailing));
        out << sanitized;
    }

    raw_out << finalize_cpp2_output(generated_output.str());
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
        // Fallback: rewrite statements line-by-line to handle variables and functions
        std::string rewritten = rewrite_free_text_segment(text, all_patterns);
        out << rewritten;
        return;
    }

    const bool nested_function_pattern =
        pattern->name == "cpp2_nested_function_with_return" ||
        pattern->name == "cpp2_nested_function_void";

    if (nested_function_pattern) {
        int depth_before = brace_depth_before(source, orbit.start_pos);
        if (depth_before == 0) {
            std::string rewritten = transform_function_signature_line(std::string(text), all_patterns);
            if (!rewritten.empty()) {
                out << rewritten;
                return;
            }
        }
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
    const auto& captured = orbit.captured_segments();
    const bool captured_ok = !captured.empty() &&
                              (!pattern->use_alternating || captured.size() == pattern->evidence_types.size());

    if (captured_ok) {
        segments = captured;
    } else if (pattern->use_alternating) {
        // Extract evidence spans for alternating anchor/evidence pattern
        segments = extract_alternating_segments(text, *pattern);
    } else {
        // Try pattern matching for function_declaration pattern
        if (pattern->name == "function_declaration" && !pattern->signature_patterns.empty()) {
            // Use tblgen pattern matcher for CPP2 pattern
            std::string cpp2_pattern = "$0: ($1) -> $2 = $3";
            auto match_result = UnifiedPatternMatcher::extract_segments(cpp2_pattern, std::string(text));

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

    if (pattern->use_alternating && !pattern->evidence_types.empty() &&
        segments.size() < pattern->evidence_types.size()) {
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG emit_orbit: insufficient segments for pattern '" << pattern->name
                                     << "' (expected " << pattern->evidence_types.size() << ", got " << segments.size()
                                     << ")\n");
        out << text;
        return;
    }

    // Apply recursive transformations to segments BEFORE substitution
    // This handles nested CPP2 syntax like variable declarations in function bodies
    bool is_function = (pattern->name.find("function") != std::string::npos);
    for (size_t i = 0; i < segments.size(); ++i) {
        auto& seg = segments[i];
        std::string_view evidence_type;
        if (i < pattern->evidence_types.size()) {
            evidence_type = pattern->evidence_types[i];
        }

        if (evidence_type == "expression") {
            trim_in_place(seg);
            if (!seg.empty() && seg.back() == ';') {
                seg.pop_back();
                trim_in_place(seg);
            }
        }

        // Check if this segment is parameters based on evidence type
        bool is_parameters = (evidence_type == "parameters");

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

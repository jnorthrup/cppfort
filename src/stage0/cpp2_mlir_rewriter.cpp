#include "cpp2_mlir_rewriter.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

#include "cpp2_mlir_loader.h"

namespace cppfort::stage0 {
namespace {

struct CutoutFragment {
    std::vector<int64_t> path;
    std::size_t length = 0;
    std::string patternName;
    std::string hierarchyLabel;
    std::string snippet;
};

std::string_view trim_view(std::string_view view) {
    std::size_t start = 0;
    while (start < view.size() && std::isspace(static_cast<unsigned char>(view[start]))) {
        ++start;
    }
    std::size_t end = view.size();
    while (end > start && std::isspace(static_cast<unsigned char>(view[end - 1]))) {
        --end;
    }
    return view.substr(start, end - start);
}

std::string trim_copy(std::string_view view) {
    return std::string(trim_view(view));
}

bool hasBalancedDelimiters(std::string_view text, char open, char close) {
    int depth = 0;
    for (char ch : text) {
        if (ch == open) {
            ++depth;
        } else if (ch == close) {
            --depth;
            if (depth < 0) {
                return false;
            }
        }
    }
    return depth == 0;
}

std::string strip_outer_parens(std::string_view view) {
    view = trim_view(view);
    if (view.size() >= 2 && view.front() == '(' && view.back() == ')') {
        std::string inner(view.substr(1, view.size() - 2));
        if (hasBalancedDelimiters(inner, '(', ')')) {
            return trim_copy(inner);
        }
    }
    return trim_copy(view);
}

std::string ensure_string_literal(std::string_view view) {
    std::string cleaned = trim_copy(view);
    if (cleaned.empty()) {
        return "\"\"";
    }
    if (cleaned.front() == '"' && cleaned.back() == '"') {
        return cleaned;
    }
    return "\"" + cleaned + "\"";
}

std::size_t find_assignment_equals(std::string_view text, std::size_t begin) {
    std::size_t pos = begin;
    while (pos < text.size()) {
        pos = text.find('=', pos);
        if (pos == std::string_view::npos) {
            return pos;
        }
        if (pos > 0) {
            const char prev = text[pos - 1];
            if (prev == '<' || prev == '>' || prev == '!') {
                ++pos;
                continue;
            }
        }
        if (pos + 1 < text.size()) {
            const char next = text[pos + 1];
            if (next == '=' || next == '>') {
                ++pos;
                continue;
            }
        }
        return pos;
    }
    return std::string_view::npos;
}

bool startsWith(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() &&
           value.substr(0, prefix.size()) == prefix;
}

bool endsWith(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

struct ParameterParts {
    std::string name;
    std::string type;
    std::string defaultValue;
    std::string passingMode;
};

std::string applyPassingMode(std::string type, std::string_view mode) {
    if (mode.empty() || type.empty()) {
        return type;
    }
    auto ensureReference = [](std::string& in, std::string_view ref) {
        if (!endsWith(in, "&") && !endsWith(in, "&&")) {
            in.append(ref);
        }
    };
    const std::string trimmedType = trim_copy(type);
    std::string result = trimmedType.empty() ? type : trimmedType;

    if (mode == "in") {
        if (!startsWith(result, "const ")) {
            result = "const " + result;
        }
        ensureReference(result, "&");
        return result;
    }
    if (mode == "out" || mode == "inout") {
        ensureReference(result, "&");
        return result;
    }
    if (mode == "move" || mode == "forward") {
        if (!endsWith(result, "&&")) {
            if (endsWith(result, "&")) {
                result.pop_back();
                result.push_back('&');
            } else {
                result.append("&&");
            }
        }
        return result;
    }
    return trimmedType.empty() ? type : trimmedType;
}

bool isPassingMode(std::string_view candidate) {
    return candidate == "in" || candidate == "out" || candidate == "inout" ||
           candidate == "copy" || candidate == "move" || candidate == "forward";
}

std::optional<ParameterParts> parseParameter(std::string_view text) {
    text = trim_view(text);
    const std::size_t colon = text.find(':');
    if (colon == std::string_view::npos) {
        return std::nullopt;
    }

    ParameterParts parts;
    parts.name = trim_copy(text.substr(0, colon));
    std::string_view remainder = trim_view(text.substr(colon + 1));
    if (parts.name.empty() || remainder.empty()) {
        return std::nullopt;
    }

    const std::size_t equals = remainder.find('=');
    std::string_view typeSegment = equals == std::string_view::npos
        ? remainder
        : trim_view(remainder.substr(0, equals));
    if (equals != std::string_view::npos) {
        parts.defaultValue = trim_copy(remainder.substr(equals + 1));
    }

    std::size_t firstSpace = typeSegment.find(' ');
    if (firstSpace != std::string_view::npos) {
        std::string_view possibleMode = trim_view(typeSegment.substr(0, firstSpace));
        if (isPassingMode(possibleMode)) {
            parts.passingMode = std::string(possibleMode);
            typeSegment = trim_view(typeSegment.substr(firstSpace + 1));
        }
    }

    parts.type = trim_copy(typeSegment);
    if (parts.type.empty()) {
        return std::nullopt;
    }
    return parts;
}

std::optional<std::string> rewriteParameter(std::string_view text) {
    auto parts = parseParameter(text);
    if (!parts) {
        return std::nullopt;
    }
    std::string type = applyPassingMode(parts->type, parts->passingMode);
    if (parts->passingMode == "copy") {
        type = parts->type;
    }
    std::string rewritten = type.empty() ? parts->name : type + " " + parts->name;
    if (!parts->defaultValue.empty()) {
        rewritten.append(" = ").append(parts->defaultValue);
    }
    return rewritten;
}

std::vector<std::string> splitCommaBalanced(std::string_view text) {
    std::vector<std::string> segments;
    std::string current;
    int parenDepth = 0;
    int angleDepth = 0;
    int braceDepth = 0;
    int bracketDepth = 0;
    char stringDelimiter = 0;

    for (std::size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (stringDelimiter != 0) {
            current.push_back(ch);
            if (ch == stringDelimiter && (i == 0 || text[i - 1] != '\\')) {
                stringDelimiter = 0;
            }
            continue;
        }
        if (ch == '"' || ch == '\'') {
            stringDelimiter = ch;
            current.push_back(ch);
            continue;
        }
        switch (ch) {
        case '(':
            ++parenDepth;
            break;
        case ')':
            if (parenDepth > 0) {
                --parenDepth;
            }
            break;
        case '<':
            ++angleDepth;
            break;
        case '>':
            if (angleDepth > 0) {
                --angleDepth;
            }
            break;
        case '[':
            ++bracketDepth;
            break;
        case ']':
            if (bracketDepth > 0) {
                --bracketDepth;
            }
            break;
        case '{':
            ++braceDepth;
            break;
        case '}':
            if (braceDepth > 0) {
                --braceDepth;
            }
            break;
        case ',':
            if (parenDepth == 0 && angleDepth == 0 &&
                bracketDepth == 0 && braceDepth == 0) {
                segments.push_back(trim_copy(current));
                current.clear();
                continue;
            }
            break;
        default:
            break;
        }
        current.push_back(ch);
    }

    if (!current.empty()) {
        segments.push_back(trim_copy(current));
    } else if (!text.empty() && text.back() == ',') {
        segments.emplace_back();
    }
    return segments;
}

std::string joinStrings(const std::vector<std::string>& values, std::string_view delimiter) {
    if (values.empty()) {
        return {};
    }
    std::ostringstream oss;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << delimiter;
        }
        oss << values[i];
    }
    return oss.str();
}

std::optional<std::string> rewriteVarAuto(std::string_view text) {
    const std::size_t walrus = text.find(":=");
    if (walrus == std::string_view::npos) {
        return std::nullopt;
    }
    const std::string name = trim_copy(text.substr(0, walrus));
    const std::string initializer = trim_copy(text.substr(walrus + 2));
    if (name.empty() || initializer.empty()) {
        return std::nullopt;
    }
    std::string rewritten = "auto ";
    rewritten.append(name).append(" = ").append(initializer);
    if (!endsWith(rewritten, ";")) {
        rewritten.append(";");
    }
    return rewritten;
}

std::optional<std::string> rewriteTypedVariable(std::string_view text) {
    const std::size_t colon = text.find(':');
    if (colon == std::string_view::npos) {
        return std::nullopt;
    }
    const std::string name = trim_copy(text.substr(0, colon));
    std::string_view remainder = trim_view(text.substr(colon + 1));
    const std::size_t equals = remainder.find('=');
    std::string type = equals == std::string_view::npos
        ? trim_copy(remainder)
        : trim_copy(remainder.substr(0, equals));
    std::string initializer;
    if (equals != std::string_view::npos) {
        initializer = trim_copy(remainder.substr(equals + 1));
    }
    if (name.empty() || type.empty()) {
        return std::nullopt;
    }
    std::string rewritten = type + " " + name;
    if (!initializer.empty()) {
        rewritten.append(" = ").append(initializer);
    }
    if (!endsWith(rewritten, ";")) {
        rewritten.append(";");
    }
    return rewritten;
}

std::size_t findMatching(std::string_view text,
                         std::size_t openIndex,
                         char openChar,
                         char closeChar) {
    int depth = 0;
    for (std::size_t i = openIndex; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == openChar) {
            ++depth;
        } else if (ch == closeChar) {
            --depth;
            if (depth == 0) {
                return i;
            }
        }
    }
    return std::string_view::npos;
}

struct FunctionParts {
    std::string templatePrefix;
    std::string leading;
    std::string name;
    std::string params;
    std::string returnType;
    std::string trailing;
    std::string requiresClause;
    std::string body;
    bool hasExplicitReturnType = false;
};

std::optional<FunctionParts> parseFunction(std::string_view text,
                                           bool expectReturnType,
                                           std::string_view defaultReturnType = {}) {
    text = trim_view(text);
    if (text.empty()) {
        return std::nullopt;
    }

    std::string buffer(text);
    std::string_view working(buffer);
    working = trim_view(working);

    FunctionParts parts;

    if (startsWith(working, "template")) {
        const std::size_t lt = working.find('<');
        if (lt == std::string_view::npos) {
            return std::nullopt;
        }
        const std::size_t gt = findMatching(working, lt, '<', '>');
        if (gt == std::string_view::npos) {
            return std::nullopt;
        }
        const std::size_t prefixLen = gt + 1;
        parts.templatePrefix = trim_copy(working.substr(0, prefixLen));
        working.remove_prefix(prefixLen);
        working = trim_view(working);
    }

    const std::size_t colon = working.find(':');
    if (colon == std::string_view::npos) {
        return std::nullopt;
    }

    std::string_view beforeColon = trim_view(working.substr(0, colon));
    if (beforeColon.empty()) {
        return std::nullopt;
    }

    const std::size_t lastSpace = beforeColon.find_last_of(" \t\r\n");
    if (lastSpace == std::string_view::npos) {
        parts.name = trim_copy(beforeColon);
    } else {
        parts.name = trim_copy(beforeColon.substr(lastSpace + 1));
        parts.leading = trim_copy(beforeColon.substr(0, lastSpace));
    }
    if (parts.name.empty()) {
        return std::nullopt;
    }

    working.remove_prefix(colon + 1);
    working = trim_view(working);

    const std::size_t openParen = working.find('(');
    if (openParen == std::string_view::npos) {
        return std::nullopt;
    }
    const std::size_t closeParen = findMatching(working, openParen, '(', ')');
    if (closeParen == std::string_view::npos || closeParen <= openParen) {
        return std::nullopt;
    }

    parts.params = trim_copy(working.substr(openParen + 1, closeParen - openParen - 1));

    std::string_view remainder = working.substr(closeParen + 1);
    remainder = trim_view(remainder);

    const std::size_t equalsPos = remainder.find('=');
    if (equalsPos == std::string_view::npos) {
        return std::nullopt;
    }

    const std::size_t arrowPos = remainder.find("->");
    parts.hasExplicitReturnType = (arrowPos != std::string_view::npos && arrowPos < equalsPos);

    if (parts.hasExplicitReturnType) {
        std::string_view between = trim_view(remainder.substr(0, arrowPos));
        if (!between.empty()) {
            parts.trailing = trim_copy(between);
        }

        const std::size_t returnStart = arrowPos + 2;
        std::string_view returnPortion = trim_view(remainder.substr(returnStart, equalsPos - returnStart));

        const std::size_t requiresPos = returnPortion.find("requires");
        if (requiresPos != std::string_view::npos) {
            parts.requiresClause = trim_copy(returnPortion.substr(requiresPos));
            returnPortion = trim_view(returnPortion.substr(0, requiresPos));
        }

        parts.returnType = trim_copy(returnPortion);
        if (parts.returnType.empty() && expectReturnType) {
            return std::nullopt;
        }
    } else {
        if (expectReturnType) {
            return std::nullopt;
        }
        std::string_view suffix = trim_view(remainder.substr(0, equalsPos));
        if (!suffix.empty()) {
            const std::size_t requiresPos = suffix.find("requires");
            if (requiresPos != std::string_view::npos) {
                parts.requiresClause = trim_copy(suffix.substr(requiresPos));
                suffix = trim_view(suffix.substr(0, requiresPos));
            }
            parts.trailing = trim_copy(suffix);
        }
        parts.returnType = defaultReturnType.empty()
            ? std::string("void")
            : std::string(defaultReturnType);
    }

    if (parts.returnType.empty()) {
        if (!defaultReturnType.empty()) {
            parts.returnType = std::string(defaultReturnType);
        } else {
            parts.returnType = "void";
        }
    }

    std::string_view bodyView = remainder.substr(equalsPos + 1);
    parts.body = trim_copy(bodyView);
    return parts;
}

std::string ensureBodyWrapped(const std::string& body) {
    if (body.empty()) {
        return std::string("{ }");
    }
    std::string trimmed = trim_copy(body);
    if (!trimmed.empty() && trimmed.front() == '{') {
        return trimmed;
    }
    return "{ " + trimmed + " }";
}

struct Transformation {
    std::string text;
    std::string symbolName;
    bool isFunction = false;
    int priority = 0;
    std::string cutoutName;
    std::vector<int64_t> hierarchyPath;
    std::string hierarchyLabel;
    std::size_t snippetLength = 0;
};

int patternPriority(std::string_view name) {
    static const std::unordered_map<std::string, int> priorities = {
        {"cpp2_main_function", 250},
        {"cpp2_main_function_signature", 245},
        {"cpp2_function_definition", 220},
        {"cpp2_function_full_signature", 215},
        {"cpp2_function_trailing_return", 210},
        {"cpp2_function_with_qualifiers", 205},
        {"cpp2_function_with_return", 200},
        {"cpp2_parameter_detailed", 195},
        {"cpp2_function_simple_void", 192},
        {"cpp2_function_void", 190},
        {"cpp2_function_declaration_vs_definition", 180},
        {"cpp2_function_with_attributes", 175},
        {"cpp2_template_function", 170},
        {"cpp2_template_function_definition", 170},
        {"cpp2_function_with_attributes_signature", 170},
        {"cpp2_parameter", 120},
        {"cpp2_parameter_inout", 118},
        {"cpp2_parameter_out", 118},
        {"cpp2_parameter_copy", 118},
        {"cpp2_parameter_move", 118},
        {"cpp2_parameter_forward", 118},
        {"cpp2_var_auto", 100},
        {"cpp2_variable_auto_deduction", 100},
        {"cpp2_typed_variable", 80},
        {"cpp2_variable_explicit_type", 80}
    };
    auto it = priorities.find(std::string(name));
    if (it != priorities.end()) {
        return it->second;
    }
    return 0;
}

bool patternRequiresExplicitReturn(std::string_view name) {
    return name.find("with_return") != std::string_view::npos ||
           name.find("trailing_return") != std::string_view::npos ||
           name.find("definition") != std::string_view::npos ||
           name.find("full_signature") != std::string_view::npos ||
           name.find("template_function") != std::string_view::npos;
}

std::string_view patternDefaultReturn(std::string_view name) {
    if (name.find("main_function") != std::string_view::npos) {
        return "int";
    }
    if (name.find("void") != std::string_view::npos) {
        return "void";
    }
    return {};
}

std::optional<Transformation> transformCutout(const CutoutFragment& fragment) {
    if (fragment.patternName.empty()) {
        return std::nullopt;
    }
    Transformation result;
    result.priority = patternPriority(fragment.patternName);
    result.cutoutName = fragment.patternName;
    result.hierarchyPath = fragment.path;
    result.hierarchyLabel = fragment.hierarchyLabel;
    result.snippetLength = fragment.length;

    const bool expectExplicitReturn = patternRequiresExplicitReturn(fragment.patternName);
    const std::string_view defaultReturn = patternDefaultReturn(fragment.patternName);

    const bool handlesFunction =
        fragment.patternName == "cpp2_function_with_return" ||
        fragment.patternName == "cpp2_function_with_qualifiers" ||
        fragment.patternName == "cpp2_function_trailing_return" ||
        fragment.patternName == "cpp2_function_definition" ||
        fragment.patternName == "cpp2_function_with_attributes" ||
        fragment.patternName == "cpp2_function_declaration_vs_definition" ||
        fragment.patternName == "cpp2_template_function" ||
        fragment.patternName == "cpp2_template_function_definition" ||
        fragment.patternName == "cpp2_function_full_signature" ||
        fragment.patternName == "cpp2_function_simple_void" ||
        fragment.patternName == "cpp2_main_function_signature" ||
        fragment.patternName == "cpp2_main_function" ||
        fragment.patternName == "cpp2_function_void";

    if (handlesFunction) {
        auto parts = parseFunction(fragment.snippet,
                                   /*expectReturnType=*/expectExplicitReturn,
                                   defaultReturn);
        if (!parts || parts->returnType.empty()) {
            return std::nullopt;
        }

        std::vector<std::string> rewrittenParams;
        for (const auto& param : splitCommaBalanced(parts->params)) {
            if (param.empty()) {
                continue;
            }
            if (auto rewritten = rewriteParameter(param)) {
                rewrittenParams.push_back(*rewritten);
            } else {
                rewrittenParams.push_back(param);
            }
        }

        std::string paramsText = joinStrings(rewrittenParams, ", ");
        std::string bodyText = ensureBodyWrapped(parts->body);

        std::ostringstream oss;
        if (!parts->templatePrefix.empty()) {
            oss << parts->templatePrefix;
            if (parts->templatePrefix.back() != '\n') {
                oss << "\n";
            }
        }
        if (!parts->leading.empty()) {
            oss << parts->leading << " ";
        }
        oss << parts->returnType << " " << parts->name << "(" << paramsText << ")";
        if (!parts->trailing.empty()) {
            oss << " " << parts->trailing;
        }
        if (!parts->requiresClause.empty()) {
            oss << " " << parts->requiresClause;
        }
        oss << " " << bodyText;

        result.text = oss.str();
        result.symbolName = parts->name;
        result.isFunction = true;
        return result;
    }

    if (fragment.patternName == "cpp2_var_auto" ||
        fragment.patternName == "cpp2_variable_auto_deduction") {
        if (auto rewritten = rewriteVarAuto(fragment.snippet)) {
            result.text = *rewritten;
            result.symbolName.clear();
            result.isFunction = false;
            return result;
        }
        return std::nullopt;
    }

    if (fragment.patternName == "cpp2_typed_variable" ||
        fragment.patternName == "cpp2_variable_explicit_type") {
        if (auto rewritten = rewriteTypedVariable(fragment.snippet)) {
            result.text = *rewritten;
            result.symbolName.clear();
            result.isFunction = false;
            return result;
        }
        return std::nullopt;
    }

    if (fragment.patternName == "cpp2_parameter" ||
        fragment.patternName == "cpp2_parameter_detailed" ||
        fragment.patternName == "cpp2_parameter_inout" ||
        fragment.patternName == "cpp2_parameter_out" ||
        fragment.patternName == "cpp2_parameter_copy" ||
        fragment.patternName == "cpp2_parameter_move" ||
        fragment.patternName == "cpp2_parameter_forward") {
        if (auto rewritten = rewriteParameter(fragment.snippet)) {
            result.text = *rewritten;
            result.symbolName.clear();
            result.isFunction = false;
            return result;
        }
        return std::nullopt;
    }

    if (fragment.patternName == "cpp2_contract_pre") {
        const std::size_t colon = fragment.snippet.find(':');
        const std::size_t equals = find_assignment_equals(fragment.snippet, colon == std::string::npos ? 0 : colon + 1);
        if (colon == std::string::npos || equals == std::string::npos) {
            return std::nullopt;
        }
        std::string condition = strip_outer_parens(fragment.snippet.substr(colon + 1, equals - colon - 1));
        std::string message = ensure_string_literal(fragment.snippet.substr(equals + 1));
        std::ostringstream oss;
        oss << "assert((" << condition << ") && " << message << ");";
        result.text = oss.str();
        result.symbolName.clear();
        result.isFunction = false;
        return result;
    }

    if (fragment.patternName == "cpp2_contract_post") {
        const std::size_t colon = fragment.snippet.find(':');
        const std::size_t equals = find_assignment_equals(fragment.snippet, colon == std::string::npos ? 0 : colon + 1);
        if (colon == std::string::npos || equals == std::string::npos) {
            return std::nullopt;
        }
        std::string condition = strip_outer_parens(fragment.snippet.substr(colon + 1, equals - colon - 1));
        std::string_view remainder = trim_view(fragment.snippet.substr(equals + 1));
        const std::size_t requiresPos = remainder.find("requires");
        std::string message;
        std::string requiresClause;
        if (requiresPos != std::string_view::npos) {
            message = trim_copy(remainder.substr(0, requiresPos));
            requiresClause = trim_copy(remainder.substr(requiresPos + std::string_view("requires").size()));
        } else {
            message = trim_copy(remainder);
        }
        std::ostringstream oss;
        oss << "/* post: " << condition;
        if (!message.empty()) {
            oss << " -> " << message;
        }
        if (!requiresClause.empty()) {
            oss << " requires " << requiresClause;
        }
        oss << " */";
        result.text = oss.str();
        result.symbolName.clear();
        result.isFunction = false;
        return result;
    }

    if (fragment.patternName == "cpp2_ufcs_call") {
        std::string_view text = trim_view(fragment.snippet);
        const std::size_t dot = text.find('.');
        if (dot == std::string::npos) {
            return std::nullopt;
        }
        std::string object = trim_copy(text.substr(0, dot));
        std::string_view remainder = trim_view(text.substr(dot + 1));
        if (object.empty() || remainder.empty()) {
            return std::nullopt;
        }
        const std::size_t paren = remainder.find('(');
        std::string method;
        std::string args;
        std::string_view suffix;
        if (paren == std::string::npos) {
            method = trim_copy(remainder);
        } else {
            method = trim_copy(remainder.substr(0, paren));
            const std::size_t close = findMatching(remainder, paren, '(', ')');
            if (close == std::string::npos) {
                return std::nullopt;
            }
            args = trim_copy(remainder.substr(paren + 1, close - paren - 1));
            suffix = trim_view(remainder.substr(close + 1));
        }
        if (method.empty()) {
            return std::nullopt;
        }
        std::ostringstream oss;
        oss << method << "(" << object;
        if (!args.empty()) {
            oss << ", " << args;
        }
        oss << ")";
        if (!suffix.empty()) {
            oss << suffix;
        }
        result.text = oss.str();
        result.symbolName.clear();
        result.isFunction = false;
        return result;
    }

    if (fragment.patternName == "cpp2_for_with_body") {
        std::string_view text = trim_view(fragment.snippet);
        const std::size_t openBrace = text.find('{');
        const std::size_t closeBrace = openBrace == std::string::npos ? std::string::npos
                                                                      : findMatching(text, openBrace, '{', '}');
        if (openBrace == std::string::npos || closeBrace == std::string::npos) {
            return std::nullopt;
        }
        std::string prefix = trim_copy(text.substr(0, openBrace));
        std::string body = trim_copy(text.substr(openBrace + 1, closeBrace - openBrace - 1));
        if (startsWith(prefix, "for")) {
            prefix = trim_copy(prefix.substr(std::string_view("for").size()));
        }
        const std::size_t colon = prefix.find(':');
        if (colon == std::string::npos) {
            return std::nullopt;
        }
        std::string variable = trim_copy(prefix.substr(0, colon));
        std::string iterable = trim_copy(prefix.substr(colon + 1));
        if (variable.empty() || iterable.empty()) {
            return std::nullopt;
        }
        std::ostringstream oss;
        oss << "for (auto&& " << variable << " : " << iterable << ") { " << body << " }";
        result.text = oss.str();
        result.symbolName.clear();
        result.isFunction = false;
        return result;
    }

    if (fragment.patternName == "cpp2_while_loop") {
        std::string_view text = trim_view(fragment.snippet);
        const std::size_t keyword = text.find("while");
        const std::size_t openBrace = text.find('{', keyword == std::string::npos ? 0 : keyword);
        const std::size_t closeBrace = openBrace == std::string::npos ? std::string::npos
                                                                      : findMatching(text, openBrace, '{', '}');
        if (keyword == std::string::npos || openBrace == std::string::npos || closeBrace == std::string::npos) {
            return std::nullopt;
        }
        std::string condition = strip_outer_parens(text.substr(keyword + std::string_view("while").size(),
                                                               openBrace - (keyword + std::string_view("while").size())));
        std::string body = trim_copy(text.substr(openBrace + 1, closeBrace - openBrace - 1));
        std::ostringstream oss;
        oss << "while (" << condition << ") { " << body << " }";
        result.text = oss.str();
        result.symbolName.clear();
        result.isFunction = false;
        return result;
    }

    if (fragment.patternName == "cpp2_loop_forever") {
        std::string_view text = trim_view(fragment.snippet);
        const std::size_t openBrace = text.find('{');
        const std::size_t closeBrace = openBrace == std::string::npos ? std::string::npos
                                                                      : findMatching(text, openBrace, '{', '}');
        if (openBrace == std::string::npos || closeBrace == std::string::npos) {
            return std::nullopt;
        }
        std::string body = trim_copy(text.substr(openBrace + 1, closeBrace - openBrace - 1));
        std::ostringstream oss;
        oss << "while (true) { " << body << " }";
        result.text = oss.str();
        result.symbolName.clear();
        result.isFunction = false;
        return result;
    }

    if (fragment.patternName == "cpp2_inspect_expression") {
        std::string_view text = trim_view(fragment.snippet);
        const std::size_t keyword = text.find("inspect");
        const std::size_t openBrace = text.find('{', keyword == std::string::npos ? 0 : keyword);
        const std::size_t closeBrace = openBrace == std::string::npos ? std::string::npos
                                                                      : findMatching(text, openBrace, '{', '}');
        if (keyword == std::string::npos || openBrace == std::string::npos || closeBrace == std::string::npos) {
            return std::nullopt;
        }
        std::string subject = trim_copy(text.substr(keyword + std::string_view("inspect").size(),
                                                    openBrace - (keyword + std::string_view("inspect").size())));
        std::string cases = trim_copy(text.substr(openBrace + 1, closeBrace - openBrace - 1));
        std::ostringstream oss;
        oss << "/* inspect */\n";
        if (!subject.empty()) {
            oss << "auto&& __inspect_value = " << subject << ";\n";
            oss << "(void)__inspect_value;\n";
        }
        if (!cases.empty()) {
            oss << "// cases:\n" << cases << "\n";
        }
        result.text = oss.str();
        result.symbolName.clear();
        result.isFunction = false;
        return result;
    }

    if (fragment.patternName == "cpp2_inspect_case_is") {
        std::string content = trim_copy(fragment.snippet);
        std::ostringstream oss;
        oss << "// inspect case: " << content;
        result.text = oss.str();
        result.symbolName.clear();
        result.isFunction = false;
        return result;
    }

    return std::nullopt;
}

class CppFortFuncRewrite : public mlir::RewritePattern {
public:
    CppFortFuncRewrite(mlir::MLIRContext* context,
                       std::string* outText,
                       std::string* outName,
                       bool* applied)
        : mlir::RewritePattern("cppfort.func", /*benefit=*/1, context),
          outText_(outText),
          outName_(outName),
          applied_(applied) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                        mlir::PatternRewriter& rewriter) const override {
        if (!outText_ || !outName_ || !applied_ || *applied_) {
            return mlir::failure();
        }

        auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name");
        auto paramsAttr = op->getAttrOfType<mlir::StringAttr>("params");
        auto returnAttr = op->getAttrOfType<mlir::StringAttr>("return_type");
        auto bodyAttr = op->getAttrOfType<mlir::StringAttr>("body");

        if (!nameAttr || !bodyAttr) {
            return mlir::failure();
        }

        const std::string funcName = nameAttr.getValue().str();
        const std::string params = paramsAttr ? paramsAttr.getValue().str() : std::string{};
        std::string result = returnAttr && !returnAttr.getValue().empty()
            ? returnAttr.getValue().str() + " "
            : std::string("void ");
        result += funcName;
        result += "(" + params + ") { " + bodyAttr.getValue().str() + " }";

        rewriter.setInsertionPoint(op);
        auto funcType = rewriter.getFunctionType({}, {});
        auto newFunc = rewriter.create<mlir::func::FuncOp>(op->getLoc(), funcName, funcType);
        auto* block = newFunc.addEntryBlock();
        mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockEnd(block);
        bodyBuilder.create<mlir::func::ReturnOp>(op->getLoc());

        newFunc->setAttr("cppfort.cpp1_text", rewriter.getStringAttr(result));
        rewriter.eraseOp(op);

        *outText_ = result;
        *outName_ = funcName;
        *applied_ = true;
        return mlir::success();
    }

private:
    std::string* outText_;
    std::string* outName_;
    bool* applied_;
};

bool extractHierarchyPath(mlir::DictionaryAttr dict, std::vector<int64_t>& outPath) {
    if (!dict) {
        return false;
    }
    auto pathAttribute = dict.get("path");
    if (!pathAttribute) {
        return false;
    }
    auto pathAttr = llvm::dyn_cast<mlir::ArrayAttr>(pathAttribute);
    if (!pathAttr) {
        return false;
    }
    outPath.clear();
    outPath.reserve(pathAttr.size());
    for (const auto& element : pathAttr) {
        const auto integerAttr = llvm::dyn_cast<mlir::IntegerAttr>(element);
        if (!integerAttr) {
            return false;
        }
        outPath.push_back(integerAttr.getInt());
    }
    return true;
}

std::string extractHierarchyLabel(mlir::DictionaryAttr dict) {
    if (!dict) {
        return {};
    }
    if (auto labelAttr = dict.get("label")) {
        if (auto stringAttr = llvm::dyn_cast<mlir::StringAttr>(labelAttr)) {
            return stringAttr.getValue().str();
        }
    }
    return {};
}

class Cpp2FunctionRewrite : public mlir::OpRewritePattern<mlir::func::FuncOp> {
public:
    Cpp2FunctionRewrite(mlir::MLIRContext* context,
                        std::string* outText,
                        std::string* outName,
                        bool* applied)
        : mlir::OpRewritePattern<mlir::func::FuncOp>(context),
          outText_(outText),
          outName_(outName),
          applied_(applied) {}

    mlir::LogicalResult matchAndRewrite(mlir::func::FuncOp funcOp,
                                        mlir::PatternRewriter& rewriter) const final {
        if (!outText_ || !outName_ || !applied_ || *applied_) {
            return mlir::failure();
        }

        if (!funcOp->hasAttr("cppfort.cutout_count")) {
            return mlir::failure();
        }

        llvm::SmallVector<CutoutFragment, 8> fragments;
        funcOp.walk([&](mlir::Operation* op) {
            if (op->getName().getStringRef() != "cppfort.cutout") {
                return;
            }
            auto spanAttr = op->getAttrOfType<mlir::DictionaryAttr>("source_span");
            if (!spanAttr) {
                return;
            }
            auto snippetAttr = spanAttr.get("snippet");
            if (!snippetAttr) {
                return;
            }
            auto snippetStringAttr = llvm::dyn_cast<mlir::StringAttr>(snippetAttr);
            if (!snippetStringAttr) {
                return;
            }

            auto nameAttr = op->getAttrOfType<mlir::StringAttr>("cutout_name");
            auto hierarchyAttr = op->getAttrOfType<mlir::DictionaryAttr>("hierarchy_index");
            CutoutFragment fragment;
            if (!extractHierarchyPath(hierarchyAttr, fragment.path)) {
                return;
            }
            if (nameAttr) {
                fragment.patternName = nameAttr.getValue().str();
            }
            fragment.hierarchyLabel = extractHierarchyLabel(hierarchyAttr);
            fragment.snippet = snippetStringAttr.getValue().str();
            fragment.length = fragment.snippet.size();
            fragments.push_back(std::move(fragment));
        });

        if (fragments.empty()) {
            return mlir::failure();
        }

        std::vector<Transformation> collected;
        collected.reserve(fragments.size());
        for (const auto& fragment : fragments) {
            if (auto candidate = transformCutout(fragment)) {
                collected.push_back(std::move(*candidate));
            }
        }

        const Transformation* bestTransform = nullptr;
        for (const auto& candidate : collected) {
            if (!candidate.isFunction || candidate.text.empty() || candidate.symbolName.empty()) {
                continue;
            }
            if (!bestTransform ||
                candidate.priority > bestTransform->priority ||
                (candidate.priority == bestTransform->priority &&
                 candidate.snippetLength > bestTransform->snippetLength)) {
                bestTransform = &candidate;
            }
        }

        std::string replacementName;
        std::string finalText;

        if (bestTransform) {
            const std::string baseName = bestTransform->symbolName.empty()
                ? funcOp.getSymName().str()
                : bestTransform->symbolName;
            const std::string sanitizedBase = sanitize_symbol(baseName);
            replacementName = "cpp1_" + sanitizedBase;
            finalText = bestTransform->text;
        } else {
            const auto comparator = [](const CutoutFragment& lhs, const CutoutFragment& rhs) {
                return lhs.path < rhs.path;
            };
            std::sort(fragments.begin(), fragments.end(), comparator);

            const auto bestFragment = std::max_element(
                fragments.begin(), fragments.end(),
                [](const CutoutFragment& lhs, const CutoutFragment& rhs) {
                    return lhs.length < rhs.length;
                });

            if (bestFragment == fragments.end() || bestFragment->snippet.empty()) {
                return mlir::failure();
            }

            replacementName = "cpp1_" + funcOp.getSymName().str();
            finalText = bestFragment->snippet;
        }

        rewriter.setInsertionPoint(funcOp);
        auto newFunc = rewriter.create<mlir::func::FuncOp>(
            funcOp.getLoc(),
            replacementName,
            rewriter.getFunctionType({}, {}));
        auto* block = newFunc.addEntryBlock();
        mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockEnd(block);
        bodyBuilder.create<mlir::func::ReturnOp>(funcOp.getLoc());

        newFunc->setAttr("cppfort.cpp1_text",
                         rewriter.getStringAttr(finalText));

        if (!collected.empty()) {
            mlir::OpBuilder attrBuilder(funcOp.getContext());
            llvm::SmallVector<mlir::Attribute, 8> rewriteEntries;
            rewriteEntries.reserve(collected.size());
            for (const auto& transform : collected) {
                if (transform.text.empty() || transform.cutoutName.empty()) {
                    continue;
                }
                llvm::SmallVector<mlir::NamedAttribute, 4> fields;
                fields.emplace_back(attrBuilder.getStringAttr("cutout"),
                                    attrBuilder.getStringAttr(transform.cutoutName));
                fields.emplace_back(attrBuilder.getStringAttr("text"),
                                    attrBuilder.getStringAttr(transform.text));
                if (!transform.hierarchyPath.empty()) {
                    llvm::SmallVector<mlir::Attribute, 8> pathAttrs;
                    for (auto index : transform.hierarchyPath) {
                        pathAttrs.push_back(attrBuilder.getI64IntegerAttr(index));
                    }
                    fields.emplace_back(attrBuilder.getStringAttr("path"),
                                        attrBuilder.getArrayAttr(pathAttrs));
                }
                if (!transform.hierarchyLabel.empty()) {
                    fields.emplace_back(attrBuilder.getStringAttr("label"),
                                        attrBuilder.getStringAttr(transform.hierarchyLabel));
                }
                rewriteEntries.push_back(attrBuilder.getDictionaryAttr(fields));
            }
            if (!rewriteEntries.empty()) {
                newFunc->setAttr("cppfort.rewritten_cutouts",
                                 attrBuilder.getArrayAttr(rewriteEntries));
            }
        }

        rewriter.eraseOp(funcOp);

        if (outText_) {
            *outText_ = finalText;
        }
        if (outName_) {
            *outName_ = replacementName;
        }
        if (applied_) {
            *applied_ = true;
        }
        return mlir::success();
    }

private:
    std::string* outText_;
    std::string* outName_;
    bool* applied_;
};

} // namespace

bool rewrite_first_function_to_cpp1(mlir::ModuleOp module,
                                    std::string& generated_cpp1,
                                    std::string& rewritten_function_name) {
    bool applied = false;
    mlir::RewritePatternSet patterns(module.getContext());
    patterns.add<CppFortFuncRewrite>(
        module.getContext(), &generated_cpp1, &rewritten_function_name, &applied);
    patterns.add<Cpp2FunctionRewrite>(
        module.getContext(), &generated_cpp1, &rewritten_function_name, &applied);

    if (mlir::failed(mlir::applyPatternsGreedily(
            module, std::move(patterns)))) {
        return false;
    }
    return applied;
}

std::vector<CutoutRewrite> collect_cutout_rewrites(mlir::ModuleOp module) {
    std::vector<CutoutRewrite> rewrites;
    module.walk([&](mlir::func::FuncOp func) {
        auto attr = func->getAttrOfType<mlir::ArrayAttr>("cppfort.rewritten_cutouts");
        if (!attr) {
            return;
        }
        for (const auto& entry : attr) {
            auto dict = llvm::dyn_cast<mlir::DictionaryAttr>(entry);
            if (!dict) {
                continue;
            }
            auto cutoutAttr = dict.get("cutout");
            auto textAttr = dict.get("text");
            if (!cutoutAttr || !textAttr) {
                continue;
            }
            auto cutoutString = llvm::dyn_cast<mlir::StringAttr>(cutoutAttr);
            auto textString = llvm::dyn_cast<mlir::StringAttr>(textAttr);
            if (!cutoutString || !textString) {
                continue;
            }

            CutoutRewrite rewrite;
            rewrite.cutoutName = cutoutString.getValue().str();
            rewrite.text = textString.getValue().str();

            if (auto pathAttrValue = dict.get("path")) {
                if (auto pathArray = llvm::dyn_cast<mlir::ArrayAttr>(pathAttrValue)) {
                    rewrite.hierarchyPath.reserve(pathArray.size());
                    for (const auto& value : pathArray) {
                        if (auto integerAttr = llvm::dyn_cast<mlir::IntegerAttr>(value)) {
                            rewrite.hierarchyPath.push_back(integerAttr.getInt());
                        }
                    }
                }
            }
            if (auto labelAttrValue = dict.get("label")) {
                if (auto labelAttr = llvm::dyn_cast<mlir::StringAttr>(labelAttrValue)) {
                    rewrite.hierarchyLabel = labelAttr.getValue().str();
                }
            }

            rewrites.push_back(std::move(rewrite));
        }
    });
    return rewrites;
}

} // namespace cppfort::stage0

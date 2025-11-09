#include "pijul_parameter_graph.h"

#include <utility>
#include <cctype>
#include <unordered_set>

#include "pijul_signature_rules.h"

namespace {

bool is_identifier_start(char ch) {
    return std::isalpha(static_cast<unsigned char>(ch)) || ch == '_';
}

bool is_identifier_char(char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
}

std::string generalize_token(const std::string& token) {
    if (token.empty()) {
        return "";
    }
    if (token == "::" || token == "->" || token == "=>" || token == "==") {
        return "OP:" + token;
    }
    if (token == "{" || token == "}" || token == "(" || token == ")" ||
        token == "[" || token == "]" || token == ";" || token == ":") {
        return "DELIM:" + token;
    }
    if (token.size() >= 2 && (token.front() == '"' || token.front() == '\'') ) {
        return "STRING";
    }
    if (std::isdigit(static_cast<unsigned char>(token.front()))) {
        return "NUMBER";
    }
    static const std::unordered_set<std::string> keywords = {
        "if","else","while","for","return","auto","concept","requires",
        "inspect","co_await","co_return","co_yield","fn","let","var"};
    if (keywords.count(token) != 0) {
        return "KW:" + token;
    }
    if (is_identifier_start(token.front())) {
        return "IDENT";
    }
    return token;
}

std::vector<std::string> tokenize_generalize(std::string_view text) {
    std::vector<std::string> result;
    std::string current;
    for (std::size_t i = 0; i < text.size(); ++i) {
        char ch = text[i];
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                result.push_back(generalize_token(current));
                current.clear();
            }
            continue;
        }
        if (!is_identifier_char(ch)) {
            if (!current.empty()) {
                result.push_back(generalize_token(current));
                current.clear();
            }
            std::string single(1, ch);
            result.push_back(generalize_token(single));
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        result.push_back(generalize_token(current));
    }
    return result;
}

std::string extract_fragment(std::string_view text, const cppfort::pijul::NodeContext& ctx) {
    if (ctx.end_pos <= ctx.start_pos || ctx.end_pos > text.size()) {
        return {};
    }
    return std::string(text.substr(ctx.start_pos, ctx.end_pos - ctx.start_pos));
}

} // namespace

namespace cppfort::pijul {

const ParameterAnchor& ParameterGraph::add_anchor(const ParameterAnchor& anchor) {
    auto it = m_index.find(anchor.signature);
    if (it != m_index.end()) {
        return m_anchors[it->second];
    }

    m_anchors.push_back(anchor);
    std::size_t index = m_anchors.size() - 1;
    m_index.emplace(anchor.signature, index);
    return m_anchors.back();
}

void ParameterGraph::add_edge(const std::string& from,
                              const std::string& to,
                              const std::string& reason) {
    m_edges.push_back(ParameterEdge{from, to, reason});
}

std::optional<ParameterAnchor> ParameterGraph::find(const std::string& signature) const {
    auto it = m_index.find(signature);
    if (it == m_index.end()) {
        return std::nullopt;
    }
    return m_anchors[it->second];
}

ParameterAnchor make_anchor(const OrbitMatchInfo& match,
                            std::string_view source_fragment,
                            std::string_view transformed_fragment,
                            const std::unordered_map<std::string, std::string>& params,
                            const std::string& description) {
    ParameterAnchor anchor;
    anchor.signature = match.key;
    anchor.pattern = match.patternName;
    anchor.depth = match.context.depth_hint;
    anchor.parameters = params;
    anchor.context = match.context;
    anchor.description = description;
    anchor.source_fragment = std::string(source_fragment);
    anchor.transformed_fragment = std::string(transformed_fragment);
    anchor.source_tokens = tokenize_generalize(anchor.source_fragment);
    anchor.transformed_tokens = tokenize_generalize(anchor.transformed_fragment);
    return anchor;
}

void populate_parameter_graph(ParameterGraph& graph,
                              const OrbitMatchCollection& source,
                              const OrbitMatchCollection& transformed,
                              const std::string& source_code,
                              const std::string& transformed_code) {
    std::unordered_map<std::string, std::string> empty;

    for (const auto& [key, info] : source.byKey) {
        auto descriptor = describe_signature(info);
        std::string fragment = extract_fragment(source_code, info.context);
        graph.add_anchor(make_anchor(info, fragment, std::string_view{}, empty, descriptor));
    }
    for (const auto& [key, info] : transformed.byKey) {
        auto descriptor = describe_signature(info);
        std::string fragment = extract_fragment(transformed_code, info.context);
        graph.add_anchor(make_anchor(info, std::string_view{}, fragment, empty, descriptor));
    }

    for (const auto& [key, info] : transformed.byKey) {
        auto it = source.byPattern.find(info.patternName);
        if (it == source.byPattern.end()) {
            continue;
        }
        for (const auto& candidate : it->second) {
            graph.add_edge(candidate, key, "pattern-match");
        }
    }
}

} // namespace cppfort::pijul

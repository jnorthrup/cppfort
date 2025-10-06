#include "correlator.h"

#include <algorithm>
#include <cctype>
#include <string_view>

namespace cppfort::stage0 {
namespace {

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

} // namespace

std::string_view FragmentCorrelator::primary_view(const OrbitFragment& fragment) {
    if (!fragment.cpp2_text.empty()) return fragment.cpp2_text;
    if (!fragment.cpp_text.empty()) return fragment.cpp_text;
    if (!fragment.c_text.empty()) return fragment.c_text;
    return {};
}

bool FragmentCorrelator::contains_token(const std::string& text, const std::string& token) {
    auto pos = text.find(token);
    if (pos == std::string::npos) {
        return false;
    }
    const auto is_ident = [](char ch) {
        return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
    };
    bool prefix_ok = (pos == 0) || !is_ident(text[pos - 1]);
    bool suffix_ok = (pos + token.size() >= text.size()) || !is_ident(text[pos + token.size()]);
    return prefix_ok && suffix_ok;
}

EvidenceGrammarKind FragmentCorrelator::classify_text(const std::string& text) const {
    return classify(text);
}

EvidenceGrammarKind FragmentCorrelator::classify(const std::string& text) const {
    if (text.empty()) {
        return EvidenceGrammarKind::Unknown;
    }

    TypeEvidence evidence;
    evidence.ingest(text);
    EvidenceGrammarKind kind = evidence.deduce();
    if (kind != EvidenceGrammarKind::Unknown) {
        return kind;
    }

    std::string trimmed = trim_copy(text);
    if (is_cpp2_syntax(trimmed)) return EvidenceGrammarKind::CPP2;
    if (is_cpp_syntax(trimmed)) return EvidenceGrammarKind::CPP;
    if (is_c_syntax(trimmed)) return EvidenceGrammarKind::C;
    return EvidenceGrammarKind::Unknown;
}

bool FragmentCorrelator::is_cpp2_syntax(const std::string& text) const {
    if (text.find(": (") != std::string::npos || text.find(":(") != std::string::npos) {
        return true;
    }
    if (text.find("->") != std::string::npos && contains_token(text, "inspect")) {
        return true;
    }
    return contains_token(text, "inspect") || contains_token(text, "contract") || contains_token(text, "let");
}

bool FragmentCorrelator::is_cpp_syntax(const std::string& text) const {
    return contains_token(text, "template") ||
           contains_token(text, "namespace") ||
           text.find("::") != std::string::npos ||
           text.find("[](") != std::string::npos ||
           contains_token(text, "class");
}

bool FragmentCorrelator::is_c_syntax(const std::string& text) const {
    if (contains_token(text, "typedef") || contains_token(text, "struct")) {
        if (!contains_token(text, "class") && text.find("::") == std::string::npos) {
            return true;
        }
    }
    return contains_token(text, "#include") || contains_token(text, "extern");
}

void FragmentCorrelator::correlate(OrbitFragment& fragment) const {
    std::string_view primary = primary_view(fragment);
    if (primary.empty()) {
        return;
    }

    std::string primary_str(primary);
    EvidenceGrammarKind kind = classify(primary_str);

    auto ensure = [&](std::string& slot) {
        if (slot.empty()) {
            slot = primary_str;
        }
    };

    switch (kind) {
        case EvidenceGrammarKind::C:
            fragment.confidence = std::max(fragment.confidence, 0.6);
            ensure(fragment.c_text);
            break;
        case EvidenceGrammarKind::CPP:
            fragment.confidence = std::max(fragment.confidence, 0.8);
            ensure(fragment.cpp_text);
            break;
        case EvidenceGrammarKind::CPP2:
            fragment.confidence = std::max(fragment.confidence, 0.9);
            ensure(fragment.cpp2_text);
            break;
        case EvidenceGrammarKind::Unknown:
        default:
            fragment.confidence = std::max(fragment.confidence, 0.25);
            break;
    }

    // Provide fallbacks so downstream consumers always have content
    ensure(fragment.c_text);
    ensure(fragment.cpp_text);
    ensure(fragment.cpp2_text);
}

} // namespace cppfort::stage0


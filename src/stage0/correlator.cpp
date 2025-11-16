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

std::string_view FragmentCorrelator::extract_view(const OrbitFragment& fragment, std::string_view source) {
    if (fragment.start_pos >= source.size() || fragment.end_pos > source.size() || fragment.start_pos >= fragment.end_pos) {
        return {};
    }
    return source.substr(fragment.start_pos, fragment.end_pos - fragment.start_pos);
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

void FragmentCorrelator::correlate(OrbitFragment& fragment, std::string_view source) const {
    std::string_view view = extract_view(fragment, source);
    if (view.empty()) {
        return;
    }

    std::string text(view);
    TypeEvidence evidence;
    evidence.ingest(text);
    EvidenceGrammarKind kind = evidence.deduce();

    // Compute confidence from evidence scores (honest baseline)
    double computed_confidence = 0.0;
    if (evidence.total_tokens > 0) {
        uint32_t max_score = 0;
        if (kind == EvidenceGrammarKind::C) {
            max_score = evidence.c_keywords * 4 + evidence.typedef_hits * 5 + evidence.struct_hits * 4 + evidence.pointer_indicators * 2;
        } else if (kind == EvidenceGrammarKind::CPP) {
            max_score = evidence.cpp_keywords * 3 + evidence.template_ids * 5 + evidence.namespace_hits * 4 + evidence.double_colon * 5 + evidence.lambda_captures * 3 + evidence.concept_hits * 5 + evidence.requires_hits * 4 + evidence.arrow + evidence.angle_open;
        } else if (kind == EvidenceGrammarKind::CPP2) {
            max_score = evidence.cpp2_keywords * 4 + evidence.cpp2_signature_hits * 6 + evidence.inspect_hits * 5 + evidence.contract_keywords * 5 + (evidence.is_keyword_hits + evidence.as_keyword_hits) * 3 + evidence.flow_keywords * 2 + evidence.arrow;
        }
        computed_confidence = static_cast<double>(max_score) / evidence.total_tokens;
        computed_confidence = std::min(computed_confidence, 1.0); // Cap at 1.0
    }

    fragment.confidence = computed_confidence;
    switch (kind) {
        case EvidenceGrammarKind::C:
            fragment.classified_grammar = ::cppfort::ir::GrammarType::C;
            break;
        case EvidenceGrammarKind::CPP:
            fragment.classified_grammar = ::cppfort::ir::GrammarType::CPP;
            break;
        case EvidenceGrammarKind::CPP2:
            fragment.classified_grammar = ::cppfort::ir::GrammarType::CPP2;
            break;
        case EvidenceGrammarKind::Unknown:
        default:
            fragment.classified_grammar = ::cppfort::ir::GrammarType::UNKNOWN;
            break;
    }
}

} // namespace cppfort::stage0


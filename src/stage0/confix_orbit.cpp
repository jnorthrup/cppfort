#include "confix_orbit.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "function_orbit.h"
#include "correlator.h"
#include "rbcursive.h"

namespace cppfort::stage0 {

namespace {
constexpr bool is_anchor_char(char ch) {
    switch (ch) {
        case '{': case '}':
        case '(': case ')':
        case '[': case ']':
        case '<': case '>':
        case ';': case ',':
        case ':':
            return true;
        default:
            return false;
    }
}
} // namespace

ConfixOrbit::ConfixOrbit(char open_char, char close_char)
    : Orbit(OrbitType::Confix), open_char_(open_char), close_char_(close_char) {}

bool ConfixOrbit::matches(const EvidenceSpan& span) const {
    if (span.content.empty()) {
        return false;
    }

    const auto first = span.content.front();
    const auto last = span.content.back();

    if (!validate_pair(first, last)) {
        return false;
    }

    int balance = 0;
    for (char ch : span.content) {
        if (ch == open_char_) {
            ++balance;
        } else if (ch == close_char_) {
            --balance;
            if (balance < 0) {
                return false;
            }
        }
    }

    return balance == 0;
}

bool ConfixOrbit::validate_pair(char open, char close) const {
    return open == open_char_ && close == close_char_;
}

void ConfixOrbit::parameterize_children(const PatternData& pattern) {
    Orbit::parameterize_children(pattern);

    auto* c_child = ensure_function_child(::cppfort::ir::GrammarType::C, std::make_unique<C_FunctionOrbit>());
    auto* cpp_child = ensure_function_child(::cppfort::ir::GrammarType::CPP, std::make_unique<CPP_FunctionOrbit>());
    auto* cpp2_child = ensure_function_child(::cppfort::ir::GrammarType::CPP2, std::make_unique<CPP2_FunctionOrbit>());

    const auto assign_pattern = [&](FunctionOrbit* child, ::cppfort::ir::GrammarType grammar, const std::string& fallback) {
        if (!child) {
            return;
        }
        auto it = pattern.grammar_variants.find(grammar);
        if (it != pattern.grammar_variants.end()) {
            child->set_pattern(it->second);
        } else {
            child->set_pattern(fallback);
        }
    };

    assign_pattern(c_child, ::cppfort::ir::GrammarType::C, "void %s()");
    assign_pattern(cpp_child, ::cppfort::ir::GrammarType::CPP, "auto %s() -> %s");
    assign_pattern(cpp2_child, ::cppfort::ir::GrammarType::CPP2, "%s: () -> %s");

    std::vector<std::pair<FunctionOrbit*, ::cppfort::ir::GrammarType>> children;
    children.reserve(3);
    if (c_child) children.emplace_back(c_child, ::cppfort::ir::GrammarType::C);
    if (cpp_child) children.emplace_back(cpp_child, ::cppfort::ir::GrammarType::CPP);
    if (cpp2_child) children.emplace_back(cpp2_child, ::cppfort::ir::GrammarType::CPP2);

    for (auto& entry : children) {
        auto* child = entry.first;
        child->clear_evidence();
        child->start_pos = start_pos;
        child->end_pos = end_pos;
        child->confidence = 0.0;
    }

    std::string aggregate_text;
    aggregate_text.reserve(256);
    for (size_t idx = 0; idx < evidence_count(); ++idx) {
        const EvidenceSpan* span = get_evidence(idx);
        if (span) {
            aggregate_text.append(span->content);
            aggregate_text.push_back('\n');
        }
    }

    FragmentCorrelator correlator;
    EvidenceGrammarKind aggregate_kind = aggregate_text.empty()
        ? EvidenceGrammarKind::Unknown
        : correlator.classify_text(aggregate_text);

    auto infer_grammar = [](const std::string& category) {
        std::string lower = category;
        for (auto& ch : lower) { ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch))); }
        if (lower.find("cpp2") != std::string::npos) return ::cppfort::ir::GrammarType::CPP2;
        if (lower.find("cpp") != std::string::npos) return ::cppfort::ir::GrammarType::CPP;
        if (lower.find("c") != std::string::npos) return ::cppfort::ir::GrammarType::C;
        return ::cppfort::ir::GrammarType::UNKNOWN;
    };

    auto matches_kind = [&](::cppfort::ir::GrammarType grammar) {
        switch (aggregate_kind) {
            case EvidenceGrammarKind::C: return grammar == ::cppfort::ir::GrammarType::C;
            case EvidenceGrammarKind::CPP: return grammar == ::cppfort::ir::GrammarType::CPP;
            case EvidenceGrammarKind::CPP2: return grammar == ::cppfort::ir::GrammarType::CPP2;
            default: return false;
        }
    };

    FunctionOrbit* best_child = nullptr;
    double best_confidence = -1.0;
    selected_grammar_ = ::cppfort::ir::GrammarType::UNKNOWN;
    winning_child_ = nullptr;

    for (size_t idx = 0; idx < evidence_count(); ++idx) {
        const EvidenceSpan* span = get_evidence(idx);
        if (!span) {
            continue;
        }

        for (auto& entry : children) {
            auto* child = entry.first;
            auto grammar = entry.second;
            double score = child->evaluate(*span);
            if (score <= 0.0 && matches_kind(grammar)) {
                score = 0.6;
            }
            if (score > child->confidence) {
                child->confidence = score;
                child->start_pos = span->start_pos;
                child->end_pos = span->end_pos;
            }
            if (child->confidence > best_confidence) {
                best_confidence = child->confidence;
                best_child = child;
                selected_grammar_ = grammar;
            }
        }
    }

    auto pattern_grammar = infer_grammar(pattern.category);
    if (pattern_grammar != ::cppfort::ir::GrammarType::UNKNOWN) {
        FunctionOrbit* preferred = nullptr;
        for (auto& entry : children) {
            if (entry.second == pattern_grammar) {
                preferred = entry.first;
                break;
            }
        }
        if (preferred) {
            double hint_conf = matches_kind(pattern_grammar) ? 0.85 : 0.65;
            if (hint_conf > preferred->confidence) {
                preferred->confidence = hint_conf;
                preferred->start_pos = start_pos;
                preferred->end_pos = end_pos;
            }
            if (preferred->confidence > best_confidence) {
                best_confidence = preferred->confidence;
                best_child = preferred;
                selected_grammar_ = pattern_grammar;
            }
        }
    }

    if ((!best_child || best_confidence <= 0.0) && !children.empty()) {
        ::cppfort::ir::GrammarType fallback_grammar = pattern_grammar;
        if (fallback_grammar == ::cppfort::ir::GrammarType::UNKNOWN) {
            switch (aggregate_kind) {
                case EvidenceGrammarKind::CPP2: fallback_grammar = ::cppfort::ir::GrammarType::CPP2; break;
                case EvidenceGrammarKind::CPP: fallback_grammar = ::cppfort::ir::GrammarType::CPP; break;
                case EvidenceGrammarKind::C: fallback_grammar = ::cppfort::ir::GrammarType::C; break;
                case EvidenceGrammarKind::Unknown: default: fallback_grammar = ::cppfort::ir::GrammarType::C; break;
            }
        } else if (!matches_kind(fallback_grammar) && aggregate_kind != EvidenceGrammarKind::Unknown) {
            switch (aggregate_kind) {
                case EvidenceGrammarKind::CPP2: fallback_grammar = ::cppfort::ir::GrammarType::CPP2; break;
                case EvidenceGrammarKind::CPP: fallback_grammar = ::cppfort::ir::GrammarType::CPP; break;
                case EvidenceGrammarKind::C: fallback_grammar = ::cppfort::ir::GrammarType::C; break;
                case EvidenceGrammarKind::Unknown: default: break;
            }
        }

        FunctionOrbit* fallback_child = nullptr;
        for (auto& entry : children) {
            if (entry.second == fallback_grammar) {
                fallback_child = entry.first;
                break;
            }
        }
        if (!fallback_child) {
            fallback_child = children.front().first;
            fallback_grammar = children.front().second;
        }

        double fallback_conf = aggregate_text.empty() ? 0.4 : 0.75;
        fallback_child->confidence = std::max(fallback_child->confidence, fallback_conf);
        fallback_child->start_pos = start_pos;
        fallback_child->end_pos = end_pos;
        best_child = fallback_child;
        best_confidence = fallback_child->confidence;
        selected_grammar_ = fallback_grammar;
    }

    if (best_child) {
        winning_child_ = best_child;
        confidence = best_child->confidence;
        start_pos = best_child->start_pos;
        end_pos = best_child->end_pos;
    } else {
        confidence = aggregate_text.empty() ? 0.0 : 0.3;
        winning_child_ = nullptr;
        if (aggregate_kind == EvidenceGrammarKind::CPP2) {
            selected_grammar_ = ::cppfort::ir::GrammarType::CPP2;
        } else if (aggregate_kind == EvidenceGrammarKind::CPP) {
            selected_grammar_ = ::cppfort::ir::GrammarType::CPP;
        } else if (aggregate_kind == EvidenceGrammarKind::C) {
            selected_grammar_ = ::cppfort::ir::GrammarType::C;
        } else {
            selected_grammar_ = ::cppfort::ir::GrammarType::UNKNOWN;
        }
    }

    if (auto* confix_scanner = get_combinator()) {
        (void)confix_scanner->patternCapabilities();
    }
}

void ConfixOrbit::extract_evidence(std::string_view text, size_t start, size_t end) {
    if (start >= text.size() || start >= end) {
        return;
    }

    end = std::min(end, text.size());
    size_t run_start = std::string_view::npos;

    for (size_t idx = start; idx < end; ++idx) {
        char ch = text[idx];
        if (is_anchor_char(ch)) {
            if (run_start != std::string_view::npos) {
                const auto length = idx - run_start;
                add_evidence(EvidenceSpan{run_start, idx, std::string{text.substr(run_start, length)}, 0.5});
                run_start = std::string_view::npos;
            }
            continue;
        }

        if (run_start == std::string_view::npos) {
            run_start = idx;
        }
    }

    if (run_start != std::string_view::npos) {
        add_evidence(EvidenceSpan{run_start, end, std::string{text.substr(run_start, end - run_start)}, 0.5});
    }
}

FunctionOrbit* ConfixOrbit::ensure_function_child(::cppfort::ir::GrammarType grammar,
                                          std::unique_ptr<FunctionOrbit> child) {
    if (auto* existing = dynamic_cast<FunctionOrbit*>(get_child(grammar))) {
        return existing;
    }

    FunctionOrbit* raw_child = child.get();
    function_children_.push_back(std::move(child));
    assign_child(grammar, raw_child);
    return raw_child;
}

} // namespace cppfort::stage0


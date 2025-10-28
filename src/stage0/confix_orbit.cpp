#include "confix_orbit.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <limits>

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

    // Create function children based on grammar_modes bitmask
    // Bit 0 = C, Bit 1 = CPP, Bit 2 = CPP2
    FunctionOrbit* c_child = nullptr;
    FunctionOrbit* cpp_child = nullptr;
    FunctionOrbit* cpp2_child = nullptr;

    if (pattern.grammar_modes & 1) {  // C mode
        c_child = ensure_function_child(::cppfort::ir::GrammarType::C, new C_FunctionOrbit());
    }
    if (pattern.grammar_modes & 2) {  // CPP mode
        cpp_child = ensure_function_child(::cppfort::ir::GrammarType::CPP, new CPP_FunctionOrbit());
    }
    if (pattern.grammar_modes & 4) {  // CPP2 mode
        cpp2_child = ensure_function_child(::cppfort::ir::GrammarType::CPP2, new CPP2_FunctionOrbit());
    }

    // Set patterns from signature_patterns if available
    if (!pattern.signature_patterns.empty()) {
        if (c_child) c_child->set_pattern(pattern.signature_patterns[0]);
        if (cpp_child) cpp_child->set_pattern(pattern.signature_patterns[0]);
        if (cpp2_child) cpp2_child->set_pattern(pattern.signature_patterns[0]);
    } else {
        // Fallback patterns
        if (c_child) c_child->set_pattern("void %s()");
        if (cpp_child) cpp_child->set_pattern("auto %s() -> %s");
        if (cpp2_child) cpp2_child->set_pattern("%s: () -> %s");
    }

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

    // Apply pattern weight as confidence boost
    if (best_child && pattern.weight > 1.0) {
        best_child->confidence = std::min(1.0, best_child->confidence * pattern.weight);
    }

    winning_child_ = best_child;
    confidence = best_confidence;
}

void ConfixOrbit::extract_evidence(std::string_view text, size_t start, size_t end) {
    if (start >= text.size() || start >= end) {
        return;
    }

    if (auto memo = recall_idempotent_span(start, end)) {
        for (const auto& span : memo->spans) {
            add_evidence(span);
        }
        return;
    }

    end = std::min(end, text.size());
    size_t run_start = std::string_view::npos;
    std::vector<EvidenceSpan> captured_spans;

    for (size_t idx = start; idx < end; ++idx) {
        char ch = text[idx];
        if (is_anchor_char(ch)) {
            if (run_start != std::string_view::npos) {
                const auto length = idx - run_start;
                captured_spans.emplace_back(run_start, idx, std::string{text.substr(run_start, length)}, 0.5);
                add_evidence(captured_spans.back());
                run_start = std::string_view::npos;
            }
            continue;
        }

        if (run_start == std::string_view::npos) {
            run_start = idx;
        }
    }

    if (run_start != std::string_view::npos) {
        captured_spans.emplace_back(run_start, end, std::string{text.substr(run_start, end - run_start)}, 0.5);
        add_evidence(captured_spans.back());
    }

    if (!captured_spans.empty()) {
        remember_idempotent_span(start, end, captured_spans);
    }
}

namespace {
uint64_t make_memento_key(size_t start, size_t end) {
    return (static_cast<uint64_t>(start) << 32) ^
           static_cast<uint64_t>(end & std::numeric_limits<uint32_t>::max());
}
} // namespace

void ConfixOrbit::remember_combinator_hit(size_t start, size_t end,
                                          std::string pattern_name,
                                          double confidence,
                                          ::cppfort::ir::GrammarType grammar) {
    const uint64_t key = make_memento_key(start, end);
    CombinatorMemento entry{start, end, std::move(pattern_name), confidence, grammar};
    memento_cache_[key] = entry;

    // Maintain anchor chain as insertion-ordered unique list keyed by same span
    auto existing = std::find_if(anchor_chain_.begin(), anchor_chain_.end(),
        [&](const CombinatorMemento& memo) { return memo.start == start && memo.end == end; });
    if (existing != anchor_chain_.end()) {
        *existing = entry;
    } else {
        anchor_chain_.push_back(entry);
    }
}

std::optional<ConfixOrbit::CombinatorMemento> ConfixOrbit::recall_combinator_hit(size_t start, size_t end) const {
    const uint64_t key = make_memento_key(start, end);
    auto it = memento_cache_.find(key);
    if (it == memento_cache_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void ConfixOrbit::remember_idempotent_span(size_t start, size_t end, const std::vector<EvidenceSpan>& spans) {
    if (spans.empty()) {
        return;
    }
    const uint64_t key = make_memento_key(start, end);
    span_cache_[key] = SpanMemento{start, end, spans};
}

std::optional<ConfixOrbit::SpanMemento> ConfixOrbit::recall_idempotent_span(size_t start, size_t end) const {
    const uint64_t key = make_memento_key(start, end);
    auto it = span_cache_.find(key);
    if (it == span_cache_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void ConfixOrbit::seed_span_memento(const SpanMemento& memo) {
    if (memo.spans.empty()) {
        return;
    }
    const uint64_t key = make_memento_key(memo.start, memo.end);
    span_cache_[key] = memo;
}

void ConfixOrbit::merge_anchor_chain(const std::vector<CombinatorMemento>& chain) {
    for (const auto& memo : chain) {
        remember_combinator_hit(memo.start,
                                memo.end,
                                memo.pattern_name,
                                memo.confidence,
                                memo.grammar);
    }
}

FunctionOrbit* ConfixOrbit::ensure_function_child(::cppfort::ir::GrammarType grammar,
                                          FunctionOrbit* child) {
    if (auto* existing = dynamic_cast<FunctionOrbit*>(get_child(grammar))) {
        delete child; // Don't need the new one
        return existing;
    }

    FunctionOrbit* raw_child = child;
    function_children_.push_back(std::unique_ptr<FunctionOrbit>(child));
    assign_child(grammar, raw_child);
    return raw_child;
}

} // namespace cppfort::stage0

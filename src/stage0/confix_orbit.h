#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>  // for std::min
#include <unordered_map>
#include <optional>

#include "orbit_ring.h"
#include "function_orbit.h"

namespace cppfort {
namespace ir {
class RBCursiveScanner;
}
}

namespace cppfort::stage0 {
 
class FunctionOrbit;

class ConfixOrbit : public Orbit {
public:
    ConfixOrbit(char open_char, char close_char);

    bool matches(const EvidenceSpan& span) const override;

    bool validate_pair(char open, char close) const;

    char open_symbol() const { return open_char_; }
    char close_symbol() const { return close_char_; }

    ::cppfort::ir::GrammarType selected_grammar() const { return selected_grammar_; }
    const FunctionOrbit* winning_child() const { return winning_child_; }
    const std::string& selected_pattern() const { return selected_pattern_; }
    void set_selected_pattern(std::string pattern) { selected_pattern_ = std::move(pattern); }
    void set_selected_grammar(::cppfort::ir::GrammarType grammar) { selected_grammar_ = grammar; }

    void set_combinator(::cppfort::ir::RBCursiveScanner* c) { combinator_ = c; }
    ::cppfort::ir::RBCursiveScanner* get_combinator() const { return combinator_; }

    void accumulate_depth(int delta) {
        current_depth_ += delta;
        if (current_depth_ < 0) {
            current_depth_ = 0;
        }
        depth_counter_ = std::max(depth_counter_, current_depth_);
    }
    int depth() const { return depth_counter_; }

    // Scanner hint propagation to parent depth
    void propagate_scanner_hint_to_parent() {
        if (depth_counter_ > 0) {
            // Propagate hint to parent depth level - increment confidence for parent scope
            parent_hint_confidence_ = std::min(1.0, parent_hint_confidence_ + 0.1);
        }
    }
    
    double get_parent_hint_confidence() const { return parent_hint_confidence_; }

    void parameterize_children(const PatternData& pattern) override;
    void extract_evidence(std::string_view text, size_t start, size_t end);

    struct CombinatorMemento {
        size_t start = 0;
        size_t end = 0;
        std::string pattern_name;
        double confidence = 0.0;
        ::cppfort::ir::GrammarType grammar = ::cppfort::ir::GrammarType::UNKNOWN;
    };

    struct SpanMemento {
        size_t start = 0;
        size_t end = 0;
        std::vector<EvidenceSpan> spans;
    };

    void remember_combinator_hit(size_t start, size_t end,
                                 std::string pattern_name,
                                 double confidence,
                                 ::cppfort::ir::GrammarType grammar);
    std::optional<CombinatorMemento> recall_combinator_hit(size_t start, size_t end) const;
    const std::vector<CombinatorMemento>& chained_anchors() const { return anchor_chain_; }
    void remember_idempotent_span(size_t start, size_t end, const std::vector<EvidenceSpan>& spans);
    std::optional<SpanMemento> recall_idempotent_span(size_t start, size_t end) const;
    void seed_span_memento(const SpanMemento& memo);
    void merge_anchor_chain(const std::vector<CombinatorMemento>& chain);

private:
    FunctionOrbit* ensure_function_child(::cppfort::ir::GrammarType grammar,
                                          FunctionOrbit* child);

    char open_char_;
    char close_char_;
    int current_depth_ = 0;
    int depth_counter_ = 0;
    double parent_hint_confidence_ = 0.0;  // Scanner hint confidence for parent depth
    ::cppfort::ir::RBCursiveScanner* combinator_ = nullptr;
    std::vector<std::unique_ptr<FunctionOrbit>> function_children_;
    ::cppfort::ir::GrammarType selected_grammar_ = ::cppfort::ir::GrammarType::UNKNOWN;
    FunctionOrbit* winning_child_ = nullptr;
    std::string selected_pattern_;
    std::unordered_map<uint64_t, CombinatorMemento> memento_cache_;
    std::vector<CombinatorMemento> anchor_chain_;
    std::unordered_map<uint64_t, SpanMemento> span_cache_;
};

} // namespace cppfort::stage0

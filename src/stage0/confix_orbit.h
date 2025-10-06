#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

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

    void set_combinator(::cppfort::ir::RBCursiveScanner* c) { combinator_ = c; }
    ::cppfort::ir::RBCursiveScanner* get_combinator() const { return combinator_; }

    void accumulate_depth(int delta) { depth_counter_ += delta; }
    int depth() const { return depth_counter_; }

    void parameterize_children(const PatternData& pattern) override;

    void extract_evidence(std::string_view text, size_t start, size_t end);

private:
    FunctionOrbit* ensure_function_child(::cppfort::ir::GrammarType grammar,
                                          std::unique_ptr<FunctionOrbit> child);

    char open_char_;
    char close_char_;
    int depth_counter_ = 0;
    ::cppfort::ir::RBCursiveScanner* combinator_ = nullptr;
    std::vector<std::unique_ptr<FunctionOrbit>> function_children_;
    ::cppfort::ir::GrammarType selected_grammar_ = ::cppfort::ir::GrammarType::UNKNOWN;
    FunctionOrbit* winning_child_ = nullptr;
    std::string selected_pattern_;
};

} // namespace cppfort::stage0


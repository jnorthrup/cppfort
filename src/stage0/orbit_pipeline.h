#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "confix_orbit.h"
#include "orbit_iterator.h"
#include "pattern_loader.h"
#include "correlator.h"
#include "grammar_tree.h"
#include "wide_scanner.h"

namespace cppfort::stage0 {

class OrbitPipeline {
public:
    OrbitPipeline() = default;

    bool load_patterns(const std::string& path);

    size_t pattern_count() const { return loader_.pattern_count(); }

    const std::vector<PatternData>& patterns() const { return loader_.patterns(); }

    void populate_iterator(const std::vector<OrbitFragment>& fragments,
                           OrbitIterator& iterator,
                           std::string_view source);

    const std::vector<std::unique_ptr<ConfixOrbit>>& orbits() const { return confix_orbits_; }

private:
    std::unique_ptr<ConfixOrbit> evaluate_fragment(std::unique_ptr<ConfixOrbit> base_orbit, const OrbitFragment& fragment, std::string_view source) const;
    std::unique_ptr<ConfixOrbit> make_base_orbit(const OrbitFragment& fragment, std::string_view source) const;
    std::pair<char, char> select_confix(const OrbitFragment& fragment, std::string_view source) const;
    const PatternData* best_pattern_for(ConfixOrbit& candidate) const;
    void cache_orbit_state(const ConfixOrbit& orbit) const;
    std::optional<ConfixOrbit::CombinatorMemento> recall_global_memento(size_t start, size_t end) const;
    std::optional<ConfixOrbit::SpanMemento> recall_span_memento(size_t start, size_t end) const;
    uint64_t make_memento_key(size_t start, size_t end) const;

    PatternLoader loader_;
    FragmentCorrelator correlator_;
    GrammarTree grammar_tree_;
    std::vector<std::unique_ptr<ConfixOrbit>> confix_orbits_;
    mutable std::unordered_map<uint64_t, ConfixOrbit::CombinatorMemento> combinator_memos_;
    mutable std::unordered_map<uint64_t, ConfixOrbit::SpanMemento> span_memos_;
    mutable std::vector<ConfixOrbit::CombinatorMemento> global_anchor_chain_;
};

// Semantic-aware orbit builder that gates orbit creation on TypeEvidence validation
class SemanticOrbitBuilder {
public:
    SemanticOrbitBuilder(const std::vector<PatternData>& patterns) : patterns_(patterns) {}
    
    // Build orbit only if TypeEvidence validates the anchor tuple and evidence
    std::unique_ptr<ConfixOrbit> build_validated_orbit(
        const OrbitFragment& fragment, 
        std::string_view source,
        const PatternData* pattern = nullptr) const;
    
    // Check if anchor tuple + evidence combination is semantically valid
    bool validate_semantic_context(
        const PatternData& pattern,
        size_t anchor_index,
        std::string_view evidence_span,
        std::string_view expected_type) const;
    
private:
    const std::vector<PatternData>& patterns_;
    
    // Analyze evidence span for semantic validity
    ::cppfort::stage0::TypeEvidence analyze_evidence_span(std::string_view span) const;
    
    // Check if evidence type matches expected semantic constraints
    bool validate_evidence_constraints(
        const std::string& expected_type,
        std::string_view evidence,
        const ::cppfort::stage0::TypeEvidence& traits) const;
};

} // namespace cppfort::stage0

#include <cassert>
#include <string>
#include <vector>

#include "confix_orbit.h"

using namespace cppfort::stage0;

namespace {
::cppfort::ir::GrammarType kGrammar = ::cppfort::ir::GrammarType::CPP;
}

int main() {
    ConfixOrbit orbit('{', '}');
    orbit.remember_combinator_hit(10, 20, "demo_pattern", 0.6, kGrammar, {});

    auto memo = orbit.recall_combinator_hit(10, 20);
    assert(memo.has_value());
    assert(memo->pattern_name == "demo_pattern");
    assert(memo->grammar == kGrammar);

    std::vector<EvidenceSpan> spans;
    spans.emplace_back(2, 5, "abc", 0.5);
    spans.emplace_back(5, 8, "def", 0.5);
    orbit.remember_idempotent_span(2, 8, spans);

    auto span_memo = orbit.recall_idempotent_span(2, 8);
    assert(span_memo.has_value());
    assert(span_memo->spans.size() == 2);

    // Seed a second orbit with cached anchors and spans
    ConfixOrbit from_cache('{', '}');
    from_cache.merge_anchor_chain(orbit.chained_anchors());
    if (span_memo) {
        from_cache.seed_span_memento(*span_memo);
    }

    auto cached_hit = from_cache.recall_combinator_hit(10, 20);
    assert(cached_hit.has_value());
    assert(cached_hit->pattern_name == "demo_pattern");

    std::string text = "xxabcdefyy";
    from_cache.extract_evidence(text, 2, 8);
    assert(from_cache.evidence_count() == span_memo->spans.size());

    return 0;
}

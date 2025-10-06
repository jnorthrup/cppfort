#include "orbit_pipeline.h"

#include <algorithm>
#include <string_view>
#include <utility>

namespace cppfort::stage0 {
namespace {

std::string_view select_primary_text(const OrbitFragment& fragment) {
    if (!fragment.cpp2_text.empty()) return fragment.cpp2_text;
    if (!fragment.cpp_text.empty()) return fragment.cpp_text;
    return fragment.c_text;
}

bool contains_any(std::string_view text, std::string_view chars) {
    return text.find_first_of(chars) != std::string_view::npos;
}

} // namespace

bool OrbitPipeline::load_patterns(const std::string& path) {
    const bool ok = loader_.load_yaml(path);
    grammar_tree_.clear();
    if (ok) {
        for (const auto& pattern : loader_.patterns()) {
            grammar_tree_.insert(pattern);
        }
    }
    return ok;
}

std::pair<char, char> OrbitPipeline::select_confix(const OrbitFragment& fragment) const {
    std::string_view text = select_primary_text(fragment);

    if (contains_any(text, "{}")) {
        return {'{', '}'};
    }
    if (contains_any(text, "()")) {
        return {'(', ')'};
    }
    if (contains_any(text, "[]")) {
        return {'[', ']'};
    }
    if (contains_any(text, "<>")) {
        return {'<', '>'};
    }
    if (contains_any(text, "\"")) {
        return {'\"', '\"'};
    }
    return {'{', '}'};
}

std::unique_ptr<ConfixOrbit> OrbitPipeline::make_base_orbit(const OrbitFragment& fragment) const {
    auto [open_char, close_char] = select_confix(fragment);
    auto orbit = std::make_unique<ConfixOrbit>(open_char, close_char);
    orbit->start_pos = fragment.start_pos;
    orbit->end_pos = fragment.end_pos;
    orbit->confidence = 0.0;
    orbit->set_selected_pattern("default");

    const auto add_span = [&](const std::string& text) {
        if (!text.empty()) {
            orbit->add_evidence(EvidenceSpan{fragment.start_pos, fragment.end_pos, text, fragment.confidence});
        }
    };

    add_span(fragment.c_text);
    add_span(fragment.cpp_text);
    add_span(fragment.cpp2_text);

    return orbit;
}

std::unique_ptr<ConfixOrbit> OrbitPipeline::evaluate_fragment(const OrbitFragment& fragment) const {
    const auto& patterns = loader_.patterns();
    if (patterns.empty()) {
        return make_base_orbit(fragment);
    }

    std::unique_ptr<ConfixOrbit> best_orbit;
    double best_confidence = -1.0;
    std::string best_pattern_name;

    for (const auto& pattern : patterns) {
        auto candidate = make_base_orbit(fragment);
        candidate->parameterize_children(pattern);
        if (candidate->confidence > best_confidence) {
            best_confidence = candidate->confidence;
            best_pattern_name = pattern.name;
            best_orbit = std::move(candidate);
        }
    }

    if (!best_orbit) {
        return make_base_orbit(fragment);
    }

    best_orbit->set_selected_pattern(std::move(best_pattern_name));
    return best_orbit;
}

void OrbitPipeline::populate_iterator(const std::vector<OrbitFragment>& fragments,
                                      OrbitIterator& iterator) {
    iterator.clear();
    confix_orbits_.clear();

    for (const auto& fragment : fragments) {
        OrbitFragment enriched = fragment;
        correlator_.correlate(enriched);
        auto orbit = evaluate_fragment(enriched);
        if (!orbit) {
            continue;
        }
        iterator.add_orbit(orbit.get());
        confix_orbits_.push_back(std::move(orbit));
    }

    iterator.reset();
}

} // namespace cppfort::stage0


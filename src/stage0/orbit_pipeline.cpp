#include "orbit_pipeline.h"

#include <algorithm>
#include <iostream>
#include <string_view>
#include <utility>

namespace cppfort::stage0 {
namespace {

std::string_view extract_fragment_view(const OrbitFragment& fragment, std::string_view source) {
    if (fragment.start_pos >= source.size() || fragment.end_pos > source.size() || fragment.start_pos >= fragment.end_pos) {
        return {};
    }
    return source.substr(fragment.start_pos, fragment.end_pos - fragment.start_pos);
}

bool contains_any(std::string_view text, std::string_view chars) {
    return text.find_first_of(chars) != std::string_view::npos;
}

} // namespace

// Classify grammar based on pattern characteristics
::cppfort::ir::GrammarType classify_grammar_from_pattern(const PatternData& pattern) {
    // Check signature patterns for grammar-specific markers
    for (const auto& sig : pattern.signature_patterns) {
        if (sig.find(": (") != std::string::npos || sig.find(":(") != std::string::npos) {
            return ::cppfort::ir::GrammarType::CPP2;
        }
        if (sig.find("::") != std::string::npos || sig.find("template") != std::string::npos) {
            return ::cppfort::ir::GrammarType::CPP;
        }
        if (sig.find("typedef") != std::string::npos || sig.find("struct") != std::string::npos) {
            return ::cppfort::ir::GrammarType::C;
        }
    }
    
    // Check pattern name for fallback classification
    if (pattern.name.find("cpp2") != std::string::npos || pattern.name.find("CPP2") != std::string::npos) {
        return ::cppfort::ir::GrammarType::CPP2;
    }
    if (pattern.name.find("cpp") != std::string::npos || pattern.name.find("CPP") != std::string::npos) {
        return ::cppfort::ir::GrammarType::CPP;
    }
    if (pattern.name.find("function") != std::string::npos) {
        return ::cppfort::ir::GrammarType::CPP2;  // function_declaration uses : (
    }
    
    return ::cppfort::ir::GrammarType::UNKNOWN;
}

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

std::pair<char, char> OrbitPipeline::select_confix(const OrbitFragment& fragment, std::string_view source) const {
    std::string_view text = extract_fragment_view(fragment, source);

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

std::unique_ptr<ConfixOrbit> OrbitPipeline::make_base_orbit(const OrbitFragment& fragment, std::string_view source) const {
    auto [open_char, close_char] = select_confix(fragment, source);
    auto orbit = std::make_unique<ConfixOrbit>(open_char, close_char);
    orbit->start_pos = fragment.start_pos;
    orbit->end_pos = fragment.end_pos;
    orbit->confidence = fragment.confidence;
    orbit->set_selected_pattern("default");

    // Extract evidence spans for non-anchor character runs within the fragment
    orbit->extract_evidence(source, fragment.start_pos, fragment.end_pos);

    return orbit;
}

std::unique_ptr<ConfixOrbit> OrbitPipeline::evaluate_fragment(std::unique_ptr<ConfixOrbit> base_orbit, const OrbitFragment& fragment, std::string_view source) const {
    
    // Try speculation if we have a combinator with patterns
    if (auto* combinator = base_orbit->get_combinator()) {
        // std::cout << "DEBUG: Have combinator, trying speculation\n";
        // Speculate on full source for alternating patterns
        combinator->speculate(source);
        
        // Get the best speculative match
        if (auto* best_match = combinator->get_best_match()) {
            // Find the corresponding pattern
            const auto& patterns = loader_.patterns();
            auto pattern_it = std::find_if(patterns.begin(), patterns.end(),
                [&](const PatternData& p) { return p.name == best_match->pattern_name; });
            
            if (pattern_it != patterns.end()) {
                // Use the speculated pattern
                base_orbit->parameterize_children(*pattern_it);
                base_orbit->set_selected_pattern(best_match->pattern_name);
                base_orbit->set_selected_grammar(classify_grammar_from_pattern(*pattern_it));

                base_orbit->confidence = std::max(base_orbit->confidence, best_match->confidence);
                return base_orbit;
            }
        }
    }
    
    // Fallback: try all patterns if speculation didn't work
    const auto& patterns = loader_.patterns();
    if (patterns.empty()) {
        return base_orbit;
    }

    std::unique_ptr<ConfixOrbit> best_orbit;
    double best_confidence = base_orbit->confidence;
    std::string best_pattern_name = base_orbit->selected_pattern();

    for (const auto& pattern : patterns) {
        auto candidate = make_base_orbit(fragment, source);
        candidate->parameterize_children(pattern);
        if (candidate->confidence > best_confidence) {
            best_confidence = candidate->confidence;
            best_pattern_name = pattern.name;
            best_orbit = std::move(candidate);
        }
    }

    if (!best_orbit) {
        return base_orbit;
    }

    best_orbit->set_selected_pattern(std::move(best_pattern_name));
    
    // Set grammar for the best pattern
    auto pattern_it = std::find_if(patterns.begin(), patterns.end(),
        [&](const PatternData& p) { return p.name == best_pattern_name; });
    if (pattern_it != patterns.end()) {
        best_orbit->set_selected_grammar(classify_grammar_from_pattern(*pattern_it));
    }
    
    return best_orbit;
}

void OrbitPipeline::populate_iterator(const std::vector<OrbitFragment>& fragments,
                                      OrbitIterator& iterator,
                                      std::string_view source) {
    // std::cout << "DEBUG: populate_iterator called with " << fragments.size() << " fragments\n";
    iterator.clear();
    iterator.set_patterns(loader_.patterns());
    confix_orbits_.clear();

    for (const auto& fragment : fragments) {
        // std::cout << "DEBUG: Processing fragment [" << fragment.start_pos << ", " << fragment.end_pos << ") confidence=" << fragment.confidence << "\n";
        OrbitFragment enriched = fragment;
        correlator_.correlate(enriched, source);
        
        // Create base orbit
        auto orbit = make_base_orbit(enriched, source);
        
        // Temporarily set up combinator for speculation during evaluation
        ::cppfort::ir::RBCursiveScanner temp_scanner;
        temp_scanner.set_patterns(loader_.patterns());
        if (auto* confix = dynamic_cast<ConfixOrbit*>(orbit.get())) {
            // std::cout << "DEBUG: Setting temporary combinator on orbit\n";
            confix->set_combinator(&temp_scanner);
        }

        // Evaluate with speculation
        orbit = evaluate_fragment(std::move(orbit), enriched, source);

        // Clear temporary combinator
        if (auto* confix = dynamic_cast<ConfixOrbit*>(orbit.get())) {
            confix->set_combinator(nullptr);
        }

        if (!orbit) {
            // std::cout << "DEBUG: evaluate_fragment returned null\n";
            continue;
        }
        iterator.add_orbit(orbit.get());
        confix_orbits_.push_back(std::move(orbit));
    }

    iterator.reset();
}

} // namespace cppfort::stage0


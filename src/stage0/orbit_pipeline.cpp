#include "orbit_pipeline.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
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

    if (auto span_memo = recall_span_memento(fragment.start_pos, fragment.end_pos)) {
        orbit->seed_span_memento(*span_memo);
    }

    if (!global_anchor_chain_.empty()) {
        orbit->merge_anchor_chain(global_anchor_chain_);
    }

    // Extract evidence spans for non-anchor character runs within the fragment
    orbit->extract_evidence(source, fragment.start_pos, fragment.end_pos);

    // Track confix depth for this orbit's region
    // Scan from start to end, updating depth when we see matching delimiters
    int depth_delta = 0;
    for (size_t pos = fragment.start_pos; pos < fragment.end_pos && pos < source.size(); ++pos) {
        char ch = source[pos];
        if (ch == open_char) {
            orbit->accumulate_depth(1);
            depth_delta++;
        } else if (ch == close_char) {
            orbit->accumulate_depth(-1);
            depth_delta--;
        }
    }

    return orbit;
}

std::unique_ptr<ConfixOrbit> OrbitPipeline::evaluate_fragment(std::unique_ptr<ConfixOrbit> base_orbit, const OrbitFragment& fragment, std::string_view source) const {
    
    if (auto* confix_cached = base_orbit.get()) {
        auto apply_memo = [&](const ConfixOrbit::CombinatorMemento& memo) -> std::unique_ptr<ConfixOrbit> {
            const auto& patterns = loader_.patterns();
            auto pattern_it = std::find_if(patterns.begin(), patterns.end(),
                [&](const PatternData& p) { return p.name == memo.pattern_name; });
            if (pattern_it == patterns.end()) {
                return nullptr;
            }
            confix_cached->parameterize_children(*pattern_it);
            confix_cached->set_selected_pattern(memo.pattern_name);
            confix_cached->set_selected_grammar(memo.grammar);
            confix_cached->confidence = std::max(confix_cached->confidence, memo.confidence);
            confix_cached->propagate_scanner_hint_to_parent();
            return std::move(base_orbit);
        };

        if (auto memo = confix_cached->recall_combinator_hit(fragment.start_pos, fragment.end_pos)) {
            if (auto cached_orbit = apply_memo(*memo)) {
                return cached_orbit;
            }
        } else if (auto global_memo = recall_global_memento(fragment.start_pos, fragment.end_pos)) {
            confix_cached->remember_combinator_hit(global_memo->start,
                                                   global_memo->end,
                                                   global_memo->pattern_name,
                                                   global_memo->confidence,
                                                   global_memo->grammar,
                                                   global_memo->traits);
            if (auto cached_orbit = apply_memo(*global_memo)) {
                return cached_orbit;
            }
        }
    }
    
    // Try speculation if we have a combinator with patterns
    if (auto* combinator = base_orbit->get_combinator()) {
        // Extract fragment text for pattern matching
        std::string_view fragment_text = source.substr(fragment.start_pos, fragment.end_pos - fragment.start_pos);
        std::cerr << "DEBUG evaluate_fragment: Speculating on fragment [" << fragment.start_pos << ", " << fragment.end_pos << "): '"
                  << fragment_text << "'\n";
        // Use backward chaining by default for better semantic accuracy
        combinator->speculate_backchain(fragment_text);

        // Log backchain usage
        std::cerr << "DEBUG: Using backward chaining for terminal speculation\n";

        // Get the best speculative match
        if (auto* best_match = combinator->get_best_match()) {
            std::cerr << "DEBUG evaluate_fragment: Best match = " << best_match->pattern_name << " (confidence=" << best_match->confidence << ")\n";
            // Find the corresponding pattern
            const auto& patterns = loader_.patterns();
            auto pattern_it = std::find_if(patterns.begin(), patterns.end(),
                [&](const PatternData& p) { return p.name == best_match->pattern_name; });

            if (pattern_it != patterns.end()) {
                // Use the speculated pattern
                base_orbit->parameterize_children(*pattern_it);
                base_orbit->set_selected_pattern(best_match->pattern_name);
                base_orbit->set_selected_grammar(classify_grammar_from_pattern(*pattern_it));

                // NEW: Propagate scanner hint to parent depth
                base_orbit->propagate_scanner_hint_to_parent();
                
                base_orbit->confidence = std::max(base_orbit->confidence, best_match->confidence);
                base_orbit->remember_combinator_hit(fragment.start_pos,
                                                    fragment.end_pos,
                                                    best_match->pattern_name,
                                                    base_orbit->confidence,
                                                    base_orbit->selected_grammar(),
                                                    {}); // Empty traits for now
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
    if (auto grammar = best_orbit->selected_grammar(); grammar != ::cppfort::ir::GrammarType::UNKNOWN) {
        best_orbit->remember_combinator_hit(fragment.start_pos,
                                            fragment.end_pos,
                                            best_orbit->selected_pattern(),
                                            best_orbit->confidence,
                                            grammar,
                                            {}); // Empty traits for now
    }
    
    return best_orbit;
}

static std::pair<char, char> select_confix_for_semantic_builder(
    const OrbitFragment& fragment, std::string_view source) {
    
    std::string_view text = source.substr(fragment.start_pos, 
                                          fragment.end_pos - fragment.start_pos);
    
    if (text.find('<') != std::string::npos && text.find('>') != std::string::npos) {
        return {'<', '>'};
    }
    if (text.find('(') != std::string::npos && text.find(')') != std::string::npos) {
        return {'(', ')'};
    }
    if (text.find('[') != std::string::npos && text.find(']') != std::string::npos) {
        return {'[', ']'};
    }
    if (text.find('"') != std::string::npos) {
        return {'"', '"'};
    }
    return {'{', '}'};
}

// SemanticOrbitBuilder implementation
std::unique_ptr<ConfixOrbit> SemanticOrbitBuilder::build_validated_orbit(
    const OrbitFragment& fragment, 
    std::string_view source,
    const PatternData* pattern) const {
    
    // If no pattern specified, try to find the best matching pattern
    const PatternData* target_pattern = pattern;
    if (!target_pattern) {
        // For now, use a simple heuristic - this could be enhanced
        for (const auto& p : patterns_) {
            if (p.name.find("function") != std::string::npos && 
                fragment.classified_grammar == ::cppfort::ir::GrammarType::CPP2) {
                target_pattern = &p;
                break;
            }
        }
        if (!target_pattern && !patterns_.empty()) {
            target_pattern = &patterns_[0];
        }
    }
    
    if (!target_pattern) {
        return nullptr; // No valid pattern found
    }
    
    // Extract evidence spans using the pattern's alternating anchors
    std::string_view fragment_text = source.substr(fragment.start_pos, 
                                                   fragment.end_pos - fragment.start_pos);
    
    std::vector<std::string_view> evidence_spans;
    size_t current_pos = 0;
    
    // Extract evidence spans based on alternating anchors
    for (size_t i = 0; i < target_pattern->alternating_anchors.size(); ++i) {
        const std::string& anchor = target_pattern->alternating_anchors[i];
        size_t anchor_pos = fragment_text.find(anchor, current_pos);
        
        if (anchor_pos == std::string::npos) {
            return nullptr; // Required anchor not found
        }
        
        // Extract evidence before this anchor (if expected)
        if (i < target_pattern->evidence_types.size()) {
            std::string_view evidence = fragment_text.substr(current_pos, anchor_pos - current_pos);
            // Trim whitespace
            size_t start = 0;
            while (start < evidence.size() && std::isspace(static_cast<unsigned char>(evidence[start]))) ++start;
            size_t end = evidence.size();
            while (end > start && std::isspace(static_cast<unsigned char>(evidence[end - 1]))) --end;
            evidence = evidence.substr(start, end - start);
            
            evidence_spans.push_back(evidence);
        }
        
        current_pos = anchor_pos + anchor.size();
    }
    
    // Extract final evidence span after last anchor
    if (target_pattern->evidence_types.size() > target_pattern->alternating_anchors.size()) {
        std::string_view evidence = fragment_text.substr(current_pos);
        // Trim whitespace and find end of construct
        size_t start = 0;
        while (start < evidence.size() && std::isspace(static_cast<unsigned char>(evidence[start]))) ++start;
        size_t end = start;
        while (end < evidence.size() && evidence[end] != ';' && evidence[end] != '\n' && evidence[end] != '}') ++end;
        evidence = evidence.substr(start, end - start);
        
        evidence_spans.push_back(evidence);
    }
    
    // Validate all evidence spans against expected types
    if (evidence_spans.size() != target_pattern->evidence_types.size()) {
        return nullptr; // Wrong number of evidence spans
    }
    
    for (size_t i = 0; i < evidence_spans.size(); ++i) {
        const std::string& expected_type = target_pattern->evidence_types[i];
        std::string_view evidence = evidence_spans[i];
        
        if (!validate_semantic_context(*target_pattern, i, evidence, expected_type)) {
            return nullptr; // Semantic validation failed
        }
    }
    
    // If validation passes, create the orbit
    auto [open_char, close_char] = select_confix_for_semantic_builder(fragment, source);
    auto orbit = std::make_unique<ConfixOrbit>(open_char, close_char);
    orbit->start_pos = fragment.start_pos;
    orbit->end_pos = fragment.end_pos;
    orbit->confidence = fragment.confidence;
    orbit->set_selected_pattern(target_pattern->name);
    orbit->set_selected_grammar(fragment.classified_grammar);
    
    return orbit;
}

bool SemanticOrbitBuilder::validate_semantic_context(
    const PatternData& pattern,
    size_t anchor_index,
    std::string_view evidence,
    std::string_view expected_type) const {
    
    // Use TypeEvidence to analyze the semantic context
    ::cppfort::stage0::TypeEvidence traits;
    traits.ingest(evidence);
    
    // Check if the evidence matches the expected type based on traits
    if (expected_type == "identifier") {
        // Check for valid identifier characters
        if (evidence.empty() || (!std::isalpha(static_cast<unsigned char>(evidence[0])) && evidence[0] != '_')) {
            return false;
        }
        for (char c : evidence) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                return false;
            }
        }
        // Identifiers shouldn't have structural punctuation
        return traits.colon == 0 && traits.double_colon == 0 && traits.angle_open == 0 &&
               traits.brace_open == 0 && traits.paren_open == 0 && traits.arrow == 0;
    } else if (expected_type == "type") {
        // Types can have :: and <> for templates/namespaces
        return traits.double_colon > 0 || traits.angle_open > 0 || 
               (evidence.find("int") != std::string::npos || evidence.find("void") != std::string::npos ||
                evidence.find("char") != std::string::npos || evidence.find("bool") != std::string::npos);
    } else if (expected_type == "expression") {
        // Expressions can have operators and parentheses
        return traits.paren_open > 0 || traits.arrow > 0 || evidence.find('+') != std::string::npos ||
               evidence.find('-') != std::string::npos || evidence.find('*') != std::string::npos ||
               evidence.find('/') != std::string::npos;
    } else if (expected_type == "statement") {
        // Statements often end with semicolons or have braces
        return traits.brace_open > 0 || std::string(evidence).find(';') != std::string::npos;
    } else if (expected_type == "function") {
        // Functions have parentheses for parameters
        return traits.paren_open > 0;
    } else if (expected_type == "class") {
        // Classes/structs have braces
        return traits.brace_open > 0;
    } else if (expected_type == "template") {
        // Templates have angle brackets
        return traits.angle_open > 0;
    } else if (expected_type == "literal") {
        // Literals are simple tokens without complex structure
        return traits.paren_open == 0 && traits.brace_open == 0 && traits.angle_open == 0 &&
               (evidence.find('"') != std::string::npos || evidence.find('\'') != std::string::npos ||
                std::all_of(evidence.begin(), evidence.end(), ::isdigit));
    } else if (expected_type == "operator") {
        // Operators are short sequences of special characters
        return evidence.size() <= 3 && (evidence.find('+') != std::string::npos || 
               evidence.find('-') != std::string::npos || evidence.find('*') != std::string::npos ||
               evidence.find('/') != std::string::npos || evidence.find('=') != std::string::npos ||
               evidence.find('<') != std::string::npos || evidence.find('>') != std::string::npos);
    } else {
        // For unknown types, do basic validation
        return !evidence.empty() && evidence.find_first_not_of(" \t\n\r") != std::string::npos;
    }
}

void OrbitPipeline::populate_iterator(const std::vector<OrbitFragment>& fragments,
                                      OrbitIterator& iterator,
                                      std::string_view source) {
    iterator.clear();
    iterator.set_patterns(loader_.patterns());
    confix_orbits_.clear();

    // Create semantic orbit builder for validated orbit creation
    SemanticOrbitBuilder semantic_builder(loader_.patterns());

    for (const auto& fragment : fragments) {
        // std::cout << "DEBUG: Processing fragment [" << fragment.start_pos << ", " << fragment.end_pos << ") confidence=" << fragment.confidence << "\n";
        OrbitFragment enriched = fragment;
        correlator_.correlate(enriched, source);
        
        // Use semantic orbit builder to create validated orbit
        auto orbit = semantic_builder.build_validated_orbit(enriched, source);
        
        if (!orbit) {
            // Semantic validation failed - skip this fragment
            continue;
        }
        
        // Temporarily set up combinator for speculation during evaluation
        ::cppfort::stage0::RBCursiveScanner temp_scanner;
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
            continue;
        }

        cache_orbit_state(*orbit);

        iterator.add_orbit(orbit.get());
        confix_orbits_.push_back(std::move(orbit));
    }

    iterator.reset();
}

uint64_t OrbitPipeline::make_memento_key(size_t start, size_t end) const {
    return (static_cast<uint64_t>(start) << 32) ^
           static_cast<uint64_t>(end & std::numeric_limits<uint32_t>::max());
}

std::optional<ConfixOrbit::CombinatorMemento> OrbitPipeline::recall_global_memento(size_t start, size_t end) const {
    const uint64_t key = make_memento_key(start, end);
    auto it = combinator_memos_.find(key);
    if (it == combinator_memos_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<ConfixOrbit::SpanMemento> OrbitPipeline::recall_span_memento(size_t start, size_t end) const {
    const uint64_t key = make_memento_key(start, end);
    auto it = span_memos_.find(key);
    if (it == span_memos_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void OrbitPipeline::cache_orbit_state(const ConfixOrbit& orbit) const {
    for (const auto& memo : orbit.chained_anchors()) {
        const uint64_t key = make_memento_key(memo.start, memo.end);
        combinator_memos_[key] = memo;

        auto existing = std::find_if(global_anchor_chain_.begin(), global_anchor_chain_.end(),
            [&](const ConfixOrbit::CombinatorMemento& entry) {
                return entry.start == memo.start && entry.end == memo.end;
            });
        if (existing != global_anchor_chain_.end()) {
            *existing = memo;
        } else {
            global_anchor_chain_.push_back(memo);
        }
    }

    if (auto span_memo = orbit.recall_idempotent_span(orbit.start_pos, orbit.end_pos)) {
        const uint64_t key = make_memento_key(span_memo->start, span_memo->end);
        span_memos_[key] = *span_memo;
    }
}

} // namespace cppfort::stage0

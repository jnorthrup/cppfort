#include "cpp2_emitter.h"

#include "confix_orbit.h"
#include "confix_tracker.h"
#include "pattern_loader.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>

namespace cppfort::stage0 {

namespace {

// Debug hook – currently disabled, can be wired to a compile definition.
constexpr bool kEmitterDebugEnabled = false;
#define CPP2_EMITTER_DEBUG(stmt) \
    do {                          \
        if (kEmitterDebugEnabled) \
            stmt;                 \
    } while (false)

inline std::string trim_copy(std::string_view text) {
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

// Simple placeholder substitution: $N where N is 0/1-based index + offset.
std::string apply_substitution_with_offset(const std::string& template_str,
                                           const std::vector<std::string>& segments,
                                           int placeholder_offset) {
    std::string result = template_str;

    for (size_t i = 0; i < segments.size(); ++i) {
        std::string placeholder = "$" + std::to_string(static_cast<int>(i) + placeholder_offset);
        size_t pos = 0;
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.length(), segments[i]);
            pos += segments[i].length();
        }
    }

    return result;
}

// Map GrammarType to the bit used in bnfc / transformation_templates.
inline int grammar_to_mask(::cppfort::ir::GrammarType g) {
    using ::cppfort::ir::GrammarType;
    switch (g) {
        case GrammarType::C: return 1;
        case GrammarType::CPP: return 2;
        case GrammarType::CPP2: return 4;
        default: return 0;
    }
}

} // namespace

// Emit transformed output from orbit iterator using pattern-driven templates.
void CPP2Emitter::emit(OrbitIterator& iterator,
                       std::string_view source,
                       std::ostream& out,
                       const std::vector<PatternData>& patterns) const {
    size_t last_pos = 0;

    iterator.reset();
    while (Orbit* orbit = iterator.next()) {
        // Emit gap between previous orbit and this one unchanged.
        if (orbit->start_pos > last_pos && orbit->start_pos <= source.size()) {
            out << source.substr(last_pos, orbit->start_pos - last_pos);
        }

        const auto* confix = dynamic_cast<ConfixOrbit*>(orbit);
        const PatternData* pattern = nullptr;

        if (confix) {
            const std::string& pattern_name = confix->selected_pattern();
            if (!pattern_name.empty()) {
                for (const auto& p : patterns) {
                    if (p.name == pattern_name) {
                        pattern = &p;
                        break;
                    }
                }
            }

            emit_orbit(*confix, source, out, pattern, patterns);
        } else {
            // Non-confix orbits are emitted as raw slices.
            if (orbit->end_pos > orbit->start_pos && orbit->end_pos <= source.size()) {
                out << source.substr(orbit->start_pos, orbit->end_pos - orbit->start_pos);
            }
        }

        last_pos = std::min(orbit->end_pos, source.size());
    }

    // Emit any trailing content after the last orbit.
    if (last_pos < source.size()) {
        out << source.substr(last_pos);
    }
}

// Depth-based emission hook – currently a simple passthrough to preserve behaviour.
void CPP2Emitter::emit_depth_based(std::string_view source,
                                   std::ostream& out,
                                   const std::vector<PatternData>&) const {
    out << source;
}

void CPP2Emitter::emit_fragment(const OrbitFragment& fragment,
                                std::string_view source,
                                std::ostream& out) const {
    out << extract_fragment_text(fragment, source);
}

void CPP2Emitter::emit_orbit(const ConfixOrbit& orbit,
                             std::string_view source,
                             std::ostream& out,
                             const PatternData* pattern,
                             const std::vector<PatternData>&) const {
    if (orbit.start_pos >= source.size() || orbit.end_pos > source.size() ||
        orbit.start_pos >= orbit.end_pos) {
        return;
    }

    std::string_view text = source.substr(orbit.start_pos, orbit.end_pos - orbit.start_pos);

    // Skip empty or whitespace-only spans.
    bool whitespace_only = true;
    for (char c : text) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            whitespace_only = false;
            break;
        }
    }
    if (whitespace_only) {
        out << text;
        return;
    }

    // If there is no pattern or the orbit has no confidence, emit original text.
    if (!pattern || orbit.confidence == 0.0 ||
        pattern->substitution_templates.empty()) {
        out << text;
        return;
    }

    // Currently we always target C++ as the output grammar.
    const int target_mask = grammar_to_mask(::cppfort::ir::GrammarType::CPP);
    auto tpl_it = pattern->substitution_templates.find(target_mask);
    if (tpl_it == pattern->substitution_templates.end()) {
        // Fallback: if no C++ template, try any available template.
        tpl_it = pattern->substitution_templates.begin();
        if (tpl_it == pattern->substitution_templates.end()) {
            out << text;
            return;
        }
    }

    std::vector<std::string> segments;
    if (pattern->use_alternating && !pattern->alternating_anchors.empty()) {
        segments = extract_alternating_segments(text, *pattern);
        if (segments.empty()) {
            CPP2_EMITTER_DEBUG(std::cerr << "CPP2Emitter: no segments for pattern '" << pattern->name
                                         << "', emitting original span\n");
            out << text;
            return;
        }
    } else {
        // Segment-based patterns are not wired in this minimal emitter; emit raw text.
        out << text;
        return;
    }

    // Apply template with 1-based placeholders for alternating patterns.
    const int placeholder_offset = pattern->use_alternating ? 1 : 0;
    std::string substituted = apply_substitution_with_offset(tpl_it->second, segments, placeholder_offset);
    out << substituted;
}

// Extract segments for alternating anchor/evidence patterns (public for recursive use).
// This implementation is adapted from the previous full emitter, but trimmed to the
// essentials and reuses the shared ConfixTracker depth map helper.
std::vector<std::string> CPP2Emitter::extract_alternating_segments(std::string_view text,
                                                                   const PatternData& pattern) const {
    std::vector<std::string> segments;

    if (pattern.alternating_anchors.empty()) {
        return segments;
    }

    // Build confix depth map to validate pattern boundaries.
    auto depth_map = build_depth_map(text);

    // Find the first anchor.
    const std::string& first_anchor = pattern.alternating_anchors[0];
    size_t anchor_pos = text.find(first_anchor);
    if (anchor_pos == std::string::npos) {
        CPP2_EMITTER_DEBUG(std::cerr << "DEBUG extract_alternating_segments: first anchor '" << first_anchor
                                     << "' not found in text='" << text << "'\n");
        return segments;
    }

    // Record the confix depth at pattern start (reserved for future heuristics).
    int pattern_start_depth = (anchor_pos < depth_map.size()) ? depth_map[anchor_pos] : 0;
    (void)pattern_start_depth;

    // Special case: one anchor, two evidence spans => before and after anchor.
    if (pattern.alternating_anchors.size() == 1 && pattern.evidence_types.size() == 2) {
        std::string before = trim_copy(text.substr(0, anchor_pos));

        size_t after_start = anchor_pos + first_anchor.length();
        std::string after = trim_copy(text.substr(after_start));
        if (!after.empty() && after.back() == ';') {
            after.pop_back();
        }

        segments.push_back(std::move(before));
        segments.push_back(std::move(after));
        return segments;
    }

    // When we have N evidence types for N-1 anchors, capture the prefix before the first anchor.
    size_t evidence_start_idx = 0;
    if (pattern.evidence_types.size() > pattern.alternating_anchors.size()) {
        std::string before = trim_copy(text.substr(0, anchor_pos));
        segments.push_back(std::move(before));
        evidence_start_idx = 1;
    }

    size_t current_pos = anchor_pos + first_anchor.length();

    // Extract evidence spans between anchors and after the last one.
    for (size_t i = evidence_start_idx; i < pattern.evidence_types.size(); ++i) {
        size_t next_anchor_pos = std::string::npos;
        size_t anchor_idx = i - evidence_start_idx + 1; // index into alternating_anchors

        if (anchor_idx < pattern.alternating_anchors.size()) {
            const std::string& next_anchor = pattern.alternating_anchors[anchor_idx];
            next_anchor_pos = text.find(next_anchor, current_pos);

            if (next_anchor_pos == std::string::npos) {
                CPP2_EMITTER_DEBUG(std::cerr << "DEBUG extract_alternating_segments: anchor '" << next_anchor
                                             << "' not found after position " << current_pos
                                             << " in text='" << text << "'\n");
                return {};
            }
        }

        size_t evidence_end = (next_anchor_pos != std::string::npos) ? next_anchor_pos : text.size();
        std::string evidence = std::string(text.substr(current_pos, evidence_end - current_pos));

        // Trim whitespace.
        evidence = trim_copy(evidence);

        // Strip trailing semicolon if present (templates add it back).
        if (!evidence.empty() && evidence.back() == ';') {
            evidence.pop_back();
        }

        // Basic brace-balance check within the evidence span.
        int brace_balance = 0;
        for (char ch : evidence) {
            if (ch == '{') {
                ++brace_balance;
            } else if (ch == '}') {
                if (brace_balance == 0) {
                    CPP2_EMITTER_DEBUG(std::cerr
                                       << "DEBUG extract_alternating_segments: unmatched closing brace in evidence for pattern '"
                                       << pattern.name << "' text='" << evidence << "'\n");
                    return {};
                }
                --brace_balance;
            }
        }

        segments.push_back(std::move(evidence));
        current_pos = evidence_end;

        if (next_anchor_pos != std::string::npos && anchor_idx < pattern.alternating_anchors.size()) {
            current_pos += pattern.alternating_anchors[anchor_idx].length();
        }
    }

    return segments;
}

std::string_view CPP2Emitter::extract_fragment_text(const OrbitFragment& fragment,
                                                    std::string_view source) const {
    if (fragment.start_pos >= source.size() || fragment.end_pos > source.size() ||
        fragment.start_pos >= fragment.end_pos) {
        return {};
    }
    return source.substr(fragment.start_pos, fragment.end_pos - fragment.start_pos);
}

namespace testing {

std::string transform_parameter_for_testing(std::string_view param) {
    // Very small helper for unit tests: rewrite "name: type" into "type name".
    std::string_view input = param;
    size_t colon = input.find(':');
    if (colon == std::string::npos) {
        return std::string(param);
    }

    std::string name = trim_copy(input.substr(0, colon));
    std::string type = trim_copy(input.substr(colon + 1));
    if (name.empty() || type.empty()) {
        return std::string(param);
    }

    return type + " " + name;
}

} // namespace testing

} // namespace cppfort::stage0

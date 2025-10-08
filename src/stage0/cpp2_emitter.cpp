#include "cpp2_emitter.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string_view>
#include <regex>

#include "confix_orbit.h"
#include "pattern_loader.h"

namespace cppfort::stage0 {

namespace {

// PATTERN-DRIVEN TRANSFORMATION ENGINE (anchor-based substitution)

// Extract segment from text using anchor position and delimiters
std::string extract_segment(std::string_view text, size_t anchor_pos, const AnchorSegment& seg) {
    size_t start = anchor_pos;

    // Apply offset
    if (seg.offset_from_anchor < 0) {
        // Before anchor - find identifier before anchor
        size_t scan_end = anchor_pos;
        while (scan_end > 0 && std::isspace(text[scan_end - 1])) {
            --scan_end;
        }
        size_t ident_start = scan_end;
        while (ident_start > 0 && (std::isalnum(text[ident_start - 1]) || text[ident_start - 1] == '_')) {
            --ident_start;
        }
        if (ident_start < scan_end) {
            return std::string(text.substr(ident_start, scan_end - ident_start));
        }
        return "";
    }

    // Find delimiter start
    if (!seg.delimiter_start.empty()) {
        size_t delim_pos = text.find(seg.delimiter_start, start);
        if (delim_pos == std::string::npos) return "";
        start = delim_pos + seg.delimiter_start.length();
    }

    // Find delimiter end
    if (!seg.delimiter_end.empty()) {
        // For multi-char delimiters or non-nesting delimiters, use find
        if (seg.delimiter_end.size() > 1 || (seg.delimiter_start != "(" && seg.delimiter_start != "[" && seg.delimiter_start != "{")) {
            size_t end_pos = text.find(seg.delimiter_end, start);
            if (end_pos != std::string::npos) {
                return std::string(text.substr(start, end_pos - start));
            }
        } else {
            // Original nesting logic for brackets
            int depth = 1;
            size_t end = start;
            char open_char = seg.delimiter_start.empty() ? '\0' : seg.delimiter_start[0];
            char close_char = seg.delimiter_end[0];

            while (end < text.size() && depth > 0) {
                if (open_char != '\0' && text[end] == open_char) depth++;
                else if (text[end] == close_char) {
                    depth--;
                    if (depth == 0) break;
                }
                end++;
            }
            if (depth == 0) {
                return std::string(text.substr(start, end - start));
            }
        }
    }

    return std::string(text.substr(start));
}

// Apply substitution template with extracted segments
std::string apply_substitution(const std::string& template_str, const std::vector<std::string>& segments) {
    std::string result = template_str;

    // Pre-process segments
    std::vector<std::string> processed_segments = segments;
    for (size_t i = 0; i < processed_segments.size(); ++i) {
        auto& seg = processed_segments[i];
        // Trim whitespace
        seg.erase(seg.begin(), std::find_if(seg.begin(), seg.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        seg.erase(std::find_if(seg.rbegin(), seg.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), seg.end());
        
        // Transform return type: "-> int" -> "int", empty -> "auto" (only for return type segment)
        if (i == 2) {  // return_type segment
            if (seg.starts_with("->")) {
                seg = seg.substr(2); // Remove "->"
                seg.erase(seg.begin(), std::find_if(seg.begin(), seg.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            }
            if (seg.empty()) {
                seg = "auto"; // Default return type
            }
        }
    }

    // Replace placeholders with processed segments
    // Support both 0-indexed ($0, $1, $2...) and 1-indexed ($1, $2, $3...)
    for (size_t i = 0; i < processed_segments.size(); ++i) {
        // Try 1-indexed first ($1, $2, ...)
        std::string placeholder_1 = "$" + std::to_string(i + 1);
        size_t pos = 0;
        while ((pos = result.find(placeholder_1, pos)) != std::string::npos) {
            result.replace(pos, placeholder_1.length(), processed_segments[i]);
            pos += processed_segments[i].length();
        }

        // Then try 0-indexed ($0, $1, ...)
        std::string placeholder_0 = "$" + std::to_string(i);
        pos = 0;
        while ((pos = result.find(placeholder_0, pos)) != std::string::npos) {
            result.replace(pos, placeholder_0.length(), processed_segments[i]);
            pos += processed_segments[i].length();
        }
    }

    return result;
}

// Apply recursive transformations to handle nested patterns
std::string apply_recursive_transformations(const std::string& input) {
    std::string result = input;
    
    // Transform := patterns: var := value -> auto var = value
    std::regex walrus_regex(R"(\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:=\s*([^;]+))");
    result = std::regex_replace(result, walrus_regex, "auto $1 = $2");
    
    // Transform : type = patterns: var : type = value -> type var = value
    std::regex typed_var_regex(R"(\b([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^=\s]+)\s*=\s*([^;]+))");
    result = std::regex_replace(result, typed_var_regex, "$2 $1 = $3");
    
    return result;
}

} // anonymous namespace

// Extract segments for alternating anchor/evidence patterns
std::vector<std::string> CPP2Emitter::extract_alternating_segments(std::string_view text, const PatternData& pattern) const {
    std::vector<std::string> segments;

    if (pattern.alternating_anchors.empty()) {
        return segments;
    }

    // Find the first anchor
    const std::string& first_anchor = pattern.alternating_anchors[0];
    size_t anchor_pos = text.find(first_anchor);
    if (anchor_pos == std::string::npos) {
        return segments;
    }

    // Special case: if only one anchor and 2 evidence spans, extract before AND after
    if (pattern.alternating_anchors.size() == 1 && pattern.evidence_types.size() == 2) {
        // Evidence before anchor
        std::string before = std::string(text.substr(0, anchor_pos));
        before.erase(before.begin(), std::find_if(before.begin(), before.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        before.erase(std::find_if(before.rbegin(), before.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), before.end());
        segments.push_back(before);

        // Evidence after anchor
        size_t after_start = anchor_pos + first_anchor.length();
        std::string after = std::string(text.substr(after_start));
        after.erase(after.begin(), std::find_if(after.begin(), after.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        after.erase(std::find_if(after.rbegin(), after.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), after.end());
        if (!after.empty() && after.back() == ';') {
            after.pop_back();
        }
        segments.push_back(after);

        return segments;
    }

    size_t current_pos = anchor_pos + first_anchor.length();

    // Extract evidence spans between anchors
    for (size_t i = 0; i < pattern.evidence_types.size(); ++i) {
        // Find next anchor or end
        size_t next_anchor_pos = std::string::npos;
        if (i + 1 < pattern.alternating_anchors.size()) {
            const std::string& next_anchor = pattern.alternating_anchors[i + 1];
            next_anchor_pos = text.find(next_anchor, current_pos);
        }

        size_t evidence_end = (next_anchor_pos != std::string::npos) ? next_anchor_pos : text.length();
        std::string evidence = std::string(text.substr(current_pos, evidence_end - current_pos));

        // Trim whitespace
        evidence.erase(evidence.begin(), std::find_if(evidence.begin(), evidence.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        evidence.erase(std::find_if(evidence.rbegin(), evidence.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), evidence.end());

        // Strip trailing semicolon if present (template will add it back)
        if (!evidence.empty() && evidence.back() == ';') {
            evidence.pop_back();
        }

        segments.push_back(evidence);
        current_pos = evidence_end;

        if (next_anchor_pos != std::string::npos) {
            current_pos += pattern.alternating_anchors[i + 1].length();
        }
    }

    return segments;
}

void CPP2Emitter::emit(OrbitIterator& iterator, std::string_view source, std::ostream& out, const std::vector<PatternData>& patterns) const {
    // Reset iterator to beginning
    iterator.reset();

    size_t last_pos = 0;

    // Iterate through all orbits and emit their results
    for (cppfort::stage0::Orbit* orbit = iterator.next(); orbit; orbit = iterator.next()) {
        if (auto* confix = dynamic_cast<cppfort::stage0::ConfixOrbit*>(orbit)) {
            // Fill gap between last position and current orbit
            if (last_pos < confix->start_pos) {
                out << source.substr(last_pos, confix->start_pos - last_pos);
            }

            // Find pattern for this orbit
            const PatternData* pattern = nullptr;
            std::string pattern_name = confix->selected_pattern();

            for (const auto& p : patterns) {
                if (p.name == pattern_name) {
                    pattern = &p;
                }
            }

            // Emit the orbit
            emit_orbit(*confix, source, out, pattern);
            last_pos = confix->end_pos;
        }
    }

    // Fill remaining text after last orbit
    if (last_pos < source.size()) {
        out << source.substr(last_pos);
    }
}

void CPP2Emitter::emit_fragment(const OrbitFragment& fragment, std::string_view source, std::ostream& out) const {
    // Deprecated - use emit_orbit instead
    out << extract_fragment_text(fragment, source);
}

void CPP2Emitter::emit_orbit(const ConfixOrbit& orbit, std::string_view source, std::ostream& out, const PatternData* pattern) const {
    if (orbit.start_pos >= source.size() || orbit.end_pos > source.size() || orbit.start_pos >= orbit.end_pos) {
        return;
    }

    std::string_view text = source.substr(orbit.start_pos, orbit.end_pos - orbit.start_pos);

    // ORBIT RECURSION MUST TERMINATE
    if (orbit.confidence == 0.0) {
        // For orbits with zero confidence, emit original text unchanged
        out << text;
        return;
    }

    // PATTERN-DRIVEN TRANSFORMATION
    if (!pattern) {
        // Pattern wasn't found for this orbit; emit original text and continue.
    std::cerr << "WARNING: Orbit matched but pattern '" << orbit.selected_pattern() << "' not found. Emitting original text.\n";
        out << text;
        return;
    }

    if (pattern->use_alternating) {
        if (pattern->alternating_anchors.empty() || pattern->evidence_types.empty() || pattern->substitution_templates.empty()) {
            std::cerr << "FATAL: Alternating pattern missing required fields\n";
            std::exit(1);
        }
    } else {
        if (pattern->segments.empty() || pattern->substitution_templates.empty()) {
            std::cerr << "FATAL: Segment-based pattern has no transformation data\n";
            std::exit(1);
        }
    }

    std::vector<std::string> segments;

    if (pattern->use_alternating) {
        // Extract evidence spans for alternating anchor/evidence pattern
        segments = extract_alternating_segments(text, *pattern);
    } else {
        // Find anchor (signature pattern) in text
        size_t anchor_pos = std::string::npos;
        for (const auto& sig : pattern->signature_patterns) {
            anchor_pos = text.find(sig);
            if (anchor_pos != std::string::npos) break;
        }

        if (anchor_pos == std::string::npos) {
            std::cerr << "FATAL: Anchor not found in matched text\n";
            std::exit(1);
        }

        // Extract segments using pattern definitions
        for (const auto& seg_def : pattern->segments) {
            std::string seg = extract_segment(text, anchor_pos, seg_def);
            segments.push_back(seg);
        }
    }

    // Get target grammar (CPP=2 for now)
    int target_grammar = 2;
    auto template_it = pattern->substitution_templates.find(target_grammar);
    if (template_it == pattern->substitution_templates.end()) {
        std::cerr << "FATAL: No substitution template for grammar " << target_grammar << "\n";
        std::exit(1);
    }

    // Apply substitution
    std::string result = apply_substitution(template_it->second, segments);
    out << result;
    
    // RECURSIVE TRANSFORMATION: scan result for nested patterns
    // Simple post-processing for := and : type = patterns
    std::string processed_result = apply_recursive_transformations(result);
    if (processed_result != result) {
        // Replace the output
        out.seekp(out.tellp() - std::streamoff(result.size()));
        out << processed_result;
    }
}

std::string_view CPP2Emitter::extract_fragment_text(const OrbitFragment& fragment, std::string_view source) const {
    if (fragment.start_pos >= source.size() || fragment.end_pos > source.size() ||
        fragment.start_pos >= fragment.end_pos) {
        return {};
    }
    return source.substr(fragment.start_pos, fragment.end_pos - fragment.start_pos);
}

} // namespace cppfort::stage0
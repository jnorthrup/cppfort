#include "cpp2_emitter.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string_view>

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
        // Find matching delimiter (handle nesting for braces/parens)
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

    return std::string(text.substr(start));
}

// Apply substitution template with extracted segments
std::string apply_substitution(const std::string& template_str, const std::vector<std::string>& segments) {
    std::string result = template_str;

    // Replace $0, $1, $2, etc. with extracted segments
    for (size_t i = 0; i < segments.size(); ++i) {
        std::string placeholder = "$" + std::to_string(i);
        size_t pos = 0;
        while ((pos = result.find(placeholder, pos)) != std::string::npos) {
            result.replace(pos, placeholder.length(), segments[i]);
            pos += segments[i].length();
        }
    }

    return result;
}

} // anonymous namespace

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
                    break;
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
        std::cerr << "FATAL: Orbit recursion failed at pos " << orbit.start_pos << "-" << orbit.end_pos << "\n";
        std::cerr << "Text: " << text << "\n";
        std::exit(1);
    }

    // PATTERN-DRIVEN TRANSFORMATION
    if (!pattern || pattern->segments.empty() || pattern->substitution_templates.empty()) {
        std::cerr << "FATAL: Orbit matched but pattern has no transformation data\n";
        std::cerr << "Pattern: " << orbit.selected_pattern() << "\n";
        std::exit(1);
    }

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
    std::vector<std::string> segments;
    for (const auto& seg_def : pattern->segments) {
        std::string seg = extract_segment(text, anchor_pos, seg_def);
        segments.push_back(seg);
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
}

std::string_view CPP2Emitter::extract_fragment_text(const OrbitFragment& fragment, std::string_view source) const {
    if (fragment.start_pos >= source.size() || fragment.end_pos > source.size() ||
        fragment.start_pos >= fragment.end_pos) {
        return {};
    }
    return source.substr(fragment.start_pos, fragment.end_pos - fragment.start_pos);
}

} // namespace cppfort::stage0
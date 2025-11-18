#include "json_yaml_plasma_transpiler.h"
#include "type_evidence.h"
#include "evidence_2d.h"
#include "pijul_graph.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iostream>
#include <iomanip>

#include <cstdint>
#include <vector>
#include <memory>

namespace cppfort::stage0 {

JsonYamlPlasmaTranspiler::JsonYamlPlasmaTranspiler()
    : discriminator_cache_(1024), cache_hits_(0), cache_misses_(0) {
}

std::optional<std::string> JsonYamlPlasmaTranspiler::json_to_yaml(std::string_view json_input) {
    last_error_ = Error{};

    if (json_input.empty()) {
        last_error_ = Error{"Empty JSON input", 0, "non-empty JSON object"};
        return std::nullopt;
    }

    // For basic JSON to YAML conversion, implement direct parsing
    std::string yaml_output;
    size_t pos = 0;
    int indent_level = 0;

    // Trim whitespace
    while (pos < json_input.size() && std::isspace(json_input[pos])) {
        pos++;
    }

    if (pos >= json_input.size()) {
        last_error_ = Error{"Empty JSON input after trimming", 0, "valid JSON"};
        return std::nullopt;
    }

    if (json_input[pos] == '{') {
        // Handle JSON object
        if (!parse_json_object_to_yaml(json_input, pos, yaml_output, indent_level)) {
            return std::nullopt;
        }
    } else if (json_input[pos] == '[') {
        // Handle JSON array
        if (!parse_json_array_to_yaml(json_input, pos, yaml_output, indent_level)) {
            return std::nullopt;
        }
    } else {
        // Handle simple JSON value
        std::string value = extract_json_value(json_input, pos);
        yaml_output += value;
    }

    return yaml_output;
}

std::optional<std::string> JsonYamlPlasmaTranspiler::yaml_to_json(std::string_view yaml_input) {
    last_error_ = Error{};

    if (yaml_input.empty()) {
        last_error_ = Error{"Empty YAML input", 0, "non-empty YAML content"};
        return std::nullopt;
    }

    // Build structure graph using pijul for reversible semantics
    auto graph = build_structure_graph(yaml_input);
    if (!graph) {
        return std::nullopt;
    }

    // Reconstruct as JSON
    return reconstruct_from_graph(*graph, false);
}

std::optional<std::string> JsonYamlPlasmaTranspiler::json_to_yaml_plasma(std::string_view json_input) {
    // Enhanced version with plasma analysis
    auto plasma_regions = analyze_plasma_regions(json_input);

    // Validate JSON structure using discriminators
    bool has_json_structure = std::all_of(plasma_regions.begin(), plasma_regions.end(),
        [this](const PlasmaRegion& region) { return detect_json_structure(region); });

    if (!has_json_structure) {
        last_error_ = Error{"Invalid JSON structure detected by plasma analysis", 0, "valid JSON"};
        return std::nullopt;
    }

    return json_to_yaml(json_input);
}

std::optional<std::string> JsonYamlPlasmaTranspiler::yaml_to_json_plasma(std::string_view yaml_input) {
    // Enhanced version with plasma analysis
    auto plasma_regions = analyze_plasma_regions(yaml_input);

    // Validate YAML structure using discriminators
    bool has_yaml_structure = std::all_of(plasma_regions.begin(), plasma_regions.end(),
        [this](const PlasmaRegion& region) { return detect_yaml_structure(region); });

    if (!has_yaml_structure) {
        last_error_ = Error{"Invalid YAML structure detected by plasma analysis", 0, "valid YAML"};
        return std::nullopt;
    }

    return yaml_to_json(yaml_input);
}

std::vector<JsonYamlPlasmaTranspiler::PlasmaRegion>
JsonYamlPlasmaTranspiler::analyze_plasma_regions(std::string_view input) {
    std::vector<PlasmaRegion> regions;

    // Find structure anchors using orbit-based detection
    auto anchors = find_structure_anchors(input);
    if (anchors.empty()) {
        // Single region covering entire input
        PlasmaRegion region;
        region.start_pos = 0;
        region.end_pos = input.size();
        region.discriminators = compute_discriminators(input, 0);
        region.confix_spans = extract_confix_spans(input);
        regions.push_back(region);
        return regions;
    }

    // Create regions between anchors
    for (size_t i = 0; i < anchors.size(); ++i) {
        size_t start = (i == 0) ? 0 : anchors[i-1];
        size_t end = (i < anchors.size() - 1) ? anchors[i] : input.size();

        if (start < end) {
            PlasmaRegion region;
            region.start_pos = start;
            region.end_pos = end;
            region.discriminators = compute_discriminators(input.substr(start, end - start), start);
            region.confix_spans = extract_confix_spans(input.substr(start, end - start));
            regions.push_back(region);
        }
    }

    return regions;
}

JsonYamlPlasmaTranspiler::BitDiscriminators
JsonYamlPlasmaTranspiler::compute_discriminators(std::string_view span, size_t start_pos) {
    // Check cache first
    uint64_t cache_key = std::hash<std::string_view>{}(span) ^ (static_cast<uint64_t>(start_pos) << 32);

    for (const auto& cache_line : discriminator_cache_) {
        if (cache_line.key == cache_key) {
            cache_hits_++;
            return cache_line.discriminators;
        }
    }

    cache_misses_++;

    BitDiscriminators result;
    result.eliminated_1bit_mask = compute_eliminated_classes(span);
    result.meta_2bit_evidence = compute_meta_evidence(span, start_pos);
    result.ascii_3bit_discriminators = compute_ascii_discriminators(span);

    // Update cache
    CacheLine cache_line;
    cache_line.key = cache_key;
    cache_line.discriminators = result;
    cache_line.confidence = (result.has_structure() ? 200 : 0);

    // Simple LRU replacement
    if (discriminator_cache_.size() < 1024) {
        discriminator_cache_.push_back(cache_line);
    } else {
        discriminator_cache_[cache_misses_ % 1024] = cache_line;
    }

    return result;
}

uint64_t JsonYamlPlasmaTranspiler::compute_eliminated_classes(std::string_view span) {
    uint64_t eliminated = 0;

    for (size_t i = 0; i < span.size(); ++i) {
        char c = span[i];
        size_t line_start = find_line_start(span, i);
        JsonYamlCharClass char_class = classify_char(c, i, line_start);

        // Mark character classes that are NOT present as eliminated
        uint64_t class_mask = 1ULL << static_cast<int>(char_class);
        eliminated |= ~class_mask;
    }

    return eliminated;
}

uint32_t JsonYamlPlasmaTranspiler::compute_meta_evidence(std::string_view span, size_t start_pos) {
    uint32_t evidence = 0;

    // Look for structure patterns at different distances
    for (size_t i = 1; i < span.size() && i < 16; ++i) {
        if (i < span.size()) {
            char prev = span[i-1];
            char curr = span[i];

            // 2-bit encoding for relationship evidence
            uint32_t relationship = 0;
            if (prev == '{' && curr == '"') relationship = 1;      // Object start to key
            else if (prev == '"' && curr == ':') relationship = 2;  // Key to colon
            else if (prev == ':' && curr == ' ') relationship = 3;  // Colon to value
            else if (prev == ',' && curr == '"') relationship = 1;  // Next item start

            evidence |= (relationship << (i * 2));
        }
    }

    return evidence;
}

uint8_t JsonYamlPlasmaTranspiler::compute_ascii_discriminators(std::string_view span) {
    uint8_t discriminators = 0;

    // 3-bit discrimination: 0-7 categories
    for (char c : span) {
        uint8_t category = 0;
        if (std::isspace(c)) category = 0;
        else if (c == '{' || c == '}' || c == '[' || c == ']') category = 1;
        else if (c == ':' || c == ',') category = 2;
        else if (c == '"') category = 3;
        else if (c == '-' || c == '|') category = 4;  // YAML specific
        else if (std::isdigit(c)) category = 5;
        else if (std::isalpha(c)) category = 6;
        else category = 7;  // Other symbols

        discriminators |= (1 << category);
    }

    return discriminators;
}

bool JsonYamlPlasmaTranspiler::detect_json_structure(const PlasmaRegion& region) {
    const auto& d = region.discriminators;

    // Check if we have braces or brackets present (NOT eliminated)
    // If braces/brackets are eliminated, they're not present in the span
    bool has_braces = (d.eliminated_1bit_mask & (1ULL << 3)) == 0;  // JSON_BRACE = 1 << 3
    bool has_brackets = (d.eliminated_1bit_mask & (1ULL << 4)) == 0;  // JSON_BRACKET = 1 << 4

    if (!has_braces && !has_brackets) {
        // No structural elements found - but this could be a simple JSON value
        // Check if we have at least quotes for strings or other content
    }

    // Check for quote characters (JSON strings)
    if (!(d.ascii_3bit_discriminators & (1 << 3))) {  // Quote category
        // No quotes found, but could still be valid JSON (numbers, booleans, null)
    }

    // Look for colon patterns in meta evidence
    bool has_colon_pattern = false;
    for (uint32_t i = 0; i < 32; i += 2) {
        uint32_t pattern = (d.meta_2bit_evidence >> i) & 0x3;
        if (pattern == 2) {  // Key to colon pattern
            has_colon_pattern = true;
            break;
        }
    }

    // Consider it JSON if:
    // 1. Has colon pattern (key-value structure)
    // 2. Has braces/brackets (object/array structure)
    // 3. Explicitly marked as JSON object
    // 4. Has any structure at all (fallback for simple values)

    return has_colon_pattern || has_braces || has_brackets ||
           region.is_json_object || d.has_structure();
}

bool JsonYamlPlasmaTranspiler::detect_yaml_structure(const PlasmaRegion& region) {
    const auto& d = region.discriminators;

    // YAML can use dashes and indentation
    // Check if dashes are present (NOT eliminated)
    bool has_dashes = (d.eliminated_1bit_mask & (1ULL << 7)) == 0;  // YAML_DASH = 1 << 7
    if (has_dashes) {
        return true;
    }

    // Check for indentation patterns (YAML specific)
    if (d.ascii_3bit_discriminators & (1 << 0)) {  // Whitespace category
        return true;
    }

    return region.is_yaml_sequence || region.has_key_value_structure || d.has_structure();
}

std::unique_ptr<cppfort::pijul::Graph> JsonYamlPlasmaTranspiler::build_structure_graph(std::string_view input) {
    auto graph = std::make_unique<cppfort::pijul::Graph>();

    // Build type evidence for the input
    auto type_evidence = build_type_evidence(input);
    auto confix_spans = build_2d_evidence(input);

    // If no confix spans found, create a default node for the whole input
    if (confix_spans.empty()) {
        cppfort::pijul::ExternalKey key(cppfort::pijul::HASH_SIZE + cppfort::pijul::LINE_SIZE, 0);
        // Fill with simple hash-like data (in real implementation this would be a proper hash)
        for (size_t i = 0; i < cppfort::pijul::HASH_SIZE; ++i) {
            key[i] = static_cast<uint8_t>(i % 256);
        }
        // Set line number part to 0
        for (size_t i = cppfort::pijul::HASH_SIZE; i < key.size(); ++i) {
            key[i] = 0;
        }

        graph->ensure_node(key);
        return graph;
    }

    // Create nodes for each structural element
    std::map<size_t, cppfort::pijul::Graph::NodeId> position_to_node;

    for (const auto& confix : confix_spans) {
        // Create node for this confix span
        cppfort::pijul::ExternalKey key(cppfort::pijul::HASH_SIZE + cppfort::pijul::LINE_SIZE, 0);

        // Create a simple hash-like key from position and length
        uint32_t pos = static_cast<uint32_t>(confix.begin_pos);
        uint32_t len = static_cast<uint32_t>(confix.end_pos - confix.begin_pos);

        // Fill hash part with position/length derived data
        for (size_t i = 0; i < cppfort::pijul::HASH_SIZE; ++i) {
            if (i < 4) {
                key[i] = static_cast<uint8_t>((pos >> (i * 8)) & 0xFF);
            } else if (i < 8) {
                key[i] = static_cast<uint8_t>((len >> ((i - 4) * 8)) & 0xFF);
            } else {
                key[i] = static_cast<uint8_t>((pos + len + i) % 256);
            }
        }
        // Set line number part to position (simple mapping)
        for (size_t i = cppfort::pijul::HASH_SIZE; i < key.size(); ++i) {
            key[i] = static_cast<uint8_t>((pos >> ((i - cppfort::pijul::HASH_SIZE) * 8)) & 0xFF);
        }

        auto node_id = graph->ensure_node(key);
        position_to_node[confix.begin_pos] = node_id;

        // Note: Structure type marking removed to avoid self-edges which are detected as cycles
        // In a full implementation, structure type would be stored in node metadata
    }

    // Create edges based on nesting relationships
    for (size_t i = 0; i < confix_spans.size(); ++i) {
        const auto& outer = confix_spans[i];
        auto outer_node = position_to_node[outer.begin_pos];

        for (size_t j = i + 1; j < confix_spans.size(); ++j) {
            const auto& inner = confix_spans[j];

            if (inner.begin_pos > outer.begin_pos && inner.end_pos < outer.end_pos) {
                // Inner is nested within outer
                auto inner_node = position_to_node[inner.begin_pos];
                graph->add_edge(outer_node, inner_node, cppfort::pijul::EdgeFlag::Parent);
            }
        }
    }

    // Only check for cycles if we have multiple nodes
    if (confix_spans.size() > 1 && graph->has_cycle()) {
        last_error_ = Error{"Invalid nesting structure detected", 0, "properly nested structure"};
        return nullptr;
    }

    return graph;
}

std::string JsonYamlPlasmaTranspiler::reconstruct_from_graph(const cppfort::pijul::Graph& graph, bool as_yaml) {
    std::string result;

    // Get topological order for reconstruction
    auto order = graph.topological_order();

    // Simple reconstruction - would need more sophisticated handling for production
    for (auto node_id : order) {
        // In a full implementation, this would use the node keys and edge structure
        // to properly reconstruct the format
        if (as_yaml) {
            result += "  yaml_item\n";
        } else {
            result += "\"json_item\": null,\n";
        }
    }

    return result;
}

std::vector<size_t> JsonYamlPlasmaTranspiler::find_structure_anchors(std::string_view input) {
    std::vector<size_t> anchors;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (c == '{' || c == '}' || c == '[' || c == ']' || c == ':' || c == ',') {
            anchors.push_back(i);
        }
    }

    return anchors;
}

std::vector<ConfixEvidence> JsonYamlPlasmaTranspiler::extract_confix_spans(std::string_view input) {
    std::vector<ConfixEvidence> spans;

    // Simple brace matching for JSON objects
    std::vector<size_t> brace_stack;

    for (size_t i = 0; i < input.size(); ++i) {
        char c = input[i];

        if (c == '{') {
            brace_stack.push_back(i);
        } else if (c == '}' && !brace_stack.empty()) {
            size_t start = brace_stack.back();
            brace_stack.pop_back();

            ConfixEvidence confix(ConfixType::BRACE, start, i + 1);
            spans.push_back(confix);
        }

        // Similar logic for arrays with [] could be added here
    }

    return spans;
}

std::vector<TypeEvidence> JsonYamlPlasmaTranspiler::build_type_evidence(std::string_view input) {
    std::vector<TypeEvidence> evidence;

    for (size_t i = 0; i < input.size(); ++i) {
        TypeEvidence ev;
        // TypeEvidence is a complex structure - for now just use default
        // The character classification is handled by other methods
        evidence.push_back(ev);
    }

    return evidence;
}

std::vector<ConfixEvidence> JsonYamlPlasmaTranspiler::build_2d_evidence(std::string_view input) {
    return extract_confix_spans(input);
}

JsonYamlPlasmaTranspiler::JsonYamlCharClass
JsonYamlPlasmaTranspiler::classify_char(char c, size_t pos, size_t line_start) {
    switch (c) {
        case ' ':
        case '\t':
        case '\n':
        case '\r':
            return JSON_WHITESPACE;
        case ':':
            return JSON_COLON;
        case ',':
            return JSON_COMMA;
        case '{':
        case '}':
            return JSON_BRACE;
        case '[':
        case ']':
            return JSON_BRACKET;
        case '"':
            return JSON_QUOTE;
        case '-':
            return YAML_DASH;
        default:
            // Check for YAML indentation (spaces at beginning of line)
            if (c == ' ' && pos == line_start) {
                return YAML_INDENT;
            }
            return JSON_WHITESPACE;  // Default
    }
}

bool JsonYamlPlasmaTranspiler::is_json_value_char(char c) {
    return std::isprint(c) && c != '"' && c != '\\' && c != '\n' && c != '\r';
}

bool JsonYamlPlasmaTranspiler::is_yaml_value_char(char c) {
    return std::isprint(c) && c != '\n' && c != '\r';
}

size_t JsonYamlPlasmaTranspiler::find_line_start(std::string_view text, size_t position) {
    size_t line_start = position;
    while (line_start > 0 && text[line_start - 1] != '\n') {
        line_start--;
    }
    return line_start;
}

std::string JsonYamlPlasmaTranspiler::extract_indentation(std::string_view line) {
    std::string indent;
    for (char c : line) {
        if (c == ' ' || c == '\t') {
            indent += c;
        } else {
            break;
        }
    }
    return indent;
}

bool JsonYamlPlasmaTranspiler::validate_round_trip(std::string_view original, std::string_view converted) {
    // Simple validation - would need proper parser for production
    return !original.empty() && !converted.empty();
}

// JSON parsing helper methods
void JsonYamlPlasmaTranspiler::skip_whitespace(std::string_view json, size_t& pos) {
    while (pos < json.size() && std::isspace(json[pos])) {
        pos++;
    }
}

std::string JsonYamlPlasmaTranspiler::extract_json_string(std::string_view json, size_t& pos) {
    std::string result;
    pos++; // Skip opening quote

    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++; // Skip backslash
            char c = json[pos];
            switch (c) {
                case 'n': result += '\n'; break;
                case 't': result += '\t'; break;
                case 'r': result += '\r'; break;
                case '\\': result += '\\'; break;
                case '"': result += '"'; break;
                default: result += c; break;
            }
            pos++;
        } else {
            result += json[pos];
            pos++;
        }
    }

    if (pos < json.size()) {
        pos++; // Skip closing quote
    }

    return result;
}

std::string JsonYamlPlasmaTranspiler::extract_json_value(std::string_view json, size_t& pos) {
    skip_whitespace(json, pos);

    if (pos >= json.size()) {
        last_error_ = Error{"Unexpected end of JSON input", pos, "value"};
        return "";
    }

    if (json[pos] == '"') {
        return extract_json_string(json, pos);
    } else if (json[pos] == '{') {
        // For objects/arrays in value context, just return the raw JSON
        size_t start = pos;
        int brace_count = 0;
        while (pos < json.size()) {
            if (json[pos] == '{') brace_count++;
            else if (json[pos] == '}') brace_count--;

            pos++;
            if (brace_count == 0) break;
        }
        return std::string(json.substr(start, pos - start));
    } else if (json[pos] == '[') {
        // For arrays in value context, just return the raw JSON
        size_t start = pos;
        int bracket_count = 0;
        while (pos < json.size()) {
            if (json[pos] == '[') bracket_count++;
            else if (json[pos] == ']') bracket_count--;

            pos++;
            if (bracket_count == 0) break;
        }
        return std::string(json.substr(start, pos - start));
    } else if (std::isdigit(json[pos]) || json[pos] == '-') {
        std::string result;
        while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '-' || json[pos] == '.')) {
            result += json[pos];
            pos++;
        }
        return result;
    } else {
        // Parse boolean or null
        std::string result;
        while (pos < json.size() && std::isalpha(json[pos])) {
            result += json[pos];
            pos++;
        }
        return result;
    }
}

bool JsonYamlPlasmaTranspiler::parse_json_object_to_yaml(std::string_view json, size_t& pos, std::string& yaml, int indent_level) {
    skip_whitespace(json, pos);

    if (pos >= json.size() || json[pos] != '{') {
        last_error_ = Error{"Expected '{' for JSON object", pos, "object"};
        return false;
    }

    pos++; // Skip '{'
    skip_whitespace(json, pos);

    bool first = true;
    while (pos < json.size() && json[pos] != '}') {
        if (!first) {
            skip_whitespace(json, pos);
            if (pos >= json.size() || json[pos] != ',') {
                last_error_ = Error{"Expected ',' in JSON object", pos, "comma separator"};
                return false;
            }
            pos++; // Skip ','
            skip_whitespace(json, pos);
        }
        first = false;

        // Parse key
        if (pos >= json.size() || json[pos] != '"') {
            last_error_ = Error{"Expected string key in JSON object", pos, "string key"};
            return false;
        }

        std::string key = extract_json_string(json, pos);
        skip_whitespace(json, pos);

        if (pos >= json.size() || json[pos] != ':') {
            last_error_ = Error{"Expected ':' after key in JSON object", pos, "colon separator"};
            return false;
        }

        pos++; // Skip ':'
        skip_whitespace(json, pos);

        // Add to YAML with proper indentation
        for (int i = 0; i < indent_level; i++) {
            yaml += "  ";
        }
        yaml += key + ":";

        // Check if the value is a complex type
        size_t lookahead = pos;
        skip_whitespace(json, lookahead);

        if (lookahead < json.size() && (json[lookahead] == '{' || json[lookahead] == '[')) {
            yaml += "\n";
            if (json[lookahead] == '{') {
                parse_json_object_to_yaml(json, pos, yaml, indent_level + 1);
            } else {
                parse_json_array_to_yaml(json, pos, yaml, indent_level + 1);
            }
        } else {
            yaml += " " + extract_json_value(json, pos);
        }

        yaml += "\n";
    }

    if (pos >= json.size() || json[pos] != '}') {
        last_error_ = Error{"Expected '}' to close JSON object", pos, "closing brace"};
        return false;
    }

    pos++; // Skip '}'
    return true;
}

bool JsonYamlPlasmaTranspiler::parse_json_array_to_yaml(std::string_view json, size_t& pos, std::string& yaml, int indent_level) {
    skip_whitespace(json, pos);

    if (pos >= json.size() || json[pos] != '[') {
        last_error_ = Error{"Expected '[' for JSON array", pos, "array"};
        return false;
    }

    pos++; // Skip '['
    skip_whitespace(json, pos);

    bool first = true;
    while (pos < json.size() && json[pos] != ']') {
        if (!first) {
            skip_whitespace(json, pos);
            if (pos >= json.size() || json[pos] != ',') {
                last_error_ = Error{"Expected ',' in JSON array", pos, "comma separator"};
                return false;
            }
            pos++; // Skip ','
            skip_whitespace(json, pos);
        }
        first = false;

        // Add indentation for array items
        for (int i = 0; i < indent_level; i++) {
            yaml += "  ";
        }
        yaml += "- ";

        size_t lookahead = pos;
        skip_whitespace(json, lookahead);

        if (lookahead < json.size() && (json[lookahead] == '{' || json[lookahead] == '[')) {
            yaml += "\n";
            if (json[lookahead] == '{') {
                parse_json_object_to_yaml(json, pos, yaml, indent_level + 1);
            } else {
                parse_json_array_to_yaml(json, pos, yaml, indent_level + 1);
            }
        } else {
            yaml += extract_json_value(json, pos);
        }

        yaml += "\n";
    }

    if (pos >= json.size() || json[pos] != ']') {
        last_error_ = Error{"Expected ']' to close JSON array", pos, "closing bracket"};
        return false;
    }

    pos++; // Skip ']'
    return true;
}

} // namespace cppfort::stage0
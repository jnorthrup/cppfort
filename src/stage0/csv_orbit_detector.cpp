#include "csv_orbit_detector.h"
#include <cctype>
#include <algorithm>
#include <sstream>
#include <iostream>

namespace cppfort::stage0 {

CSVSchema CSVOrbitDetector::detect_schema_from_memory(std::string_view file_contents) {
    CSVSchema schema;
    
    // Split by rows (orbit boundaries)
    std::vector<std::string_view> rows;
    size_t pos = 0;
    while (pos < file_contents.size()) {
        size_t newline = file_contents.find('\n', pos);
        if (newline == std::string_view::npos) {
            rows.push_back(file_contents.substr(pos));
            break;
        }
        rows.push_back(file_contents.substr(pos, newline - pos));
        pos = newline + 1;
    }
    
    schema.row_count = rows.size();
    if (rows.empty()) return schema;
    
    // Process each row with orbit evidence accumulation
    for (size_t i = 0; i < rows.size(); ++i) {
        auto field_types = process_row(rows[i], i, schema.column_count);
        
        if (i == 0) {
            // First row - initialize schema
            schema.column_count = field_types.size();
            column_schemas.resize(schema.column_count);
            for (size_t j = 0; j < schema.column_count; ++j) {
                column_schemas[j].field_index = j;
                column_schemas[j].detected_type = field_types[j];
            }
        } else {
            // Subsequent rows - validate consistency
            if (field_types.size() != schema.column_count) {
                schema.is_well_formed = false;
                // Apply wobbling to fix column count mismatches
                HeuristicOrbit orbit;
                orbit.window_start = 0;
                orbit.window_end = rows[i].size();
                if (wobble_field_boundaries(rows[i], 0, orbit)) {
                    // Retry with wobbled boundaries
                    field_types = process_row(rows[i], i, schema.column_count);
                }
            }
            
            // Accumulate evidence per column
            for (size_t j = 0; j < std::min(field_types.size(), schema.column_count); ++j) {
                column_schemas[j].row_count++;
                
                // Check type consistency
                if (field_types[j] != column_schemas[j].detected_type && 
                    column_schemas[j].detected_type != CSVFieldEvidence::FieldType::STRING) {
                    // Type changed - downgrade to STRING which is most general
                    column_schemas[j].detected_type = CSVFieldEvidence::FieldType::STRING;
                    column_schemas[j].is_consistent = false;
                    column_schemas[j].confidence *= 0.8; // Penalty for inconsistency
                }
            }
        }
    }
    
    // Finalize schema with terminal evidence
    schema.fields = column_schemas;
    
    // Calculate overall confidence
    double total_confidence = 0.0;
    for (const auto& field : schema.fields) {
        total_confidence += field.confidence;
    }
    schema.overall_confidence = schema.fields.empty() ? 0.0 : 
                                 total_confidence / schema.fields.size();
    
    return schema;
}

std::vector<CSVFieldEvidence::FieldType> CSVOrbitDetector::process_row(
    std::string_view row, 
    size_t row_index,
    size_t expected_columns) {
    
    std::vector<CSVFieldEvidence::FieldType> field_types;
    orbit_evidence.reset(); // Reset for new row
    
    size_t pos = 0;
    size_t field_start = 0;
    bool in_quotes = false;
    int paren_depth = 0; // Track nested parentheses
    
    while (pos <= row.size()) {
        char ch = (pos < row.size()) ? row[pos] : '\0';
        
        // Observe character with orbit awareness
        if (pos < row.size()) {
            orbit_evidence.observe_char_with_orbit(ch, pos);
        }
        
        // Track quote state for CSV parsing
        if (ch == '"' && (pos == 0 || row[pos-1] != '\\')) {
            in_quotes = !in_quotes;
            orbit_evidence.observe_confix_with_orbit(
                ConfixType::STRING_DOUBLE, in_quotes, pos);
        }
        
        // Track parentheses depth
        if (!in_quotes && ch == '(') {
            paren_depth++;
            orbit_evidence.observe_confix_with_orbit(
                ConfixType::PAREN, true, pos);
        } else if (!in_quotes && ch == ')') {
            paren_depth--;
            orbit_evidence.observe_confix_with_orbit(
                ConfixType::PAREN, false, pos);
        }
        
        // Field boundary detection (considering quotes and nesting)
        bool is_boundary = false;
        if (!in_quotes && paren_depth == 0) {
            if (ch == ',' || pos == row.size()) {
                is_boundary = true;
            }
        }
        
        if (is_boundary) {
            // Extract field value
            std::string_view field_value = row.substr(field_start, pos - field_start);
            
            // Trim whitespace
            while (!field_value.empty() && std::isspace(field_value.front())) {
                field_value.remove_prefix(1);
            }
            while (!field_value.empty() && std::isspace(field_value.back())) {
                field_value.remove_suffix(1);
            }
            
            // Finalize current evidence span
            EvidenceSpan span(field_start, pos, field_value);
            orbit_evidence.accumulate_span(span);
            
            // Classify field type using orbit evidence
            CSVFieldEvidence::FieldType field_type = classify_field_type(
                field_value, span.evidence);
            field_types.push_back(field_type);
            
            field_start = pos + 1; // Next field starts after comma
            orbit_evidence.reset(); // Reset for next field
        }
        
        pos++;
    }
    
    // Handle case where row ends with empty field
    if (field_start <= row.size() && field_start > 0) {
        if (row.back() == ',') {
            // Empty trailing field
            EvidenceSpan span(row.size(), row.size(), "");
            orbit_evidence.accumulate_span(span);
            field_types.push_back(CSVFieldEvidence::FieldType::STRING);
        }
    }
    
    return field_types;
}

CSVFieldEvidence::FieldType CSVOrbitDetector::classify_field_type(
    std::string_view field_value,
    const TypeEvidence& evidence) {
    
    // Check if in center (perfectly balanced, fixed type)
    if (is_in_center(evidence)) {
        // Center elements - strongly typed data
        if (evidence.number_literals > 0 && evidence.digits > (field_value.size() / 2)) {
            // Mostly digits - numeric type
            return CSVFieldEvidence::FieldType::NUMERIC;
        }
        if (evidence.truefalse > 0) {
            // Contains true/false - boolean type
            return CSVFieldEvidence::FieldType::BOOLEAN;
        }
    }
    
    // Orbit-based classification
    if (orbit_evidence.is_in_center(evidence)) {
        // Balanced evidence - check orbit type
        auto orbit_type = orbit_evidence.get_current_orbit();
        
        switch (orbit_type) {
            case QuaternionOrbitEvidence::OrbitType::ORBIT_I:
                // Template/numeric context
                if (evidence.digits > 0 && evidence.alpha == 0) {
                    return CSVFieldEvidence::FieldType::NUMERIC;
                }
                break;
                
            case QuaternionOrbitEvidence::OrbitType::ORBIT_J:
                // Function/scope context  
                return CSVFieldEvidence::FieldType::STRING; // Default to string
                
            case QuaternionOrbitEvidence::OrbitType::ORBIT_K:
                // Expression/operator context
                if (evidence.arrow > 0 || evidence.colon > 0) {
                    // Date-like separators
                    return CSVFieldEvidence::FieldType::DATE;
                }
                break;
                
            case QuaternionOrbitEvidence::OrbitType::CENTER_FIXED:
            case QuaternionOrbitEvidence::OrbitType::CENTER_NEG:
                // Fixed point - strongly typed
                if (evidence.digits == field_value.size()) {
                    return CSVFieldEvidence::FieldType::NUMERIC;
                }
                break;
                
            default:
                break;
        }
    }
    
    // Heuristic fallback based on character composition
    if (field_value.empty()) {
        return CSVFieldEvidence::FieldType::STRING; // Empty treated as string
    }
    
    bool all_numeric = true;
    bool has_decimal = false;
    
    for (char c : field_value) {
        if (!std::isdigit(c) && c != '.' && c != '-' && c != '+') {
            all_numeric = false;
            break;
        }
        if (c == '.') {
            has_decimal = true;
        }
    }
    
    if (all_numeric && evidence.digits >= field_value.size() / 2) {
        return CSVFieldEvidence::FieldType::NUMERIC;
    }
    
    // Check for date patterns (YYYY-MM-DD, etc.)
    if (evidence.digits >= 6 && evidence.dashes > 0) {
        return CSVFieldEvidence::FieldType::DATE;
    }
    
    // Default to string for safety
    return CSVFieldEvidence::FieldType::STRING;
}

bool CSVOrbitDetector::wobble_field_boundaries(
    std::string_view row,
    size_t start_pos,
    HeuristicOrbit& orbit) {
    
    // Wobbling: slide window to find valid field boundaries
    // Try expanding/shrinking evidence window
    
    size_t window_size = orbit.window_end - orbit.window_start;
    const size_t MAX_WOBBLE = 3; // Maximum adjustments
    
    while (orbit.wobble_count < MAX_WOBBLE) {
        // Try expanding right
        if (orbit.window_end < row.size()) {
            size_t new_end = orbit.window_end + 1;
            std::string_view extended = row.substr(orbit.window_start, new_end - orbit.window_start);
            
            // Re-accumulate evidence for extended window
            QuaternionOrbitEvidence test_evidence;
            for (size_t i = orbit.window_start; i < new_end && i < row.size(); ++i) {
                test_evidence.observe_char_with_orbit(row[i], i);
            }
            
            // Check if extended window is in center (balanced)
            if (test_evidence.is_in_center(test_evidence.get_current_evidence())) {
                orbit.window_end = new_end;
                orbit.wobble_count++;
                orbit.confidence *= 0.95; // Small penalty for wobbling
                continue; // Success, try more expansion
            }
        }
        
        // Try shrinking from left
        if (orbit.window_end - orbit.window_start > 2) {
            size_t new_start = orbit.window_start + 1;
            std::string_view shrunk = row.substr(new_start, orbit.window_end - new_start);
            
            // Re-accumulate evidence for shrunken window
            QuaternionOrbitEvidence test_evidence;
            for (size_t i = new_start; i < orbit.window_end && i < row.size(); ++i) {
                test_evidence.observe_char_with_orbit(row[i], i);
            }
            
            if (test_evidence.is_in_center(test_evidence.get_current_evidence())) {
                orbit.window_start = new_start;
                orbit.wobble_count++;
                orbit.confidence *= 0.95; // Small penalty for wobbling
                continue; // Success, try more shrinking
            }
        }
        
        break; // No more improvements
    }
    
    return orbit.wobble_count > 0; // Return true if we wobbled
}

} // namespace cppfort::stage0
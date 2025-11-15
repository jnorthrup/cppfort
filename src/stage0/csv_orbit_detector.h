#pragma once

#include "type_evidence.h"
#include "quaternion_evidence.h"
#include <vector>
#include <string>
#include <string_view>
#include <optional>

namespace cppfort::stage0 {

/**
 * CSV Schema Detection using Quaternion Orbit Evidence
 * 
 * Maps CSV parsing to Q₈ group actions:
 * - Balanced rows ↔ Center Z(Q₈) 
 * - Field types ↔ Orbit types (I=numeric, J=string, K=date)
 * - Sliding window ↔ Orbit conjugation action
 * - Terminal evidence ↔ Final schema after full file scan
 */
struct CSVFieldEvidence {
    enum class FieldType {
        UNKNOWN = 0,
        STRING = 1,    // Orbit J - default/string context
        NUMERIC = 2,   // Orbit I - numeric/template context  
        DATE = 3,      // Orbit K - expression/operator context
        BOOLEAN = 4    // Center - fixed type
    };
    
    FieldType detected_type = FieldType::UNKNOWN;
    TypeEvidence evidence;              // Accumulated evidence for this field
    size_t field_index = 0;             // Column position
    size_t row_count = 0;               // Rows processed
    double confidence = 0.0;            // Schema confidence (0.0-1.0)
    bool is_consistent = true;          // Type consistency across rows
    
    CSVFieldEvidence() = default;
};

struct CSVSchema {
    std::vector<CSVFieldEvidence> fields;  // Schema per column
    size_t row_count = 0;
    size_t column_count = 0;
    double overall_confidence = 0.0;
    bool is_well_formed = true;            // All rows have same column count
};

class CSVOrbitDetector {
private:
    // Q₈ orbit evidence accumulator
    QuaternionOrbitEvidence orbit_evidence;
    
    // Schema building across rows
    std::vector<CSVFieldEvidence> column_schemas;
    
    // Terminal evidence after full file scan
    TypeEvidence terminal_evidence;
    
    // Heuristic orbits for sliding window (wobbling support)
    struct HeuristicOrbit {
        size_t window_start = 0;
        size_t window_end = 0;
        CSVFieldEvidence::FieldType detected_type;
        double confidence = 0.0;
        uint16_t wobble_count = 0;  // Track window adjustments
    };
    std::vector<HeuristicOrbit> heuristic_orbits;
    
    // CSV delimiters as orbit anchors
    static constexpr char FIELD_DELIM = ',';
    static constexpr char ROW_DELIM = '\n';
    static constexpr char QUOTE_CHAR = '"';
    
public:
    CSVOrbitDetector() = default;
    
    /**
     * Detect CSV schema by scanning entire file with mmap
     * Creates terminal evidence through orbit accumulation
     */
    CSVSchema detect_schema_from_memory(std::string_view file_contents);
    
    /**
     * Process single row with orbit evidence accumulation
     * Returns detected field types for this row
     */
    std::vector<CSVFieldEvidence::FieldType> process_row(
        std::string_view row, 
        size_t row_index,
        size_t expected_columns = 0);
        
    /**
     * Apply heuristic orbit sliding to malformed CSV
     * Wobbles evidence window to find valid boundaries
     */
    bool wobble_field_boundaries(
        std::string_view row,
        size_t start_pos,
        HeuristicOrbit& orbit);
    
    /**
     * Get accumulated terminal evidence
     * Used for final schema validation
     */
    const TypeEvidence& get_terminal_evidence() const {
        return terminal_evidence;
    }
    
private:
    /**
     * Classify field type using quaternion orbit types
     * Evidence selection based on group action principles
     */
    CSVFieldEvidence::FieldType classify_field_type(
        std::string_view field_value,
        const TypeEvidence& evidence);
        
    /**
     * Check if field is in center (balanced, fixed type)
     * Well-formed data typically stays in center
     */
    bool is_in_center(const TypeEvidence& evidence) const {
        return evidence.all_confix_balanced() && 
               evidence.comma < 2 &&     // Not overly nested
               evidence.dquotes % 2 == 0; // Balanced quotes
    }
    
    /**
     * Apply conjugation action to transform field type
     * Simulates how evidence changes under group action
     */
    TypeEvidence conjugate_field_evidence(
        const TypeEvidence& evidence,
        uint8_t action);
};

} // namespace cppfort::stage0
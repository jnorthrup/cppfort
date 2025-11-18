#pragma once

#include "type_evidence.h"
#include "evidence_2d.h"
#include "pijul_graph.h"
// #include "pijul_orbit_builder.h"  // Commented out due to conflicts
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <memory>

namespace cppfort::stage0 {

/**
 * JSON/YAML Plasma Transpiler
 *
 * Uses bit-level discriminators and evidence spans for JSON/YAML structure detection,
 * implementing the numerical plasma sifter approach from the architecture.
 *
 * Discriminators:
 * - 1-bit: eliminated class memoization for cache line retention
 * - 2-bit: medium-range meta evidence spans
 * - 3-bit: ASCII character discrimination
 *
 * Reversible graph semantics via pijul for round-trip conversion.
 */
class JsonYamlPlasmaTranspiler {
public:
    JsonYamlPlasmaTranspiler();
    ~JsonYamlPlasmaTranspiler() = default;

    // Main transpiler interface
    std::optional<std::string> json_to_yaml(std::string_view json_input);
    std::optional<std::string> yaml_to_json(std::string_view yaml_input);

    // Bit-discriminator analysis
    struct BitDiscriminators {
        uint64_t eliminated_1bit_mask = 0;      // Classes eliminated from this span
        uint32_t meta_2bit_evidence = 0;        // 2-bit meta evidence spans
        uint8_t ascii_3bit_discriminators = 0;  // 3-bit ASCII discrimination

        bool has_structure() const {
            return eliminated_1bit_mask != 0 ||
                   meta_2bit_evidence != 0 ||
                   ascii_3bit_discriminators != 0;
        }
    };

    // Plasma sifter results
    struct PlasmaRegion {
        size_t start_pos;
        size_t end_pos;
        BitDiscriminators discriminators;
        std::vector<ConfixEvidence> confix_spans;
        uint8_t confidence = 0;

        // JSON/YAML specific structure hints
        bool is_json_object = false;
        bool is_yaml_sequence = false;
        bool has_key_value_structure = false;
    };

    // Convert with plasma analysis
    std::optional<std::string> json_to_yaml_plasma(std::string_view json_input);
    std::optional<std::string> yaml_to_json_plasma(std::string_view yaml_input);

    // Error reporting
    struct Error {
        std::string message;
        size_t position = 0;
        std::string expected_structure;
    };

    const Error& last_error() const { return last_error_; }
    bool has_error() const { return !last_error_.message.empty(); }

private:
    // Plasma sifter core algorithms
    std::vector<PlasmaRegion> analyze_plasma_regions(std::string_view input);
    BitDiscriminators compute_discriminators(std::string_view span, size_t start_pos);

    // Character class elimination (1-bit memoization)
    uint64_t compute_eliminated_classes(std::string_view span);

    // Meta evidence spans (2-bit)
    uint32_t compute_meta_evidence(std::string_view span, size_t start_pos);

    // ASCII discrimination (3-bit)
    uint8_t compute_ascii_discriminators(std::string_view span);

    // Structure detection using discriminators
    bool detect_json_structure(const PlasmaRegion& region);
    bool detect_yaml_structure(const PlasmaRegion& region);

    // Pijul graph for reversible semantics
    std::unique_ptr<cppfort::pijul::Graph> build_structure_graph(std::string_view input);
    std::string reconstruct_from_graph(const cppfort::pijul::Graph& graph, bool as_yaml);

    // Orbit-based anchor detection for JSON/YAML
    std::vector<size_t> find_structure_anchors(std::string_view input);
    std::vector<ConfixEvidence> extract_confix_spans(std::string_view input);

    // Evidence span processing
    std::vector<TypeEvidence> build_type_evidence(std::string_view input);
    std::vector<ConfixEvidence> build_2d_evidence(std::string_view input);

    // JSON/YAML specific character classes
    enum JsonYamlCharClass {
        JSON_WHITESPACE = 1 << 0,
        JSON_COLON = 1 << 1,
        JSON_COMMA = 1 << 2,
        JSON_BRACE = 1 << 3,
        JSON_BRACKET = 1 << 4,
        JSON_QUOTE = 1 << 5,
        YAML_INDENT = 1 << 6,
        YAML_DASH = 1 << 7,
        YAML_COLON = 1 << 8,
    };

    // Character classification for discriminators
    JsonYamlCharClass classify_char(char c, size_t pos, size_t line_start);

    // Cache optimization
    struct CacheLine {
        uint64_t key;
        BitDiscriminators discriminators;
        uint8_t confidence;
    };

    std::vector<CacheLine> discriminator_cache_;
    uint64_t cache_hits_ = 0;
    uint64_t cache_misses_ = 0;

    // State
    Error last_error_;
    bool strict_mode_ = false;

    // Helper methods
    bool is_json_value_char(char c);
    bool is_yaml_value_char(char c);
    size_t find_line_start(std::string_view text, size_t position);
    std::string extract_indentation(std::string_view line);

    // Round-trip validation using pijul graph
    bool validate_round_trip(std::string_view original, std::string_view converted);

    // JSON parsing helpers for direct conversion
    bool parse_json_object_to_yaml(std::string_view json, size_t& pos, std::string& yaml, int indent_level);
    bool parse_json_array_to_yaml(std::string_view json, size_t& pos, std::string& yaml, int indent_level);
    std::string extract_json_value(std::string_view json, size_t& pos);
    std::string extract_json_string(std::string_view json, size_t& pos);
    void skip_whitespace(std::string_view json, size_t& pos);
};

} // namespace cppfort::stage0
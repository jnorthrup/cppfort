#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <algorithm>

namespace cppfort::stage0 {

// ═══════════════════════════════════════════════════════════════════════════════════════
// TYPEALIAS FORWARD DESIGN (TrikeShed Pattern)
// ═══════════════════════════════════════════════════════════════════════════════════════

template<typename A, typename B>
using Join = std::pair<A, B>;

// YamlDocument = Join<YamlStructuralBitmap, YamlContentIndex>
using YamlStructuralBitmap = Join<std::vector<uint64_t>, std::vector<int32_t>>; // bitmap + type evidence
using YamlContentIndex = Join<std::vector<int32_t>, std::string_view>; // positions + content
using YamlDocument = Join<YamlStructuralBitmap, YamlContentIndex>;

// ═══════════════════════════════════════════════════════════════════════════════════════
// YAML TYPE EVIDENCE
// ═══════════════════════════════════════════════════════════════════════════════════════

enum class YamlType : int32_t {
    Invalid = 0,
    Mapping = 1,
    Sequence = 2,
    String = 3,
    Number = 4,
    Boolean = 5,
    Null = 6,
    KeySeparator = 7,  // ':'
    ItemSeparator = 8, // '-'
    Indentation = 9,
    DocumentStart = 10, // '---'
    DocumentEnd = 11    // '...'
};

struct YamlTypeEvidence {
    YamlType type;
    int32_t depth;
    int32_t indent_level;
    
    YamlTypeEvidence(YamlType t = YamlType::Invalid, int32_t d = 0, int32_t indent = 0) 
        : type(t), depth(d), indent_level(indent) {}
};

struct YamlTypedValue {
    YamlTypeEvidence evidence;
    std::string value;
    int32_t line_number;
    int32_t column_number;
};

// ═══════════════════════════════════════════════════════════════════════════════════════
// YAML SCANNER INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════════════

class YamlScanner {
public:
    struct YamlEvent {
        size_t position;
        YamlTypeEvidence evidence;
        char character;
        bool is_structural;
        
        YamlEvent(size_t pos = 0, YamlTypeEvidence ev = {}, char ch = 0, bool structural = false)
            : position(pos), evidence(ev), character(ch), is_structural(structural) {}
    };

    // Parse YAML document and return typed values with evidence
    static std::vector<YamlTypedValue> parseYamlDocument(
        std::string_view source,
        YamlDocument& out_document
    );

    // Scan for structural elements (mappings, sequences, scalars)
    static std::vector<YamlEvent> scanYamlStructure(
        std::string_view source
    );

    // Extract key-value pairs with type evidence
    static std::unordered_map<std::string, YamlTypedValue> extractMapping(
        std::string_view source,
        size_t start_position = 0,
        int32_t indent_level = 0
    );

    // Extract sequence items with type evidence
    static std::vector<YamlTypedValue> extractSequence(
        std::string_view source,
        size_t start_position = 0,
        int32_t indent_level = 0
    );

    // Detect YAML document boundaries (--- and ...)
    static std::vector<size_t> findDocumentBoundaries(
        std::string_view source
    );

    // Calculate indentation level for a given position
    static int32_t calculateIndentation(
        std::string_view source,
        size_t position
    );

private:
    // Helper: Detect if character at position is part of a YAML key
    static bool isYamlKey(
        std::string_view source,
        size_t position
    );

    // Helper: Extract scalar value with proper type detection
    static YamlTypedValue parseYamlScalar(
        std::string_view source,
        size_t start_position
    );

    // Helper: Check if line is a comment
    static bool isCommentLine(
        std::string_view source,
        size_t line_start
    );

    // SIMD-accelerated scanning for YAML structural characters
    static std::vector<YamlEvent> scanYamlSIMD(
        std::string_view source
    );
};

} // namespace cppfort::stage0

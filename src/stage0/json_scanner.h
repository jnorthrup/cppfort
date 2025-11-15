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

// JsonDocument = Join<JsonStructuralBitmap, JsonContentIndex>
using JsonStructuralBitmap = Join<std::vector<uint64_t>, std::vector<int32_t>>; // bitmap + type evidence
using JsonContentIndex = Join<std::vector<int32_t>, std::string_view>; // positions + content
using JsonDocument = Join<JsonStructuralBitmap, JsonContentIndex>;

// ═══════════════════════════════════════════════════════════════════════════════════════
// JSON TYPE EVIDENCE
// ═══════════════════════════════════════════════════════════════════════════════════════

enum class JsonType : int32_t {
    Invalid = 0,
    Object = 1,
    Array = 2,
    String = 3,
    Number = 4,
    Boolean = 5,
    Null = 6,
    Colon = 7,
    Comma = 8
};

struct JsonTypeEvidence {
    JsonType type;
    int32_t depth;
    
    JsonTypeEvidence(JsonType t = JsonType::Invalid, int32_t d = 0) : type(t), depth(d) {}
};

struct JsonTypedValue {
    JsonTypeEvidence evidence;
    std::string value;
    
    JsonTypedValue() : evidence(JsonType::Invalid, 0) {}
    JsonTypedValue(JsonTypeEvidence ev, std::string val) : evidence(ev), value(std::move(val)) {}
};

// ═══════════════════════════════════════════════════════════════════════════════════════
// COMPACT JSON SCANNER (TrikeShed Elegance)
// ═══════════════════════════════════════════════════════════════════════════════════════

/**
 * Compact JSON scanner with JsonTypeEvidence
 * Ported from TrikeShed Kotlin implementation
 * Uses 64-bit words for bitmap (TrikeShed uses 64-bit chunks)
 */
class JsonScanner {
public:
    explicit JsonScanner(std::string_view input) : input_(input) {}
    
    /**
     * Single-pass scan with bitmap extraction
     */
    JsonDocument scan() {
        const size_t bitmap_size = (input_.length() + 63) >> 6; // 64-bit chunks, rounded up
        std::vector<uint64_t> bitmap(bitmap_size, 0);
        std::vector<JsonTypeEvidence> evidence_map;
        evidence_map.reserve(input_.length());
        
        // Initialize evidence map
        for (size_t i = 0; i < input_.length(); ++i) {
            evidence_map.emplace_back(JsonType::Invalid, 0);
        }
        
        std::vector<int32_t> index_series;
        size_t pos = 0;
        int32_t depth = 0;
        
        while (pos < input_.length()) {
            const char ch = input_[pos];
            const size_t word_index = pos >> 6;
            const size_t bit_index = pos & 63;
            
            switch (ch) {
                case '{':
                    bitmap[word_index] |= (1ULL << bit_index);
                    evidence_map[pos] = JsonTypeEvidence(JsonType::Object, depth++);
                    index_series.push_back(static_cast<int32_t>(pos));
                    break;
                    
                case '}':
                    bitmap[word_index] |= (2ULL << bit_index);
                    depth--;
                    break;
                    
                case '[':
                    bitmap[word_index] |= (4ULL << bit_index);
                    evidence_map[pos] = JsonTypeEvidence(JsonType::Array, depth++);
                    index_series.push_back(static_cast<int32_t>(pos));
                    break;
                    
                case ']':
                    bitmap[word_index] |= (8ULL << bit_index);
                    depth--;
                    break;
                    
                case '"':
                    bitmap[word_index] |= (16ULL << bit_index);
                    if (evidence_map[pos].type == JsonType::Invalid) {
                        evidence_map[pos] = JsonTypeEvidence(JsonType::String, depth);
                    }
                    break;
                    
                case ':':
                    bitmap[word_index] |= (32ULL << bit_index);
                    evidence_map[pos] = JsonTypeEvidence(JsonType::Colon, depth);
                    break;
                    
                case ',':
                    bitmap[word_index] |= (64ULL << bit_index);
                    evidence_map[pos] = JsonTypeEvidence(JsonType::Comma, depth);
                    break;
                    
                case 't': // true
                    if (pos + 4 <= input_.length() && 
                        input_.substr(pos, 4) == "true") {
                        if (evidence_map[pos].type == JsonType::Invalid) {
                            evidence_map[pos] = JsonTypeEvidence(JsonType::Boolean, depth);
                        }
                    }
                    break;
                    
                case 'f': // false
                    if (pos + 5 <= input_.length() && 
                        input_.substr(pos, 5) == "false") {
                        if (evidence_map[pos].type == JsonType::Invalid) {
                            evidence_map[pos] = JsonTypeEvidence(JsonType::Boolean, depth);
                        }
                    }
                    break;
                    
                case 'n': // null
                    if (pos + 4 <= input_.length() && 
                        input_.substr(pos, 4) == "null") {
                        if (evidence_map[pos].type == JsonType::Invalid) {
                            evidence_map[pos] = JsonTypeEvidence(JsonType::Null, depth);
                        }
                    }
                    break;
                    
                default:
                    if ((ch >= '0' && ch <= '9') || ch == '-') {
                        if (evidence_map[pos].type == JsonType::Invalid) {
                            evidence_map[pos] = JsonTypeEvidence(JsonType::Number, depth);
                        }
                    }
                    break;
            }
            pos++;
        }
        
        // Convert evidence map to int32_t vector for storage
        std::vector<int32_t> type_evidences;
        type_evidences.reserve(evidence_map.size());
        for (const auto& ev : evidence_map) {
            type_evidences.push_back(static_cast<int32_t>(ev.type) | (ev.depth << 16));
        }
        
        return JsonDocument{
            JsonStructuralBitmap{std::move(bitmap), std::move(type_evidences)},
            JsonContentIndex{std::move(index_series), input_}
        };
    }
    
    /**
     * Extract JSON value at position
     */
    std::string extract_value(size_t pos) {
        if (pos >= input_.length()) return "";
        
        // Skip whitespace
        while (pos < input_.length() && std::isspace(input_[pos])) {
            pos++;
        }
        
        if (pos >= input_.length()) return "";
        
        const char start_char = input_[pos];
        
        if (start_char == '"') {
            // String extraction
            size_t end = pos + 1;
            while (end < input_.length() && input_[end] != '"') {
                if (input_[end] == '\\' && end + 1 < input_.length()) {
                    end += 2; // Skip escape
                } else {
                    end++;
                }
            }
            if (end < input_.length()) {
                return std::string(input_.substr(pos, end - pos + 1));
            }
        } else if (start_char == 't' && pos + 4 <= input_.length() && input_.substr(pos, 4) == "true") {
            return "true";
        } else if (start_char == 'f' && pos + 5 <= input_.length() && input_.substr(pos, 5) == "false") {
            return "false";
        } else if (start_char == 'n' && pos + 4 <= input_.length() && input_.substr(pos, 4) == "null") {
            return "null";
        } else if (start_char == '{' || start_char == '[') {
            // Object or array - find matching closing brace/bracket
            char open = start_char;
            char close = (open == '{') ? '}' : ']';
            int depth = 0;
            size_t end = pos;
            
            while (end < input_.length()) {
                if (input_[end] == open) depth++;
                else if (input_[end] == close) {
                    depth--;
                    if (depth == 0) {
                        return std::string(input_.substr(pos, end - pos + 1));
                    }
                }
                end++;
            }
        }
        
        // Number or other
        size_t end = pos;
        while (end < input_.length() && 
               (input_[end] == '-' || input_[end] == '.' || 
                (input_[end] >= '0' && input_[end] <= '9'))) {
            end++;
        }
        return std::string(input_.substr(pos, end - pos));
    }
    
private:
    std::string_view input_;
};

// ═══════════════════════════════════════════════════════════════════════════════════════
// JSON DOCUMENT ACCESSOR (TrikeShed Pattern)
// ═══════════════════════════════════════════════════════════════════════════════════════

class JsonDocumentAccessor {
public:
    explicit JsonDocumentAccessor(const JsonDocument& doc) : doc_(doc) {}
    
    const std::vector<uint64_t>& bitmap() const {
        return doc_.first.first;
    }
    
    const std::vector<int32_t>& type_evidences() const {
        return doc_.first.second;
    }
    
    const std::vector<int32_t>& indices() const {
        return doc_.second.first;
    }
    
    std::string_view content() const {
        return doc_.second.second;
    }
    
    /**
     * Get evidence at position
     */
    JsonTypeEvidence get_evidence_at(size_t pos) const {
        const auto& types = type_evidences();
        if (pos < types.size()) {
            int32_t encoded = types[pos];
            JsonType type = static_cast<JsonType>(encoded & 0xFFFF);
            int32_t depth = encoded >> 16;
            return JsonTypeEvidence(type, depth);
        }
        return JsonTypeEvidence(JsonType::Invalid, 0);
    }
    
    /**
     * Check if position has any structural evidence
     */
    bool has_evidence_at(size_t pos) const {
        const auto& types = type_evidences();
        if (pos >= types.size()) return false;
        
        int32_t encoded = types[pos];
        JsonType type = static_cast<JsonType>(encoded & 0xFFFF);
        return type != JsonType::Invalid;
    }
    
private:
    const JsonDocument& doc_;
};

} // namespace cppfort::stage0

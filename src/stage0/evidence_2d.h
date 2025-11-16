#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <string_view>
#include <vector>
#include "type_evidence.h"

namespace cppfort::stage0 {

// 2D Evidence System: Confix Type × Position
// Each confix has exactly: 1 type, 1 begin position, 1 end position
// Enhanced with TEMPLATE CONFIX context awareness for >> disambiguation

// Use canonical ConfixType defined in `type_evidence.h`

// 2D evidence point: confix type + spatial position
struct ConfixEvidence {
    ConfixType type;
    size_t begin_pos;    // Single begin position
    size_t end_pos;      // Single end position
    
    ConfixEvidence(ConfixType t = ConfixType::INVALID, size_t begin = 0, size_t end = 0)
        : type(t), begin_pos(begin), end_pos(end) {}
    
    bool is_valid() const { return type != ConfixType::INVALID && begin_pos < end_pos; }
    size_t length() const { return end_pos - begin_pos; }
};

// TEMPLATE CONFIX context for >> disambiguation
// Tracks template nesting depth throughout analysis
struct TemplateConfixContext {
    int angle_depth = 0;
    bool in_template_confix = false;
    
    void update(char c) {
        if (c == '<') {
            angle_depth++;
            in_template_confix = true;
        } else if (c == '>') {
            angle_depth--;
            if (angle_depth <= 0) {
                angle_depth = 0;
                in_template_confix = false;
            }
        }
    }
    
    bool should_split_double_angle() const {
        return in_template_confix && angle_depth >= 2;
    }
    
    void reset() {
        angle_depth = 0;
        in_template_confix = false;
    }
};

// 2D evidence span: operates within locality bubble
struct EvidenceSpan2D {
    size_t start_pos = 0;
    size_t end_pos = 0;
    std::string content;
    double confidence = 0.0;
    
    // 2D confix evidence: type × spatial positions
    std::vector<ConfixEvidence> confixes;
    
    // Per-type evidence (2D: type × metrics)
    std::array<uint16_t, static_cast<uint8_t>(ConfixType::MAX_TYPE)> confix_type_counts = {0};
    std::array<uint16_t, static_cast<uint8_t>(ConfixType::MAX_TYPE)> confix_depth_at_start = {0};
    std::array<uint16_t, static_cast<uint8_t>(ConfixType::MAX_TYPE)> confix_depth_at_end = {0};
    
    EvidenceSpan2D() = default;
    EvidenceSpan2D(size_t start, size_t end, std::string text, double conf = 0.0)
        : start_pos(start), end_pos(end), content(std::move(text)), confidence(conf) {}
    
    // Add a confix evidence point to this span
    void add_confix(ConfixType type, size_t begin, size_t end) {
        if (type != ConfixType::INVALID && begin < end) {
            confixes.emplace_back(type, begin, end);
            confix_type_counts[static_cast<uint8_t>(type)]++;
        }
    }
    
    // Get all confixes of a specific type (2D slice)
    std::vector<ConfixEvidence> get_confixes_of_type(ConfixType type) const {
        std::vector<ConfixEvidence> result;
        for (const auto& confix : confixes) {
            if (confix.type == type) {
                result.push_back(confix);
            }
        }
        return result;
    }
    
    // Check if this span has balanced confixes (2D validation)
    bool has_balanced_confixes() const {
        for (uint8_t i = 1; i < static_cast<uint8_t>(ConfixType::MAX_TYPE); ++i) {
            auto type_confixes = get_confixes_of_type(static_cast<ConfixType>(i));
            if (confix_type_counts[i] % 2 != 0) {
                return false; // Odd number of confixes of this type
            }
        }
        return true;
    }
    
    // Get the dominant confix type in this span
    ConfixType get_dominant_confix_type() const {
        ConfixType dominant = ConfixType::INVALID;
        uint16_t max_count = 0;
        
        for (uint8_t i = 1; i < static_cast<uint8_t>(ConfixType::MAX_TYPE); ++i) {
            if (confix_type_counts[i] > max_count) {
                max_count = confix_type_counts[i];
                dominant = static_cast<ConfixType>(i);
            }
        }
        return dominant;
    }
    
    // Calculate 2D evidence confidence based on confix patterns
    double calculate_2d_confidence() const {
        if (confixes.empty()) return 0.0;
        
        double score = 0.0;
        
        // Bonus for having confixes
        score += 0.3;
        
        // Bonus for balanced confixes
        if (has_balanced_confixes()) {
            score += 0.4;
        }
        
        // Bonus for multiple confix types (diversity)
        uint8_t type_count = 0;
        for (uint8_t i = 1; i < static_cast<uint8_t>(ConfixType::MAX_TYPE); ++i) {
            if (confix_type_counts[i] > 0) type_count++;
        }
        score += (type_count * 0.1);
        
        return std::min(1.0, score);
    }
};

// 2D evidence analyzer: processes text into spatial confix evidence
class Evidence2DAnalyzer {
public:
    // Analyze text and build 2D evidence within locality bounds
    static EvidenceSpan2D analyze_span(std::string_view text, size_t start_offset = 0) {
        EvidenceSpan2D span(start_offset, start_offset + text.length(), std::string(text), 0.0);
        
        // Track confix depth for each type (2D: type × depth)
        std::array<int, static_cast<uint8_t>(ConfixType::MAX_TYPE)> depths = {0};
        TemplateConfixContext template_ctx;
        
        // Scan for structural characters and comments
        for (size_t i = 0; i < text.length(); ++i) {
            char c = text[i];
            
            // Handle comments first (they have special syntax)
            if (c == '/' && i + 1 < text.length()) {
                if (text[i + 1] == '*') { // C comment /* ... */
                    size_t end = find_c_comment_end(text, i);
                    if (end > i) {
                        span.add_confix(ConfixType::COMMENT_BLOCK, start_offset + i, start_offset + end);
                        i = end - 1;
                        continue;
                    }
                } else if (text[i + 1] == '/') {
                    // C++ comment // ... \n
                    size_t end = find_cpp_comment_end(text, i);
                    if (end > i) {
                        span.add_confix(ConfixType::COMMENT_LINE, start_offset + i, start_offset + end);
                        i = end - 1;
                        continue;
                    }
                }
            } else if (c == '`' && i + 2 < text.length() && 
                      text[i + 1] == '`' && text[i + 2] == '`') {
                // Cpp2 comment ``` ... ```
                size_t end = find_cpp2_comment_end(text, i);
                if (end > i) {
                    span.add_confix(ConfixType::COMMENT_DOC, start_offset + i, start_offset + end);
                    i = end - 1;
                    continue;
                }
            }
            
            // Handle structural confixes with TEMPLATE CONFIX awareness
            ConfixType type = get_confix_type(c);
            if (type != ConfixType::INVALID) {
                // Check for >> BEFORE updating context for the first >
                if (c == '>' && i + 1 < text.length() && text[i + 1] == '>') {
                    if (template_ctx.should_split_double_angle()) {
                        // Treat as two separate angle closes in template context
                        span.add_confix(ConfixType::ANGLE, start_offset + i, start_offset + i + 1);
                        span.add_confix(ConfixType::ANGLE, start_offset + i + 1, start_offset + i + 2);
                        depths[static_cast<uint8_t>(ConfixType::ANGLE)] -= 2;
                        template_ctx.angle_depth -= 2;
                        i++; // Skip the second >
                        continue;
                    }
                }
                
                // Normal single character confix
                size_t begin_pos = start_offset + i;
                size_t end_pos = begin_pos + 1;
                span.add_confix(type, begin_pos, end_pos);
                
                // Update TEMPLATE CONFIX context tracking
                uint8_t type_idx = static_cast<uint8_t>(type);
                if (is_confix_begin(c)) {
                    depths[type_idx]++;
                    span.confix_depth_at_start[type_idx] = depths[type_idx];
                    if (type == ConfixType::ANGLE) {
                        template_ctx.angle_depth++;
                    }
                } else if (is_confix_end(c)) {
                    depths[type_idx]--;
                    span.confix_depth_at_end[type_idx] = depths[type_idx];
                    if (type == ConfixType::ANGLE) {
                        template_ctx.angle_depth--;
                    }
                }
            }
        }
        
        // Calculate confidence based on 2D patterns
        span.confidence = span.calculate_2d_confidence();
        
        return span;
    }
    
    // Get confix type from character
    static ConfixType get_confix_type(char c) {
        switch (c) {
            case '(': case ')': return ConfixType::PAREN;
            case '{': case '}': return ConfixType::BRACE;
            case '[': case ']': return ConfixType::BRACKET;
            case '<': case '>': return ConfixType::ANGLE;
            default: return ConfixType::INVALID;
        }
    }

private:
    // Helper: determine if character is confix begin
    static bool is_confix_begin(char c) {
        return c == '(' || c == '{' || c == '[' || c == '<';
    }
    
    // Helper: determine if character is confix end
    static bool is_confix_end(char c) {
        return c == ')' || c == '}' || c == ']' || c == '>';
    }
    
    // Find end of C comment /* ... */
    static size_t find_c_comment_end(std::string_view text, size_t start) {
        if (start + 1 < text.length() && text[start] == '/' && text[start + 1] == '*') {
            for (size_t i = start + 2; i < text.length() - 1; ++i) {
                if (text[i] == '*' && text[i + 1] == '/') {
                    return i + 2;
                }
            }
        }
        return start + 1; // Fallback
    }
    
    // Find end of C++ comment // ... \n
    static size_t find_cpp_comment_end(std::string_view text, size_t start) {
        if (start + 1 < text.length() && text[start] == '/' && text[start + 1] == '/') {
            for (size_t i = start + 2; i < text.length(); ++i) {
                if (text[i] == '\n') {
                    return i + 1;
                }
            }
            return text.length(); // To end of text
        }
        return start + 1; // Fallback
    }
    
    // Find end of Cpp2 comment ``` ... ```
    static size_t find_cpp2_comment_end(std::string_view text, size_t start) {
        if (start + 2 < text.length() && text[start] == '`' && text[start + 1] == '`' && text[start + 2] == '`') {
            for (size_t i = start + 3; i < text.length() - 2; ++i) {
                if (text[i] == '`' && text[i + 1] == '`' && text[i + 2] == '`') {
                    return i + 3;
                }
            }
        }
        return start + 1; // Fallback
    }
};

// Helper functions for 2D evidence validation
namespace Evidence2D {
    // Check if a set of spans forms a valid 2D pattern
    inline bool validate_2d_pattern(const std::vector<EvidenceSpan2D>& spans) {
        for (const auto& span : spans) {
            if (!span.has_balanced_confixes()) {
                return false;
            }
        }
        return true;
    }
    
    // Find the span with the strongest 2D evidence
    inline const EvidenceSpan2D* find_strongest_evidence(const std::vector<EvidenceSpan2D>& spans) {
        const EvidenceSpan2D* strongest = nullptr;
        double max_confidence = 0.0;
        
        for (const auto& span : spans) {
            if (span.confidence > max_confidence) {
                max_confidence = span.confidence;
                strongest = &span;
            }
        }
        
        return strongest;
    }
    
    // Get total 2D evidence across all spans
    inline std::array<uint16_t, static_cast<uint8_t>(ConfixType::MAX_TYPE)> 
    get_total_2d_evidence(const std::vector<EvidenceSpan2D>& spans) {
        std::array<uint16_t, static_cast<uint8_t>(ConfixType::MAX_TYPE)> total = {0};
        
        for (const auto& span : spans) {
            for (uint8_t i = 0; i < static_cast<uint8_t>(ConfixType::MAX_TYPE); ++i) {
                total[i] += span.confix_type_counts[i];
            }
        }
        
        return total;
    }
}

} // namespace cppfort::stage0

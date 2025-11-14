#pragma once

#include "type_evidence.h"
#include "evidence.h"
#include <vector>
#include <unordered_map>
#include <functional>

namespace cppfort::stage0 {

/**
 * QuaternionOrbitEvidence: Integrates quaternion group Q₈ orbit principles
 * into TypeEvidence accumulation and semantic transformation.
 * 
 * Key insights from Q₈ structure:
 * - Center Z(Q₈) = {±1} ↔ Balance detection (Δ(type) = 0)
 * - Orbit sizes: 1, 2, 4, 8 ↔ Evidence span groupings
 * - Stabilizer subgroups ↔ What keeps evidence invariant
 * - Conjugacy classes ↔ Evidence transformation patterns
 */
class QuaternionOrbitEvidence {
public:
    // Orbit types based on Q₈ conjugacy classes
    enum class OrbitType : uint8_t {
        CENTER_FIXED = 0,      // {1} - Fixed point, always balanced
        CENTER_NEG = 1,        // {-1} - Negation, flips balance
        ORBIT_I = 2,           // {±i} - Template/nested type context
        ORBIT_J = 3,           // {±j} - Function/scope context  
        ORBIT_K = 4,           // {±k} - Expression/operator context
        INVALID = 5
    };

    // Evidence span with orbit classification
    struct OrbitEvidenceSpan {
        EvidenceSpan span;
        OrbitType orbit_type = OrbitType::INVALID;
        uint16_t stabilizer_order = 0;  // Size of stabilizer subgroup
        uint16_t orbit_size = 0;        // Size of orbit
        
        // Quaternion group action state
        bool is_balanced = false;       // In center Z(Q₈)
        bool is_conjugated = false;     // Under conjugation action
        
        OrbitEvidenceSpan() = default;
        OrbitEvidenceSpan(const EvidenceSpan& s) : span(s) {}
    };

private:
    // Current evidence being accumulated
    TypeEvidence current_evidence;
    
    // Orbit classification for current position
    OrbitType current_orbit = OrbitType::INVALID;
    
    // Evidence spans grouped by orbit type (like conjugacy classes)
    std::vector<OrbitEvidenceSpan> orbit_spans[5];  // One per OrbitType
    
    // Stabilizer tracking: what keeps evidence invariant
    std::unordered_map<uint32_t, uint16_t> stabilizer_map;  // hash(evidence) -> stabilizer order
    
    // Center detection: evidence that commutes with everything
    std::vector<EvidenceSpan> center_spans;  // Z(Q₈) equivalent

public:
    QuaternionOrbitEvidence() = default;

    /**
     * Observe character with quaternion orbit awareness
     * Like group action: g · x = updated evidence
     */
    void observe_char_with_orbit(char ch, size_t position) {
        // Update base evidence
        current_evidence.observe_char(ch);
        
        // Determine orbit type based on character context
        // This is like determining which conjugacy class we're in
        OrbitType new_orbit = classify_orbit_by_char(ch);
        
        if (new_orbit != current_orbit && current_orbit != OrbitType::INVALID) {
            // Orbit changed - finalize previous span
            finalize_current_span(position);
        }
        
        current_orbit = new_orbit;
    }

    /**
     * Observe confix delimiter with orbit tracking
     * Confix balance ↔ center membership in Q₈
     */
    void observe_confix_with_orbit(ConfixType type, bool is_open, size_t position) {
        if (is_open) {
            current_evidence.observe_confix_open(type);
        } else {
            current_evidence.observe_confix_close(type);
        }
        
        // Check if we're in center (balanced)
        if (current_evidence.all_confix_balanced()) {
            // In Z(Q₈) - center of group
            current_orbit = OrbitType::CENTER_FIXED;
        } else {
            // Not in center - classify by confix type
            current_orbit = classify_orbit_by_confix(type);
        }
    }

    /**
     * Accumulate evidence span with orbit calculation
     * Applies orbit-stabilizer theorem: |Orbit| = |G|/|Stabilizer|
     */
    void accumulate_span(const EvidenceSpan& span) {
        OrbitEvidenceSpan orbit_span(span);
        
        // Calculate stabilizer (what keeps this evidence invariant)
        uint32_t evidence_hash = hash_evidence(span.evidence);
        uint16_t stabilizer = calculate_stabilizer_order(span.evidence);
        
        orbit_span.stabilizer_order = stabilizer;
        orbit_span.orbit_size = (stabilizer == 0) ? 0 : 8 / stabilizer;  // |G|=8 for Q₈
        orbit_span.is_balanced = span.evidence.all_confix_balanced();
        orbit_span.orbit_type = current_orbit;
        
        // Classify like Q₈ conjugacy classes
        if (orbit_span.is_balanced && orbit_span.orbit_size == 1) {
            // Fixed point - in center
            orbit_span.orbit_type = OrbitType::CENTER_FIXED;
            center_spans.push_back(span);
        } else if (orbit_span.orbit_size == 2) {
            // Size 2 orbit - like {±i}, {±j}, {±k}
            // Already classified by current_orbit
        } else if (orbit_span.orbit_size == 4) {
            // Size 4 orbit - stabilizer of order 2
            orbit_span.is_conjugated = true;
        }
        
        if (current_orbit != OrbitType::INVALID) {
            orbit_spans[static_cast<uint8_t>(current_orbit)].push_back(orbit_span);
        }
    }

    /**
     * Get evidence spans for a specific orbit type
     * Like getting all elements of a conjugacy class
     */
    const std::vector<OrbitEvidenceSpan>& get_orbit_spans(OrbitType type) const {
        static std::vector<OrbitEvidenceSpan> empty;
        uint8_t idx = static_cast<uint8_t>(type);
        if (idx >= 5) return empty;
        return orbit_spans[idx];
    }

    /**
     * Get center spans (Z(Q₈) equivalent)
     * These are spans where Δ(type) = 0 (perfectly balanced)
     */
    const std::vector<EvidenceSpan>& get_center_spans() const {
        return center_spans;
    }

    /**
     * Check if evidence is in center (commutes with everything)
     * ↔ all confix types are balanced
     */
    bool is_in_center(const TypeEvidence& evidence) const {
        return evidence.all_confix_balanced();
    }

    /**
     * Apply conjugation action: g · evidence → new_evidence
     * Simulates how group elements act on evidence by conjugation
     */
    TypeEvidence conjugate_evidence(const TypeEvidence& evidence, uint8_t action_type) const {
        TypeEvidence result = evidence;
        
        // Different actions (like i, j, k) transform evidence differently
        switch (action_type) {
            case 1:  // Action 'i' - template context transformation
                result.template_ids = result.template_ids * 2;
                result.max_confix_depth[static_cast<uint8_t>(ConfixType::ANGLE)] += 1;
                break;
                
            case 2:  // Action 'j' - function context transformation  
                result.flow_keywords = result.flow_keywords * 2;
                result.max_confix_depth[static_cast<uint8_t>(ConfixType::BRACE)] += 1;
                break;
                
            case 3:  // Action 'k' - expression context transformation
                result.comma = result.comma * 2;
                result.max_confix_depth[static_cast<uint8_t>(ConfixType::PAREN)] += 1;
                break;
                
            case 0:  // Action '-1' - negation
                // Flip signs, invert balances
                for (size_t i = 0; i < 12; ++i) {
                    std::swap(result.confix_open[i], result.confix_close[i]);
                }
                break;
        }
        
        return result;
    }

    /**
     * Get orbit structure summary
     * Returns counts like Q₈ conjugacy class sizes
     */
    struct OrbitStructure {
        size_t center_fixed = 0;    // Size 1 orbits (fixed points)
        size_t center_neg = 0;      // Size 1 orbits (negation)
        size_t orbit_i = 0;         // Size 2 orbits (template context)
        size_t orbit_j = 0;         // Size 2 orbits (function context)
        size_t orbit_k = 0;         // Size 2 orbits (expression context)
    };

    OrbitStructure get_orbit_structure() const {
        OrbitStructure structure;
        structure.center_fixed = center_spans.size();
        structure.orbit_i = orbit_spans[static_cast<uint8_t>(OrbitType::ORBIT_I)].size();
        structure.orbit_j = orbit_spans[static_cast<uint8_t>(OrbitType::ORBIT_J)].size();
        structure.orbit_k = orbit_spans[static_cast<uint8_t>(OrbitType::ORBIT_K)].size();
        return structure;
    }

    /**
     * Reset accumulation state
     */
    void reset() {
        current_evidence.reset();
        current_orbit = OrbitType::INVALID;
        for (auto& spans : orbit_spans) {
            spans.clear();
        }
        stabilizer_map.clear();
        center_spans.clear();
    }
    
    /**
     * Get transformation path (simplified for integration)
     * Returns vector of positions representing path through evidence
     */
    std::vector<size_t> get_transformation_path(size_t start_pos, size_t end_pos) const {
        std::vector<size_t> path;
        // Simple linear path for now
        for (size_t pos = start_pos; pos <= end_pos && pos < 10000; ++pos) {
            path.push_back(pos);
        }
        return path;
    }
    
    // Public helper methods for testing and integration
    QuaternionOrbitEvidence::OrbitType classify_orbit_public(char ch) const {
        return this->classify_orbit_by_char(ch);
    }
    
    bool is_in_center_public(const TypeEvidence& evidence) const {
        return this->is_in_center(evidence);
    }
    
    const TypeEvidence& get_current_evidence() const {
        return current_evidence;
    }
    
private:
    // Classify orbit based on character context
    OrbitType classify_orbit_by_char(char ch) const {
        if (ch == '<' || ch == '>' || ch == ':' || ch == ':') {
            return QuaternionOrbitEvidence::OrbitType::ORBIT_I;  // Template/type context
        } else if (ch == '{' || ch == '}' || ch == ';') {
            return QuaternionOrbitEvidence::OrbitType::ORBIT_J;  // Function/scope context
        } else if (ch == '(' || ch == ')' || ch == ',' || ch == '+' || ch == '-' || 
                   ch == '*' || ch == '/') {
            return QuaternionOrbitEvidence::OrbitType::ORBIT_K;  // Expression/operator context
        } else if (ch == '1' || ch == 't' || ch == 'T') {
            return QuaternionOrbitEvidence::OrbitType::CENTER_FIXED;  // Fixed point
        } else if (ch == '-' || ch == '!') {
            return OrbitType::CENTER_NEG;  // Negation
        }
        return current_orbit;  // Keep current orbit
    }

    // Classify orbit based on confix type
    OrbitType classify_orbit_by_confix(ConfixType type) {
        switch (type) {
            case ConfixType::ANGLE:
                return OrbitType::ORBIT_I;  // Template parameters
            case ConfixType::BRACE:
                return OrbitType::ORBIT_J;  // Code blocks
            case ConfixType::BRACKET:
            case ConfixType::PAREN:
                return OrbitType::ORBIT_K;  // Expressions
            default:
                return OrbitType::INVALID;
        }
    }

    // Hash evidence for stabilizer calculation
    uint32_t hash_evidence(const TypeEvidence& evidence) const {
        // Simple hash combining key fields
        uint32_t hash = 0;
        hash ^= evidence.digits * 0x9e3779b9;
        hash ^= evidence.alpha * 0x9e3779b9;
        hash ^= evidence.special * 0x9e3779b9;
        for (size_t i = 0; i < 12; ++i) {
            hash ^= (evidence.confix_open[i] + evidence.confix_close[i]) * (0x9e3779b9 + i);
        }
        return hash;
    }

    // Calculate stabilizer order (size of subgroup that keeps evidence invariant)
    uint16_t calculate_stabilizer_order(const TypeEvidence& evidence) const {
        // Evidence with more structure has smaller stabilizer (larger orbit)
        uint32_t complexity = 0;
        
        // Count active features
        if (evidence.digits > 0) complexity += 1;
        if (evidence.alpha > 0) complexity += 1;
        if (evidence.template_ids > 0) complexity += 2;
        if (evidence.cpp_identifiers > 0) complexity += 1;
        if (evidence.flow_keywords > 0) complexity += 1;
        
        // Count confix types in use
        uint8_t confix_types = 0;
        for (size_t i = 0; i < 12; ++i) {
            if (evidence.confix_open[i] > 0 || evidence.confix_close[i] > 0) {
                confix_types++;
            }
        }
        complexity += confix_types;
        
        // Stabilizer order: higher complexity → smaller stabilizer → larger orbit
        // Q₈ has orders: 8 (center), 4, 2, 1 (free action)
        if (complexity == 0) return 8;  // Trivial case - full stabilizer
        if (complexity == 1) return 4;  // Simple - stabilizer of order 4
        if (complexity <= 3) return 2;  // Moderate - stabilizer of order 2
        return 1;  // Complex - trivial stabilizer (free action)
    }

    // Finalize current span accumulation
    void finalize_current_span(size_t position) {
        // Implementation would create EvidenceSpan from current state
        // and call accumulate_span()
    }
};

/**
 * QuaternionEvidenceTransformer: Applies quaternion group actions to semantic transformations
 * Connects graph representations to the semantic transformation pipeline
 */
class QuaternionEvidenceTransformer {
private:
    QuaternionOrbitEvidence orbit_evidence;
    
    // Graph representation of evidence transformations
    struct EvidenceNode {
        TypeEvidence evidence;
        size_t position;
        std::vector<size_t> transitions;  // Like Cayley graph edges
    };
    
    std::vector<EvidenceNode> evidence_graph;

public:
    QuaternionEvidenceTransformer() = default;

    /**
     * Transform evidence using quaternion group actions
     * Like applying generators from Cayley graph
     */
    TypeEvidence transform_evidence(const TypeEvidence& input,
                                   const std::string& action) {
        // Map actions to quaternion generators
        if (action == "template_open") {
            // Like multiplying by 'i' - enters template orbit
            return orbit_evidence.conjugate_evidence(input, 1);
        } else if (action == "function_open") {
            // Like multiplying by 'j' - enters function orbit  
            return orbit_evidence.conjugate_evidence(input, 2);
        } else if (action == "expression_open") {
            // Like multiplying by 'k' - enters expression orbit
            return orbit_evidence.conjugate_evidence(input, 3);
        } else if (action == "negate") {
            // Like multiplying by '-1' - flips evidence
            return orbit_evidence.conjugate_evidence(input, 0);
        }
        
        return input;  // Identity action
    }

    /**
     * Build evidence transformation graph
     * Similar to Cayley graph construction for Q₈
     */
    void build_transformation_graph(const std::string& source) {
        evidence_graph.clear();
        orbit_evidence.reset();
        
        // Scan source and build graph nodes
        for (size_t pos = 0; pos < source.length(); ++pos) {
            char ch = source[pos];
            
            // Observe with orbit awareness
            orbit_evidence.observe_char_with_orbit(ch, pos);
            
            // Create node
            EvidenceNode node;
            node.evidence = orbit_evidence.get_current_evidence();
            node.position = pos;
            
            // Add transitions based on possible actions
            // Like adding edges in Cayley graph
            if (ch == '<') {
                node.transitions.push_back(pos + 1);  // Template action
            } else if (ch == '{') {
                node.transitions.push_back(pos + 1);  // Function action
            } else if (ch == '(') {
                node.transitions.push_back(pos + 1);  // Expression action
            }
            
            evidence_graph.push_back(node);
        }
    }

    /**
     * Get transformation path through evidence graph
     * Like finding path in Cayley graph from generators
     */
    std::vector<size_t> get_transformation_path(size_t start_pos, size_t end_pos) const {
        std::vector<size_t> path;
        
        // Simple path finding through graph
        // In practice, would use BFS/DFS with orbit constraints
        size_t current = start_pos;
        path.push_back(current);
        
        while (current < end_pos && current < evidence_graph.size()) {
            const auto& node = evidence_graph[current];
            if (!node.transitions.empty()) {
                current = node.transitions[0];  // Follow first transition
                path.push_back(current);
            } else {
                current++;  // Linear progression
                path.push_back(current);
            }
        }
        
        return path;
    }

};

} // namespace cppfort::stage0
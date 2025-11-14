#pragma once

#include "quaternion_evidence.h"
#include "graph_semantic_transformer.h"
#include "orbit_pipeline.h"
#include <memory>

namespace cppfort::stage0 {

/**
 * QuaternionSemanticIntegration: Bridges quaternion orbit principles
 * with the semantic transformation pipeline
 * 
 * This integrates:
 * 1. QuaternionOrbitEvidence - mathematical structure from Q₈
 * 2. GraphSemanticTransformer - semantic transformation graph
 * 3. OrbitPipeline - evidence accumulation and pattern matching
 * 
 * The integration applies orbit-stabilizer principles to guide
 * semantic transformations based on evidence structure.
 */
class QuaternionSemanticIntegration {
private:
    // Core components
    std::unique_ptr<QuaternionOrbitEvidence> orbit_evidence;
    std::unique_ptr<cppfort::ir::GraphSemanticTransformer> graph_transformer;
    
    // Pipeline integration
    OrbitPipeline* pipeline = nullptr;
    
    // Current transformation state
    struct TransformState {
        size_t position = 0;
        QuaternionOrbitEvidence::OrbitType current_orbit;
        TypeEvidence accumulated_evidence;
        std::vector<size_t> transformation_path;
        bool in_center = false;  // In Z(Q₈) - balanced state
    } current_state;

public:
    QuaternionSemanticIntegration() 
        : orbit_evidence(std::make_unique<QuaternionOrbitEvidence>()),
          graph_transformer(std::make_unique<cppfort::ir::GraphSemanticTransformer>()) {
    }

    /**
     * Initialize integration with pipeline
     */
    void initialize(OrbitPipeline* pipe) {
        pipeline = pipe;
        reset();
    }

    /**
     * Process source code with quaternion orbit awareness
     * Integrates evidence accumulation with semantic transformation
     */
    std::string process_with_quaternion_semantics(const std::string& source) {
        reset();
        
        // Accumulate quaternion evidence by scanning source
        for (size_t i = 0; i < source.length(); ++i) {
            accumulate_quaternion_evidence(source[i], i);
        }
        
        // Apply quaternion-guided semantic transformations
        return apply_quaternion_transformations(source);
    }

    /**
     * Apply semantic transformations guided by quaternion orbit structure
     */
    std::string apply_quaternion_transformations(const std::string& input) {
        std::string result = input;
        
        // Get topological order from graph transformer
        auto transformation_order = graph_transformer->topologicalSort();
        
        // Apply transformations with orbit awareness
        for (const auto& node_id : transformation_order) {
            // Get transformation node
            auto path = graph_transformer->getTransformationPath(node_id);
            if (path.empty()) continue;
            
            // Get evidence for this transformation region
            auto evidence_path = orbit_evidence->get_transformation_path(
                current_state.position, 
                current_state.position + 100  // Region size
            );
            
            // Determine orbit type for this transformation
            auto orbit_type = classify_transformation_orbit(node_id);
            
            // Apply orbit-appropriate transformation
            if (apply_orbit_aware_transformation(result, orbit_type, evidence_path)) {
                update_state(evidence_path);
            }
        }
        
        return result;
    }

    /**
     * Accumulate evidence with quaternion orbit tracking
     * Called during pipeline processing
     */
    void accumulate_quaternion_evidence(char ch, size_t position) {
        // Track orbit state
        orbit_evidence->observe_char_with_orbit(ch, position);
        
        // Update current state
        current_state.position = position;
        current_state.accumulated_evidence.observe_char(ch);
        
        // Check if we're in center (balanced)
        if (orbit_evidence->is_in_center(current_state.accumulated_evidence)) {
            current_state.in_center = true;
            current_state.current_orbit = QuaternionOrbitEvidence::OrbitType::CENTER_FIXED;
        } else {
            current_state.in_center = false;
            current_state.current_orbit = classify_position_orbit(ch);
        }
    }

    /**
     * Finalize evidence span with orbit calculation
     */
    void finalize_evidence_span(const EvidenceSpan& span) {
        orbit_evidence->accumulate_span(span);
        
        // Update transformation path
        current_state.transformation_path.push_back(span.start_pos);
    }

    /**
     * Get current orbit structure (like Q₈ conjugacy class summary)
     */
    QuaternionOrbitEvidence::OrbitStructure get_current_orbit_structure() const {
        return orbit_evidence->get_orbit_structure();
    }

    /**
     * Get orbit statistics for debugging/analysis
     */
    struct OrbitStatistics {
        size_t total_spans = 0;
        size_t center_spans = 0;
        size_t template_orbits = 0;  // ORBIT_I
        size_t function_orbits = 0;  // ORBIT_J
        size_t expression_orbits = 0; // ORBIT_K
        double center_ratio = 0.0;
        std::string dominant_orbit;
    };

    OrbitStatistics get_orbit_statistics() const {
        OrbitStatistics stats;
        auto structure = orbit_evidence->get_orbit_structure();
        
        stats.center_spans = structure.center_fixed;
        stats.template_orbits = structure.orbit_i;
        stats.function_orbits = structure.orbit_j;
        stats.expression_orbits = structure.orbit_k;
        
        stats.total_spans = stats.center_spans + stats.template_orbits + 
                           stats.function_orbits + stats.expression_orbits;
        
        if (stats.total_spans > 0) {
            stats.center_ratio = static_cast<double>(stats.center_spans) / stats.total_spans;
        }
        
        // Find dominant orbit
        size_t max_orbit = std::max({stats.template_orbits, stats.function_orbits, 
                                    stats.expression_orbits});
        if (max_orbit == stats.template_orbits) {
            stats.dominant_orbit = "template";
        } else if (max_orbit == stats.function_orbits) {
            stats.dominant_orbit = "function";
        } else if (max_orbit == stats.expression_orbits) {
            stats.dominant_orbit = "expression";
        } else {
            stats.dominant_orbit = "center";
        }
        
        return stats;
    }

private:
    /**
     * Reset integration state
     */
    void reset() {
        orbit_evidence->reset();
        current_state = TransformState{};
    }

    /**
     * Classify transformation node to orbit type
     */
    QuaternionOrbitEvidence::OrbitType classify_transformation_orbit(
        const std::string& node_id) {
        
        if (node_id.find("template") != std::string::npos ||
            node_id.find("type") != std::string::npos ||
            node_id.find("param") != std::string::npos) {
            return QuaternionOrbitEvidence::OrbitType::ORBIT_I;
        } else if (node_id.find("function") != std::string::npos ||
                   node_id.find("scope") != std::string::npos ||
                   node_id.find("block") != std::string::npos) {
            return QuaternionOrbitEvidence::OrbitType::ORBIT_J;
        } else if (node_id.find("expression") != std::string::npos ||
                   node_id.find("operator") != std::string::npos ||
                   node_id.find("call") != std::string::npos) {
            return QuaternionOrbitEvidence::OrbitType::ORBIT_K;
        }
        
        return QuaternionOrbitEvidence::OrbitType::CENTER_FIXED;
    }

    /**
     * Apply orbit-aware transformation based on Q₈ structure
     */
    bool apply_orbit_aware_transformation(
        std::string& code,
        QuaternionOrbitEvidence::OrbitType orbit_type,
        const std::vector<size_t>& evidence_path) {
        
        if (evidence_path.size() < 2) return false;
        
        size_t start = evidence_path[0];
        size_t end = evidence_path[evidence_path.size() - 1];
        
        if (end <= start || end > code.length()) return false;
        
        std::string region = code.substr(start, end - start);
        std::string transformed;
        
        // Apply transformation based on orbit type
        switch (orbit_type) {
            case QuaternionOrbitEvidence::OrbitType::ORBIT_I:
                // Template context transformation
                transformed = transform_template_region(region);
                break;
                
            case QuaternionOrbitEvidence::OrbitType::ORBIT_J:
                // Function context transformation
                transformed = transform_function_region(region);
                break;
                
            case QuaternionOrbitEvidence::OrbitType::ORBIT_K:
                // Expression context transformation
                transformed = transform_expression_region(region);
                break;
                
            case QuaternionOrbitEvidence::OrbitType::CENTER_FIXED:
                // Balanced region - minimal transformation
                transformed = transform_balanced_region(region);
                break;
                
            default:
                return false;
        }
        
        // Apply transformation
        code.replace(start, end - start, transformed);
        return true;
    }

    /**
     * Transform template region (ORBIT_I)
     */
    std::string transform_template_region(const std::string& region) {
        // Apply template-specific transformations
        std::string result = region;
        
        // Example: Convert Cpp2 template syntax to C++
        // This would use the graph transformer for actual mappings
        if (region.find("template") != std::string::npos) {
            // Apply template transformations from graph
            result = graph_transformer->transformCpp2ToCpp(region);
        }
        
        return result;
    }

    /**
     * Transform function region (ORBIT_J)
     */
    std::string transform_function_region(const std::string& region) {
        std::string result = region;
        
        // Apply function-specific transformations
        if (region.find("main:") != std::string::npos ||
            region.find(": ()") != std::string::npos) {
            result = graph_transformer->transformCpp2ToCpp(region);
        }
        
        return result;
    }

    /**
     * Transform expression region (ORBIT_K)
     */
    std::string transform_expression_region(const std::string& region) {
        std::string result = region;
        
        // Apply expression-specific transformations
        // Parameter modes, operators, etc.
        result = graph_transformer->transformCpp2ToCpp(region);
        
        return result;
    }

    /**
     * Transform balanced region (CENTER)
     */
    std::string transform_balanced_region(const std::string& region) {
        // Balanced regions need minimal transformation
        // They're already in center Z(Q₈)
        return region;
    }

    /**
     * Update state after transformation
     */
    void update_state(const std::vector<size_t>& evidence_path) {
        if (!evidence_path.empty()) {
            current_state.position = evidence_path.back();
        }
    }

    /**
     * Classify orbit by character position
     */
    QuaternionOrbitEvidence::OrbitType classify_position_orbit(char ch) {
        return orbit_evidence->classify_orbit_public(ch);
    }
};

} // namespace cppfort::stage0
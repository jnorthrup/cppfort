#pragma once

#include "node.h"
#include "orbit_mask.h"
#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace cppfort {
namespace ir {

// Forward declaration
class PatternMatcher;

/**
 * Extended target language enum that includes C/C++ alongside MLIR dialects
 */
enum class ProjectionTarget {
    C,              // C language
    CPP,            // C++ language
    CPP2,           // CPP2 language
    MLIR_ARITH,     // MLIR Arithmetic dialect
    MLIR_CF,        // MLIR Control Flow dialect
    MLIR_SCF,       // MLIR Structured Control Flow
    MLIR_MEMREF,    // MLIR Memory Reference dialect
    MLIR_FUNC,      // MLIR Function dialect
    COUNT           // Total number of targets
};

/**
 * Result of attempting to project a pattern to a target language
 */
struct ProjectionResult {
    ProjectionTarget target;
    bool feasible;              // Can this pattern be lowered to this target?
    double confidence;          // Roundtrip confidence (0.0-1.0)
    std::string reason;         // Why feasible/infeasible or confidence boost explanation

    ProjectionResult()
        : target(ProjectionTarget::COUNT), feasible(false), confidence(0.0) {}

    ProjectionResult(ProjectionTarget t, bool f, double c, const std::string& r)
        : target(t), feasible(f), confidence(c), reason(r) {}
};

/**
 * Speculative match with n-way projection information
 */
struct SpeculativeMatch {
    OrbitMatch orbit;                                           // Original orbit match
    double base_confidence;                                     // Base confidence from orbit scanner
    std::array<ProjectionResult, static_cast<size_t>(ProjectionTarget::COUNT)> projections;  // N-way projections
    
    SpeculativeMatch() : base_confidence(0.0) {}
    
    explicit SpeculativeMatch(const OrbitMatch& match) 
        : orbit(match), base_confidence(match.confidence) {}
    
    // Get the best projection (highest confidence)
    ProjectionResult getBestProjection() const;
    
    // Get all feasible projections
    std::vector<ProjectionResult> getFeasibleProjections() const;
    
    // Check if any projection is feasible
    bool hasAnyFeasibleProjection() const;
};

/**
 * Projection Oracle: O(1) feasibility checking for n-way target projection
 * 
 * This oracle sits between the orbit scanner and pattern matcher, providing
 * fast feasibility checks without full codegen. It uses:
 * - Hash-based pattern registry lookups (O(1) per target)
 * - Roundtrip capability matrix (compile-time constant)
 * - Confidence boosting for patterns with roundtrip support
 * 
 * Complexity: O(T) where T = number of targets (constant = 8)
 * NOT O(n²) explosion because T is fixed and lookups are O(1)
 */
class ProjectionOracle {
private:
    // Roundtrip support matrix: [source][target] -> supports roundtrip
    // CPP2 ↔ C++, CPP2 ↔ C have roundtrip capability
    std::array<std::array<bool, static_cast<size_t>(ProjectionTarget::COUNT)>,
               static_cast<size_t>(GrammarType::UNKNOWN)> m_roundtripMatrix;
    
    // Pattern feasibility cache: (pattern_name, target) -> feasible
    // Populated lazily as patterns are checked
    using FeasibilityKey = std::pair<std::string, ProjectionTarget>;
    struct FeasibilityKeyHash {
        std::size_t operator()(const FeasibilityKey& k) const {
            return std::hash<std::string>()(k.first) ^
                   (std::hash<int>()(static_cast<int>(k.second)) << 1);
        }
    };
    mutable std::unordered_map<FeasibilityKey, bool, FeasibilityKeyHash> m_feasibilityCache;
    
    // Reference to pattern matcher for feasibility checks
    const PatternMatcher* m_patternMatcher;
    
    // Confidence boost factors
    static constexpr double ROUNDTRIP_BOOST = 1.15;      // 15% boost for roundtrip capability
    static constexpr double NO_PATTERN_PENALTY = 0.85;   // 15% penalty if target missing
    static constexpr double MULTI_TARGET_BOOST = 1.05;   // 5% boost per additional feasible target
    
    // Initialize roundtrip support matrix
    void initializeRoundtripMatrix();
    
    // Convert GrammarType to ProjectionTarget
    ProjectionTarget grammarToTarget(GrammarType grammar) const;
    
    // Check if a pattern can be lowered to a target (with caching)
    bool checkFeasibility(const std::string& patternName, 
                         GrammarType sourceGrammar,
                         ProjectionTarget target) const;

public:
    /**
     * Constructor
     * @param patternMatcher Optional pattern matcher for feasibility checks
     */
    explicit ProjectionOracle(const PatternMatcher* patternMatcher = nullptr);
    
    /**
     * Set the pattern matcher for feasibility checks
     */
    void setPatternMatcher(const PatternMatcher* patternMatcher);
    
    /**
     * Project an orbit match to all n-way targets
     * Returns a SpeculativeMatch with projection results and adjusted confidence
     * 
     * Complexity: O(T) where T = 8 targets (constant time)
     */
    SpeculativeMatch projectToAllTargets(const OrbitMatch& match) const;
    
    /**
     * Check if a specific projection is feasible
     * O(1) hash lookup with caching
     */
    bool isProjectionFeasible(const std::string& patternName,
                             GrammarType sourceGrammar,
                             ProjectionTarget target) const;
    
    /**
     * Check if roundtrip is supported between source and target
     * O(1) array lookup
     */
    bool supportsRoundtrip(GrammarType source, ProjectionTarget target) const;
    
    /**
     * Get roundtrip confidence boost for a grammar-target pair
     * Returns multiplier (e.g., 1.15 for 15% boost)
     */
    double getRoundtripBoost(GrammarType source, ProjectionTarget target) const;
    
    /**
     * Clear the feasibility cache (useful for testing or pattern updates)
     */
    void clearCache();
    
    /**
     * Get cache statistics
     */
    struct CacheStats {
        size_t hits = 0;
        size_t misses = 0;
        size_t entries = 0;
    };
    CacheStats getCacheStats() const;
};

} // namespace ir
} // namespace cppfort
#include "projection_oracle.h"
#include "pattern_matcher.h"
#include <algorithm>
#include <numeric>

namespace cppfort::ir {

// SpeculativeMatch implementation

ProjectionResult SpeculativeMatch::getBestProjection() const {
    auto feasible = getFeasibleProjections();
    if (feasible.empty()) {
        return ProjectionResult();
    }
    
    return *std::max_element(feasible.begin(), feasible.end(),
        [](const ProjectionResult& a, const ProjectionResult& b) {
            return a.confidence < b.confidence;
        });
}

std::vector<ProjectionResult> SpeculativeMatch::getFeasibleProjections() const {
    std::vector<ProjectionResult> result;
    for (const auto& proj : projections) {
        if (proj.feasible) {
            result.push_back(proj);
        }
    }
    return result;
}

bool SpeculativeMatch::hasAnyFeasibleProjection() const {
    return std::any_of(projections.begin(), projections.end(),
        [](const ProjectionResult& p) { return p.feasible; });
}

// ProjectionOracle implementation

ProjectionOracle::ProjectionOracle(const PatternMatcher* patternMatcher)
    : m_patternMatcher(patternMatcher) {
    initializeRoundtripMatrix();
}

void ProjectionOracle::setPatternMatcher(const PatternMatcher* patternMatcher) {
    m_patternMatcher = patternMatcher;
    // Clear cache when pattern matcher changes
    clearCache();
}

void ProjectionOracle::initializeRoundtripMatrix() {
    // Initialize all to false
    for (auto& row : m_roundtripMatrix) {
        row.fill(false);
    }
    
    // Define roundtrip capabilities
    // CPP2 ↔ C++ (bidirectional transpilation)
    m_roundtripMatrix[static_cast<size_t>(GrammarType::CPP2)]
                     [static_cast<size_t>(ProjectionTarget::CPP)] = true;
    m_roundtripMatrix[static_cast<size_t>(GrammarType::CPP)]
                     [static_cast<size_t>(ProjectionTarget::CPP2)] = true;
    
    // CPP2 ↔ C (with some limitations)
    m_roundtripMatrix[static_cast<size_t>(GrammarType::CPP2)]
                     [static_cast<size_t>(ProjectionTarget::C)] = true;
    m_roundtripMatrix[static_cast<size_t>(GrammarType::C)]
                     [static_cast<size_t>(ProjectionTarget::CPP2)] = true;
    
    // C ↔ C++ (mostly compatible)
    m_roundtripMatrix[static_cast<size_t>(GrammarType::C)]
                     [static_cast<size_t>(ProjectionTarget::CPP)] = true;
    m_roundtripMatrix[static_cast<size_t>(GrammarType::CPP)]
                     [static_cast<size_t>(ProjectionTarget::C)] = true;
    
    // Self-roundtrips (identity)
    m_roundtripMatrix[static_cast<size_t>(GrammarType::C)]
                     [static_cast<size_t>(ProjectionTarget::C)] = true;
    m_roundtripMatrix[static_cast<size_t>(GrammarType::CPP)]
                     [static_cast<size_t>(ProjectionTarget::CPP)] = true;
    m_roundtripMatrix[static_cast<size_t>(GrammarType::CPP2)]
                     [static_cast<size_t>(ProjectionTarget::CPP2)] = true;
}

ProjectionTarget ProjectionOracle::grammarToTarget(GrammarType grammar) const {
    switch (grammar) {
        case GrammarType::C:    return ProjectionTarget::C;
        case GrammarType::CPP:  return ProjectionTarget::CPP;
        case GrammarType::CPP2: return ProjectionTarget::CPP2;
        default:                return ProjectionTarget::COUNT;
    }
}

bool ProjectionOracle::checkFeasibility(const std::string& patternName,
                                       GrammarType sourceGrammar,
                                       ProjectionTarget target) const {
    // Check cache first
    FeasibilityKey key{patternName, target};
    auto it = m_feasibilityCache.find(key);
    if (it != m_feasibilityCache.end()) {
        return it->second;
    }
    
    // If no pattern matcher, assume feasible for same-language projections
    if (!m_patternMatcher) {
        ProjectionTarget sourceTarget = grammarToTarget(sourceGrammar);
        bool feasible = (sourceTarget == target);
        m_feasibilityCache[key] = feasible;
        return feasible;
    }
    
    // For now, we'll use a heuristic based on pattern naming conventions
    // In a full implementation, this would query the PatternMatcher registry
    // TODO: Integrate with PatternMatcher::hasPattern(NodeKind, TargetLanguage)
    
    // Heuristic: patterns with target-specific names
    std::string targetName;
    switch (target) {
        case ProjectionTarget::C:           targetName = "c_"; break;
        case ProjectionTarget::CPP:         targetName = "cpp_"; break;
        case ProjectionTarget::CPP2:        targetName = "cpp2_"; break;
        case ProjectionTarget::MLIR_ARITH:  targetName = "arith"; break;
        case ProjectionTarget::MLIR_CF:     targetName = "cf"; break;
        case ProjectionTarget::MLIR_SCF:    targetName = "scf"; break;
        case ProjectionTarget::MLIR_MEMREF: targetName = "memref"; break;
        case ProjectionTarget::MLIR_FUNC:   targetName = "func"; break;
        default: targetName = ""; break;
    }
    
    // Pattern is feasible if:
    // 1. It's a generic pattern (no specific target prefix)
    // 2. It matches the target prefix
    // 3. Roundtrip is supported
    bool isGeneric = (patternName.find("_") == std::string::npos);
    bool matchesTarget = (patternName.find(targetName) != std::string::npos);
    bool hasRoundtrip = supportsRoundtrip(sourceGrammar, target);
    
    bool feasible = isGeneric || matchesTarget || hasRoundtrip;
    m_feasibilityCache[key] = feasible;
    return feasible;
}

SpeculativeMatch ProjectionOracle::projectToAllTargets(const OrbitMatch& match) const {
    SpeculativeMatch spec(match);
    
    // Track how many targets are feasible for multi-target boost
    size_t feasibleCount = 0;
    
    // Project to all targets (O(T) where T = 8 is constant)
    for (size_t i = 0; i < static_cast<size_t>(ProjectionTarget::COUNT); ++i) {
        ProjectionTarget target = static_cast<ProjectionTarget>(i);
        
        // Quick feasibility check (O(1) with caching)
        bool feasible = checkFeasibility(match.patternName, match.grammarType, target);
        
        if (feasible) {
            feasibleCount++;
            
            // Calculate confidence for this projection
            double confidence = 1.0;
            std::string reason = "Pattern exists";
            
            // Bonus: If target supports roundtrip, boost confidence
            if (supportsRoundtrip(match.grammarType, target)) {
                confidence = 1.0;
                reason = "Roundtrip supported";
                spec.base_confidence *= ROUNDTRIP_BOOST;
            }
            
            spec.projections[i] = ProjectionResult(target, true, confidence, reason);
        } else {
            spec.projections[i] = ProjectionResult(target, false, 0.0, "No lowering pattern");
            // Penalize base confidence if target is missing
            spec.base_confidence *= NO_PATTERN_PENALTY;
        }
    }
    
    // Multi-target boost: patterns that work across multiple targets get confidence boost
    if (feasibleCount > 1) {
        double multiTargetBoost = 1.0 + (feasibleCount - 1) * (MULTI_TARGET_BOOST - 1.0);
        spec.base_confidence *= multiTargetBoost;
    }
    
    // Cap confidence at 1.0
    spec.base_confidence = std::min(1.0, spec.base_confidence);
    
    return spec;
}

bool ProjectionOracle::isProjectionFeasible(const std::string& patternName,
                                           GrammarType sourceGrammar,
                                           ProjectionTarget target) const {
    return checkFeasibility(patternName, sourceGrammar, target);
}

bool ProjectionOracle::supportsRoundtrip(GrammarType source, ProjectionTarget target) const {
    size_t sourceIdx = static_cast<size_t>(source);
    size_t targetIdx = static_cast<size_t>(target);
    
    if (sourceIdx >= static_cast<size_t>(GrammarType::UNKNOWN) ||
        targetIdx >= static_cast<size_t>(ProjectionTarget::COUNT)) {
        return false;
    }
    
    return m_roundtripMatrix[sourceIdx][targetIdx];
}

double ProjectionOracle::getRoundtripBoost(GrammarType source, ProjectionTarget target) const {
    return supportsRoundtrip(source, target) ? ROUNDTRIP_BOOST : 1.0;
}

void ProjectionOracle::clearCache() {
    m_feasibilityCache.clear();
}

ProjectionOracle::CacheStats ProjectionOracle::getCacheStats() const {
    CacheStats stats;
    stats.entries = m_feasibilityCache.size();
    // Note: hits/misses would need to be tracked separately with counters
    return stats;
}

} // namespace cppfort::ir

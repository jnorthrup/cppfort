#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <array>

namespace cppfort {
namespace ir {

/**
 * Unified Orbit Pattern Categories - Common Tree Trunk
 *
 * These categories represent the fundamental AST structures that are
 * shared across C, C++, and CPP2 grammars where deviations are negligible.
 */
enum class UnifiedOrbitCategory : uint8_t {
    // Common Tree Trunk (negligible deviations)
    Declaration = 1,      // Variable/function/type declarations
    Function = 2,         // Function definitions
    TypeDefinition = 3,   // Type/struct/class definitions
    Namespace = 4,        // Namespace/module definitions
    ControlFlow = 5,      // if/while/for statements
    Expression = 6,       // Arithmetic/logical expressions
    ParameterList = 7,    // Function parameter lists
    MemberAccess = 8,     // Object member access
    ScopeResolution = 9,  // Namespace/class scope resolution

    // Significant Deviations (separate branches)
    CPP2_Only = 100,      // Pure CPP2 constructs (: type annotations, etc.)
    CPP_Only = 200,       // Pure C++ constructs (templates, inheritance)

    // Legacy Compatibility
    Legacy_C = 255        // Legacy C patterns for backward compatibility
};

/**
 * Unified Orbit Pattern - Common Tree Trunk Structure
 *
 * Represents patterns that are isomorphic across C, C++, CPP2 where
 * deviations are negligible. Uses unified AST representation.
 */
struct UnifiedOrbitPattern {
    ::std::string name;
    UnifiedOrbitCategory category;
    uint32_t orbit_id;

    // Unified signature patterns (canonical forms across grammars)
    ::std::vector<::std::string> unified_signatures;

    // Grammar-specific variants (for negligible deviations)
    ::std::unordered_map<uint8_t, ::std::vector<::std::string>> grammar_variants;

    // Common properties
    double weight = 1.0;
    uint8_t grammar_modes = 0x07;  // Default: C | CPP | CPP2
    uint16_t lattice_filter = 0xFFFF;
    uint8_t confix_mask = 0x3F;
    ::std::string scope_requirement = "any";

    // Unified AST node type this pattern produces
    ::std::string ast_node_type;

    // Check if this pattern applies to a specific grammar
    bool supportsGrammar(uint8_t grammar_mode) const {
        return (grammar_modes & grammar_mode) != 0;
    }

    // Get signatures for a specific grammar (unified + variants)
    ::std::vector<::std::string> getSignaturesForGrammar(uint8_t grammar_mode) const {
        ::std::vector<::std::string> result = unified_signatures;

        auto it = grammar_variants.find(grammar_mode);
        if (it != grammar_variants.end()) {
            result.insert(result.end(), it->second.begin(), it->second.end());
        }

        return result;
    }
};

/**
 * Unified Orbit Pattern Database
 *
 * Manages patterns that reflect the common tree trunk across grammars.
 * Provides unified AST construction where deviations are negligible.
 */
class UnifiedOrbitDatabase {
private:
    ::std::unordered_map<::std::string, UnifiedOrbitPattern> _patterns;
    ::std::unordered_map<UnifiedOrbitCategory, ::std::vector<UnifiedOrbitPattern>> _patternsByCategory;

public:
    // Load patterns from the groomed cppfort_core_patterns.yaml bundle
    bool loadUnifiedPatterns(const ::std::string& patternFile);

    // Query patterns
    const UnifiedOrbitPattern* getPattern(const ::std::string& name) const;
    ::std::vector<UnifiedOrbitPattern> getPatternsByCategory(UnifiedOrbitCategory category) const;
    ::std::vector<UnifiedOrbitPattern> getPatternsForGrammar(uint8_t grammar_mode) const;

    // Unified AST construction
    ::std::string getUnifiedASTNodeType(const ::std::string& pattern_name) const;

    // Pattern unification metrics
    struct UnificationStats {
        size_t total_patterns;
        size_t unified_patterns;
        size_t grammar_specific_patterns;
        double unification_ratio;  // unified / total
    };
    UnificationStats getUnificationStats() const;
};

} // namespace ir
} // namespace cppfort
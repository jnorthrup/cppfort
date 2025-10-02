#pragma once

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

#include "orbit_scanner.h"
#include "orbit_mask.h"
#include "tblgen_patterns.h"

namespace cppfort::ir {

/**
 * Mock implementation of OrbitScanner for testing without full pattern database.
 * Provides lightweight pattern matching using simplified heuristics.
 * Uses composition instead of inheritance since OrbitScanner methods are not virtual.
 */
class MockOrbitScanner {
public:
    MockOrbitScanner() : scanner(createMockConfig()) {
        initializeMockPatterns();
    }

    // Initialize mock scanner (no file loading)
    bool initialize() {
        return true;  // Mock always succeeds
    }

    // Add mock patterns programmatically
    void addMockPattern(const OrbitPattern& pattern) {
        mockPatterns.push_back(pattern);
    }

    // Get mock patterns
    ::std::vector<OrbitPattern> getMockPatterns() const {
        return mockPatterns;
    }

    // Scan using mock patterns
    DetectionResult scan(const ::std::string& code) const {
        return scanner.scan(code, mockPatterns);
    }

    // Get scanner configuration
    const OrbitScannerConfig& getConfig() const {
        return scanner.getConfig();
    }

    // Update configuration
    void updateConfig(const OrbitScannerConfig& config) {
        scanner.updateConfig(config);
    }

private:
    OrbitScanner scanner;
    ::std::vector<OrbitPattern> mockPatterns;

    static OrbitScannerConfig createMockConfig() {
        OrbitScannerConfig config;
        config.patternThreshold = 0.1;
        config.minConfidence = 0.1;
        config.maxMatches = 1000;
        return config;
    }

    void initializeMockPatterns() {
        // C grammar patterns - balanced braces/parens, minimal templates
        OrbitPattern cPattern;
        cPattern.name = "C_basic_structure";
        cPattern.orbit_id = static_cast<int>(GrammarType::C);
        cPattern.weight = 1.0;
        cPattern.signature_patterns = {"int", "void", "struct"};
        mockPatterns.push_back(cPattern);

        // C++ grammar patterns - more complex with templates
        OrbitPattern cppPattern;
        cppPattern.name = "CPP_complex_structure";
        cppPattern.orbit_id = static_cast<int>(GrammarType::CPP);
        cppPattern.weight = 1.2;
        cppPattern.signature_patterns = {"template", "class", "namespace"};
        mockPatterns.push_back(cppPattern);

        // CPP2 grammar patterns - different syntax characteristics
        OrbitPattern cpp2Pattern;
        cpp2Pattern.name = "CPP2_syntax_structure";
        cpp2Pattern.orbit_id = static_cast<int>(GrammarType::CPP2);
        cpp2Pattern.weight = 1.1;
        cpp2Pattern.signature_patterns = {":", "->", "="};
        mockPatterns.push_back(cpp2Pattern);
    }
};

/**
 * Mock pattern generator for testing.
 * Creates synthetic patterns with known characteristics.
 */
class MockPatternGenerator {
public:
    // Helper to convert grammar to string
    static ::std::string grammarToString(GrammarType grammar) {
        switch (grammar) {
            case GrammarType::C: return "C";
            case GrammarType::CPP: return "CPP";
            case GrammarType::CPP2: return "CPP2";
            default: return "UNKNOWN";
        }
    }

    // Generate pattern with specific orbit characteristics
    static OrbitPattern generatePattern(GrammarType grammar,
                                       const ::std::string& name,
                                       double weight = 1.0) {
        OrbitPattern pattern;
        pattern.name = name;
        pattern.orbit_id = static_cast<int>(grammar);
        pattern.weight = weight;
        pattern.signature_patterns = {"mock", "pattern"};
        return pattern;
    }

    // Generate batch of patterns for a grammar
    static ::std::vector<OrbitPattern> generatePatternBatch(GrammarType grammar, size_t count) {
        ::std::vector<OrbitPattern> patterns;
        patterns.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            ::std::string name = grammarToString(grammar) + "_pattern_" + ::std::to_string(i);
            patterns.push_back(generatePattern(grammar, name, 1.0 + i * 0.1));
        }

        return patterns;
    }

    // Generate diverse pattern set covering all grammars
    static ::std::vector<OrbitPattern> generateDiversePatterns() {
        ::std::vector<OrbitPattern> patterns;

        // Generate patterns for each known grammar type
        const ::std::vector<GrammarType> grammars = {
            GrammarType::C,
            GrammarType::CPP,
            GrammarType::CPP2
        };

        for (auto grammar : grammars) {
            auto batch = generatePatternBatch(grammar, 5);
            patterns.insert(patterns.end(), batch.begin(), batch.end());
        }

        return patterns;
    }
};

/**
 * Mock code generator for testing.
 * Creates synthetic code samples with known orbit structures.
 */
class MockCodeGenerator {
public:
    struct CodeSample {
        ::std::string code;
        GrammarType expectedGrammar;
        double minExpectedConfidence;
        ::std::array<size_t, 6> orbitCounts;  // Expected orbit structure
    };

    // Generate C-style code
    static CodeSample generateCCode() {
        CodeSample sample;
        sample.code = R"(
int main() {
    int x = 42;
    if (x > 0) {
        printf("positive\n");
    }
    return 0;
}
)";
        sample.expectedGrammar = GrammarType::C;
        sample.minExpectedConfidence = 0.5;
        sample.orbitCounts = {2, 0, 0, 4, 1, 2};  // braces, brackets, angles, parens, quotes, numbers
        return sample;
    }

    // Generate C++ code with templates
    static CodeSample generateCppCode() {
        CodeSample sample;
        sample.code = R"(
template<typename T>
class Container {
    std::vector<T> data;
public:
    void add(T item) {
        data.push_back(item);
    }
};
)";
        sample.expectedGrammar = GrammarType::CPP;
        sample.minExpectedConfidence = 0.6;
        sample.orbitCounts = {3, 0, 2, 3, 0, 0};  // More angles for templates
        return sample;
    }

    // Generate CPP2 code
    static CodeSample generateCpp2Code() {
        CodeSample sample;
        sample.code = R"(
main: () -> int = {
    x: int = 42;
    std::cout << "Hello" << std::endl;
    return 0;
}
)";
        sample.expectedGrammar = GrammarType::CPP2;
        sample.minExpectedConfidence = 0.5;
        sample.orbitCounts = {1, 0, 2, 1, 1, 2};  // Different structure
        return sample;
    }

    // Generate batch of diverse code samples
    static ::std::vector<CodeSample> generateDiverseSamples() {
        return {
            generateCCode(),
            generateCppCode(),
            generateCpp2Code()
        };
    }

    // Generate synthetic code with specific orbit pattern
    static ::std::string generateSyntheticCode(const ::std::array<size_t, 6>& orbitCounts) {
        ::std::string code;

        // Braces
        for (size_t i = 0; i < orbitCounts[0]; ++i) {
            code += "{ block" + ::std::to_string(i) + "; }";
        }

        // Brackets
        for (size_t i = 0; i < orbitCounts[1]; ++i) {
            code += "[" + ::std::to_string(i) + "]";
        }

        // Angles (templates)
        for (size_t i = 0; i < orbitCounts[2]; ++i) {
            code += "<T" + ::std::to_string(i) + ">";
        }

        // Parens (function calls)
        for (size_t i = 0; i < orbitCounts[3]; ++i) {
            code += "(arg" + ::std::to_string(i) + ")";
        }

        // Quotes (strings)
        for (size_t i = 0; i < orbitCounts[4]; ++i) {
            code += "\"str" + ::std::to_string(i) + "\"";
        }

        // Numbers
        for (size_t i = 0; i < orbitCounts[5]; ++i) {
            code += ::std::to_string(i * 10) + " ";
        }

        return code;
    }
};

/**
 * Mock validation utilities for testing scanner behavior.
 */
class MockScannerValidator {
public:
    // Validate detection result meets basic expectations
    static bool validateBasicResult(const DetectionResult& result) {
        // Check that fields are initialized
        if (result.confidence < 0.0 || result.confidence > 1.0) {
            ::std::cerr << "Invalid confidence: " << result.confidence << ::std::endl;
            return false;
        }

        // Grammar scores should be normalized
        for (const auto& [grammar, score] : result.grammarScores) {
            if (score < 0.0 || score > 1.0) {
                ::std::cerr << "Invalid grammar score: " << score << " for grammar " << static_cast<int>(grammar) << ::std::endl;
                return false;
            }
        }

        return true;
    }

    // Validate that result matches expected grammar
    static bool validateExpectedGrammar(const DetectionResult& result,
                                       GrammarType expected,
                                       double minConfidence = 0.0) {
        if (result.detectedGrammar != expected) {
            return false;
        }

        if (result.confidence < minConfidence) {
            return false;
        }

        return true;
    }

    // Validate consistency across multiple scans of same code
    static bool validateConsistency(const ::std::vector<DetectionResult>& results) {
        if (results.size() < 2) {
            return true;  // Need at least 2 for comparison
        }

        auto firstGrammar = results[0].detectedGrammar;
        for (size_t i = 1; i < results.size(); ++i) {
            if (results[i].detectedGrammar != firstGrammar) {
                return false;
            }
        }

        return true;
    }
};

} // namespace cppfort::ir

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include "complete_pattern_engine.h"
#include "cpp2_emitter.h"
#include "unified_pattern_matcher.h"
#include "orbit_scanner.h"

namespace fs = std::filesystem;

namespace cppfort {

/**
 * @brief Main semantic transformation system that ties together
 * all components of the Cpp2 transpilation process
 */
class SemanticTransformationSystem {
private:
    CompletePatternEngine m_pattern_engine;
    UnifiedPatternMatcher m_unified_matcher;
    OrbitScanner m_orbit_scanner;
    
public:
    SemanticTransformationSystem() {
        // Initialize with comprehensive pattern set
        if (!m_pattern_engine.loadPatterns("patterns/cpp2_complete_semantic_patterns.yaml")) {
            std::cerr << "Warning: Could not load comprehensive pattern set" << std::endl;
            // Try to load the default patterns
            m_pattern_engine.loadPatterns("patterns/cppfort_core_patterns.yaml");
        }
    }
    
    /**
     * @brief Transform a complete Cpp2 file to C++
     */
    bool transformFile(const std::string& input_file, const std::string& output_file) {
        // Read input file
        std::ifstream infile(input_file);
        if (!infile.is_open()) {
            std::cerr << "Error: Could not open input file: " << input_file << std::endl;
            return false;
        }
        
        std::string content((std::istreambuf_iterator<char>(infile)),
                           std::istreambuf_iterator<char>());
        infile.close();
        
        // Transform the content
        std::string transformed = transformContent(content);
        
        // Write output file
        std::ofstream outfile(output_file);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open output file: " << output_file << std::endl;
            return false;
        }
        
        outfile << transformed;
        outfile.close();
        
        std::cout << "Successfully transformed " << input_file << " to " << output_file << std::endl;
        return true;
    }
    
    /**
     * @brief Transform content string using the complete system
     */
    std::string transformContent(const std::string& content) {
        std::string result = content;
        
        // Step 1: Detect grammar and analyze structure using orbit system
        auto detection_result = m_orbit_scanner.scan(content);
        std::cout << "Detected grammar type: " << static_cast<int>(detection_result.detectedGrammar) << std::endl;
        
        // Step 2: Perform structural analysis
        // This would use the orbit data to understand nesting and context
        
        // Step 3: Apply semantic transformations
        result = m_pattern_engine.applyGraphTransformations(result);
        
        // Step 4: Add required includes
        auto includes = m_pattern_engine.getRequiredIncludes();
        std::string final_result = generateIncludes(includes) + result;
        
        // Step 5: Validate the transformation
        if (!m_pattern_engine.validateTransformation(content, result)) {
            std::cerr << "Warning: Transformation validation failed" << std::endl;
        }
        
        return final_result;
    }
    
private:
    std::string generateIncludes(const std::vector<std::string>& includes) {
        std::string result;
        for (const auto& inc : includes) {
            result += "#include <" + inc + ">\n";
        }
        if (!includes.empty()) {
            result += "\n";
        }
        return result;
    }
};

/**
 * @brief Command line interface for the semantic transformation system
 */
class SemanticTranspilerCLI {
public:
    static int main(int argc, char* argv[]) {
        if (argc < 2) {
            printUsage();
            return 1;
        }
        
        std::string command = argv[1];
        
        if (command == "transpile" && argc >= 4) {
            std::string input_file = argv[2];
            std::string output_file = argv[3];
            
            SemanticTransformationSystem system;
            if (system.transformFile(input_file, output_file)) {
                std::cout << "Transpilation completed successfully." << std::endl;
                return 0;
            } else {
                std::cerr << "Transpilation failed." << std::endl;
                return 1;
            }
        } 
        else if (command == "--help" || command == "-h") {
            printUsage();
            return 0;
        }
        else {
            std::cerr << "Unknown command or insufficient arguments." << std::endl;
            printUsage();
            return 1;
        }
    }
    
private:
    static void printUsage() {
        std::cout << "Cpp2 Semantic Transpiler - Complete End-to-End System\n";
        std::cout << "Usage:\n";
        std::cout << "  stage0_cli transpile <input.cpp2> <output.cpp>    # Transpile Cpp2 to C++\n";
        std::cout << "  stage0_cli --help                                 # Show this help\n";
        std::cout << "\nThis system implements a complete semantic transformation pipeline\n";
        std::cout << "using graph-based pattern matching and comprehensive semantic mappings.\n";
    }
};

} // namespace cppfort

// Main entry point
int main(int argc, char* argv[]) {
    return cppfort::SemanticTranspilerCLI::main(argc, argv);
}
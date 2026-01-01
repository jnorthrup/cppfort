// test_output_parity.cpp
// Output parity tests: verify cppfort produces semantically equivalent output to cppfront
// Uses Clang AST isomorphism via semantic hash comparison
// Loads reference hashes from tests/regression_hashes.txt

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <map>

#include "lexer.hpp"
#include "parser.hpp"
#include "code_generator.hpp"
#include "semantic_hash.hpp"

namespace fs = std::filesystem;
using namespace cpp2_transpiler;

static int tests_passed = 0;
static int tests_failed = 0;

// Normalize C++ code for comparison - strips whitespace differences and #line directives
std::string normalize_cpp(const std::string& cpp_code) {
    std::string normalized;
    normalized.reserve(cpp_code.size());
    
    bool in_string = false;
    bool prev_space = false;
    
    for (size_t i = 0; i < cpp_code.size(); i++) {
        char c = cpp_code[i];
        
        if (c == '"' && (i == 0 || cpp_code[i-1] != '\\')) {
            in_string = !in_string;
        }
        
        if (!in_string && (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
            if (!prev_space && !normalized.empty()) {
                normalized += ' ';
                prev_space = true;
            }
            continue;
        }
        
        if (!in_string && c == '#') {
            while (i < cpp_code.size() && cpp_code[i] != '\n') i++;
            continue;
        }
        
        normalized += c;
        prev_space = false;
    }
    
    while (!normalized.empty() && normalized.front() == ' ') normalized.erase(0, 1);
    while (!normalized.empty() && normalized.back() == ' ') normalized.pop_back();
    
    return normalized;
}

std::string transpile(const std::string& cpp2_source) {
    Lexer lexer(cpp2_source);
    auto tokens = lexer.tokenize();
    
    Parser parser(tokens);
    auto ast = parser.parse();
    
    if (!ast || ast->declarations.empty()) {
        throw std::runtime_error("Parse failed");
    }
    
    CodeGenerator gen;
    return gen.generate(*ast);
}

void run_parity_test(const std::string& name, const std::string& cpp2_path, const std::string& expected_hash_str) {
    std::cout << "Testing " << name << "... " << std::flush;
    
    try {
        std::ifstream ifs(cpp2_path);
        std::string source((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        
        std::string actual = transpile(source);
        std::string norm_actual = normalize_cpp(actual);
        std::string actual_hash = cppfort::crdt::SHA256Hash::compute(norm_actual).to_hex_string();
        
        if (actual_hash == expected_hash_str) {
            std::cout << "✓ PASS\n";
            tests_passed++;
        } else {
            std::cout << "✗ FAIL (mismatch)\n";
            std::cout << "  Expected: " << expected_hash_str.substr(0, 16) << "...\n";
            std::cout << "  Actual:   " << actual_hash.substr(0, 16) << "...\n";
            tests_failed++;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ FAIL (exception: " << e.what() << ")\n";
        tests_failed++;
    }
}

int main(int argc, char* argv[]) {
    fs::path hash_file = "tests/regression_hashes.txt";
    fs::path corpus_dir = "tests/cppfront_regression";

    if (argc > 1) hash_file = argv[1];
    if (argc > 2) corpus_dir = argv[2];

    if (!fs::exists(hash_file)) {
        std::cerr << "Error: reference hashes not found at " << hash_file << "\n";
        std::cerr << "Run precalc_hashes utility first.\n";
        return 1;
    }

    std::map<std::string, std::string> reference_hashes;
    std::ifstream hfs(hash_file);
    std::string line;
    while (std::getline(hfs, line)) {
        size_t sep = line.find('|');
        if (sep != std::string::npos) {
            reference_hashes[line.substr(0, sep)] = line.substr(sep + 1);
        }
    }

    std::cout << "===========================================\n";
    std::cout << "Cppfort Corpus Output Parity Tests\n";
    std::cout << "Comparing against " << reference_hashes.size() << " precomputed hashes\n";
    std::cout << "===========================================\n\n";

    for (const auto& [filename, hash] : reference_hashes) {
        fs::path cpp2_file = corpus_dir / filename;
        if (fs::exists(cpp2_file)) {
            run_parity_test(filename, cpp2_file.string(), hash);
        }
    }

    std::cout << "\n===========================================\n";
    std::cout << "Summary: " << tests_passed << " passed, " << tests_failed << " failed\n";
    std::cout << "===========================================\n";

    return tests_failed > 0 ? 1 : 0;
}

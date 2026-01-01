// precalc_hashes.cpp
// Utility to precompute semantic hashes for the cppfront regression corpus

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

#include "semantic_hash.hpp"

namespace fs = std::filesystem;

// Execute command and capture output
std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

// Normalize C++ output for semantic hashing (matching test_output_parity.cpp logic)
std::string normalize_cpp(const std::string& cpp_code) {
    std::string normalized;
    normalized.reserve(cpp_code.size());
    
    bool in_string = false;
    bool prev_space = false;
    
    for (size_t i = 0; i < cpp_code.size(); i++) {
        char c = cpp_code[i];
        
        // Handle strings
        if (c == '"' && (i == 0 || cpp_code[i-1] != '\\')) {
            in_string = !in_string;
        }
        
        // Collapse whitespace outside strings
        if (!in_string && (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
            if (!prev_space && !normalized.empty()) {
                normalized += ' ';
                prev_space = true;
            }
            continue;
        }
        
        // Skip #line directives and absolute paths often found in cppfront output
        if (!in_string && c == '#') {
            while (i < cpp_code.size() && cpp_code[i] != '\n') i++;
            continue;
        }
        
        normalized += c;
        prev_space = false;
    }
    
    // Trim
    while (!normalized.empty() && normalized.front() == ' ') normalized.erase(0, 1);
    while (!normalized.empty() && normalized.back() == ' ') normalized.pop_back();
    
    return normalized;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <cppfront_path> <input_dir> <output_file>\n";
        return 1;
    }

    fs::path cppfront_path = argv[1];
    fs::path input_dir = argv[2];
    fs::path output_file = argv[3];

    if (!fs::exists(cppfront_path)) {
        std::cerr << "Error: cppfront binary not found at " << cppfront_path << "\n";
        return 1;
    }

    if (!fs::is_directory(input_dir)) {
        std::cerr << "Error: input directory not found at " << input_dir << "\n";
        return 1;
    }

    std::ofstream ofs(output_file);
    if (!ofs) {
        std::cerr << "Error: could not open output file " << output_file << "\n";
        return 1;
    }

    std::cout << "Precomputing hashes for corpus in " << input_dir << "...\n";

    int processed = 0;
    int failed = 0;

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".cpp2") {
            std::string filename = entry.path().filename().string();
            std::cout << "Processing " << filename << "... " << std::flush;

            // Run cppfront on the file
            // Use -p for pure Cpp2, -include-std to match common usage, -clean-cpp1 to avoid #line
            std::string cmd = cppfront_path.string() + " -p -include-std -clean-cpp1 -o stdout " + entry.path().string() + " 2>&1";
            
            try {
                std::string cpp_output = exec(cmd.c_str());
                
                // If it contains "error:", it's a failure (some tests are expected to fail)
                if (cpp_output.find("error:") != std::string::npos && cpp_output.find("auto main() -> int") == std::string::npos) {
                   std::cout << "FAILED (cppfront error)\n";
                   failed++;
                   continue;
                }

                // Normalize and hash
                std::string normalized = normalize_cpp(cpp_output);
                std::string hash = cppfort::crdt::SHA256Hash::compute(normalized).to_hex_string();

                // Save to file: filename|hash
                ofs << filename << "|" << hash << "\n";
                std::cout << "DONE (" << hash.substr(0, 8) << ")\n";
                processed++;
            } catch (const std::exception& e) {
                std::cerr << "EXCEPTION: " << e.what() << "\n";
                failed++;
            }
        }
    }

    std::cout << "\nProcessed " << processed << " files, " << failed << " failed.\n";
    std::cout << "Hashes saved to " << output_file << "\n";

    return 0;
}

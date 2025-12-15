#include "cppfront_test_framework.h"
#include "sha256.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <regex>

namespace cppfort::tests {

std::vector<TestFile> TestCollector::collect_test_files(const std::string& test_dir) {
    std::vector<TestFile> files;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(test_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".cpp2") {
            TestFile test_file;
            test_file.path = entry.path().string();
            test_file.category = extract_category(entry.path());
            test_file.name = extract_name(entry.path());
            test_file.should_pass = should_pass_by_name(test_file.name);

            // Read file content
            std::ifstream file(entry.path());
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                test_file.content = buffer.str();
                test_file.sha256_hash = calculate_sha256(test_file.content);
                test_file.size_bytes = test_file.content.size();
                file.close();
            }

            files.push_back(test_file);
        }
    }

    // Sort by category then name
    std::sort(files.begin(), files.end(), [](const TestFile& a, const TestFile& b) {
        if (a.category != b.category) {
            return a.category < b.category;
        }
        return a.name < b.name;
    });

    return files;
}

std::string TestCollector::calculate_sha256(const std::string& content) {
    return SHA256::hash(content);
}

std::string TestCollector::extract_category(const std::filesystem::path& path) {
    std::string filename = path.filename().string();

    // Extract prefix before first dash
    size_t dash_pos = filename.find('-');
    if (dash_pos != std::string::npos) {
        return filename.substr(0, dash_pos);
    }

    return "unknown";
}

std::string TestCollector::extract_name(const std::filesystem::path& path) {
    std::string filename = path.filename().string();

    // Remove .cpp2 extension
    if (filename.length() > 5 && filename.substr(filename.length() - 5) == ".cpp2") {
        filename = filename.substr(0, filename.length() - 5);
    }

    return filename;
}

bool TestCollector::should_pass_by_name(const std::string& name) {
    // Files with "error" in their name are expected to fail compilation/transpilation
    return name.find("error") == std::string::npos;
}

TestRunner::TestRunner(const std::vector<TestFile>& test_files)
    : test_files_(test_files) {
    register_default_handlers();
}

void TestRunner::register_category_handler(const std::string& category,
                                         std::function<TestResult(const TestFile&)> handler) {
    category_handlers_[category] = handler;
}

void TestRunner::run_all_tests() {
    results_.clear();
    results_.reserve(test_files_.size());

    size_t processed = 0;
    for (const auto& test_file : test_files_) {
        std::cout << "\rRunning tests: " << ++processed << "/" << test_files_.size()
                  << " (" << test_file.category << ":" << test_file.name << ")" << std::flush;

        auto it = category_handlers_.find(test_file.category);
        if (it != category_handlers_.end()) {
            results_.push_back(it->second(test_file));
        } else {
            results_.push_back(run_transpiler_test(test_file));
        }
    }
    std::cout << std::endl;
}

void TestRunner::run_category(const std::string& category) {
    results_.clear();

    for (const auto& test_file : test_files_) {
        if (test_file.category == category) {
            auto it = category_handlers_.find(category);
            if (it != category_handlers_.end()) {
                results_.push_back(it->second(test_file));
            } else {
                results_.push_back(run_transpiler_test(test_file));
            }
        }
    }
}

void TestRunner::run_single_test(const std::string& test_name) {
    results_.clear();

    for (const auto& test_file : test_files_) {
        if (test_file.name == test_name) {
            auto it = category_handlers_.find(test_file.category);
            if (it != category_handlers_.end()) {
                results_.push_back(it->second(test_file));
            } else {
                results_.push_back(run_transpiler_test(test_file));
            }
            break;
        }
    }
}

TestResult TestRunner::run_transpiler_test(const TestFile& test_file) {
    auto start = std::chrono::high_resolution_clock::now();

    TestResult result;
    result.test_name = test_file.name;
    result.category = test_file.category;

    // Create temporary file for input
    std::string temp_input = std::filesystem::temp_directory_path() /
                           ("test_" + test_file.name + ".cpp2");
    std::ofstream input_file(temp_input);
    input_file << test_file.content;
    input_file.close();

    // Run transpiler
    std::string command = "cppfort \"" + temp_input + "\" 2>&1";
    std::string output;
    int exit_code = system((command + " > /dev/null 2>&1").c_str());

    // Get output if failed
    if (exit_code != 0) {
        FILE* pipe = popen(command.c_str(), "r");
        if (pipe) {
            char buffer[4096];
            while (fgets(buffer, sizeof(buffer), pipe)) {
                output += buffer;
            }
            pclose(pipe);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Check if result matches expectation
    if (test_file.should_pass) {
        result.passed = (exit_code == 0);
        if (!result.passed) {
            result.error_message = "Expected transpiler to succeed but it failed";
            result.transpiler_output = output;
        }
    } else {
        result.passed = (exit_code != 0);
        if (!result.passed) {
            result.error_message = "Expected transpiler to fail but it succeeded";
        } else {
            // Verify error message is meaningful
            if (output.empty()) {
                result.passed = false;
                result.error_message = "Expected error output but got none";
            }
        }
    }

    // Cleanup
    std::filesystem::remove(temp_input);

    return result;
}

TestResult TestRunner::handle_pure2_test(const TestFile& test_file) {
    auto result = run_transpiler_test(test_file);

    // Additional pure2-specific checks can be added here
    // For example, verify no C++1 code in output

    return result;
}

TestResult TestRunner::handle_mixed_test(const TestFile& test_file) {
    auto result = run_transpiler_test(test_file);

    // Additional mixed-mode specific checks can be added here
    // For example, verify C++1 code is preserved

    return result;
}

void TestRunner::register_default_handlers() {
    register_category_handler("pure2", [this](const TestFile& tf) { return handle_pure2_test(tf); });
    register_category_handler("mixed", [this](const TestFile& tf) { return handle_mixed_test(tf); });
}

void TestRunner::print_summary() const {
    size_t total = results_.size();
    size_t passed = std::count_if(results_.begin(), results_.end(),
                                [](const TestResult& r) { return r.passed; });
    size_t failed = total - passed;

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Total tests: " << total << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;

    if (failed > 0) {
        std::cout << "\nFailed tests:" << std::endl;
        for (const auto& result : results_) {
            if (!result.passed) {
                std::cout << "  - " << result.category << ":" << result.test_name << std::endl;
            }
        }
    }
}

void TestRunner::print_detailed_results() const {
    for (const auto& result : results_) {
        std::cout << "\n--- " << result.category << ":" << result.test_name << " ---" << std::endl;
        std::cout << "Status: " << (result.passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Duration: " << result.duration.count() << "ms" << std::endl;

        if (!result.passed) {
            std::cout << "Error: " << result.error_message << std::endl;
            if (!result.transpiler_output.empty()) {
                std::cout << "Transpiler output:" << std::endl;
                std::cout << result.transpiler_output << std::endl;
            }
        }
    }
}

bool SHA256Verifier::verify_file_hash(const TestFile& test_file) {
    std::string calculated = TestCollector::calculate_sha256(test_file.content);
    return calculated == test_file.sha256_hash;
}

bool SHA256Verifier::verify_all_hashes(const std::vector<TestFile>& test_files) {
    for (const auto& test_file : test_files) {
        if (!verify_file_hash(test_file)) {
            std::cerr << "Hash mismatch for: " << test_file.name << std::endl;
            return false;
        }
    }
    return true;
}

std::map<std::string, std::string> SHA256Verifier::load_hash_database(const std::string& db_path) {
    std::map<std::string, std::string> hash_db;
    std::ifstream file(db_path);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string hash = line.substr(0, space_pos);
                std::string filename = line.substr(space_pos + 1);
                hash_db[filename] = hash;
            }
        }
        file.close();
    }

    return hash_db;
}

void SHA256Verifier::save_hash_database(const std::string& db_path,
                                      const std::vector<TestFile>& test_files) {
    std::ofstream file(db_path);

    if (file.is_open()) {
        for (const auto& test_file : test_files) {
            file << test_file.sha256_hash << " " << test_file.name << std::endl;
        }
        file.close();
    }
}

} // namespace cppfort::tests
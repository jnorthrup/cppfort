#include "cppfront_test_framework.h"
#include <iostream>
#include <filesystem>

using namespace cppfort::tests;

int main() {
    std::string test_dir = "/tmp/cppfront-test/regression-tests";
    std::string hash_db_file = "test_hashes.txt";

    std::cout << "Generating test hash database..." << std::endl;

    // Collect test files
    auto test_files = TestCollector::collect_test_files(test_dir);

    std::cout << "Found " << test_files.size() << " test files" << std::endl;

    // Save hash database
    SHA256Verifier::save_hash_database(hash_db_file, test_files);

    std::cout << "Hash database saved to: " << hash_db_file << std::endl;

    // Generate statistics
    std::map<std::string, size_t> category_counts;
    std::map<std::string, size_t> category_errors;
    size_t total_size = 0;

    for (const auto& test : test_files) {
        category_counts[test.category]++;
        total_size += test.size_bytes;
        if (!test.should_pass) {
            category_errors[test.category]++;
        }
    }

    std::cout << "\n=== Test Statistics ===" << std::endl;
    std::cout << "Total files: " << test_files.size() << std::endl;
    std::cout << "Total size: " << total_size << " bytes" << std::endl;

    for (const auto& [category, count] : category_counts) {
        std::cout << "\n" << category << ": " << count << " files";
        if (category_errors[category] > 0) {
            std::cout << " (" << category_errors[category] << " expected to fail)";
        }
        std::cout << std::endl;
    }

    return 0;
}
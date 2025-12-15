#include "cppfront_test_framework.h"
#include <iostream>
#include <filesystem>
#include <getopt.h>
#include <fstream>

using namespace cppfort::tests;

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -t, --test-dir <dir>   Test directory path (default: /tmp/cppfront-test/regression-tests)\n"
              << "  -c, --category <cat>   Run only specific category (e.g., pure2, mixed)\n"
              << "  -s, --single <test>    Run only single test by name\n"
              << "  -o, --output <file>    Output results to file\n"
              << "  -v, --verbose          Verbose output\n"
              << "  --hash-db <file>       Hash database file path\n"
              << "  --verify-hashes        Verify file hashes before running\n"
              << "  --update-hashes        Update hash database\n"
              << "  --list-tests           List all available tests\n"
              << "  -h, --help             Show this help\n"
              << std::endl;
}

void list_tests(const std::vector<TestFile>& test_files) {
    std::cout << "\n=== Available Tests ===\n";
    std::map<std::string, std::vector<std::string>> by_category;

    for (const auto& test : test_files) {
        by_category[test.category].push_back(test.name);
    }

    for (const auto& [category, tests] : by_category) {
        std::cout << "\n" << category << " (" << tests.size() << " tests):\n";
        for (const auto& test : tests) {
            std::cout << "  - " << test;
            if (test.find("error") != std::string::npos) {
                std::cout << " (expected to fail)";
            }
            std::cout << "\n";
        }
    }
    std::cout << "\nTotal: " << test_files.size() << " tests\n" << std::endl;
}

int main(int argc, char** argv) {
    std::string test_dir = "/tmp/cppfront-test/regression-tests";
    std::string category;
    std::string single_test;
    std::string output_file;
    std::string hash_db_file = "test_hashes.txt";
    bool verbose = false;
    bool verify_hashes = false;
    bool update_hashes = false;
    bool list_tests_only = false;

    // Parse command line arguments
    option long_options[] = {
        {"test-dir", required_argument, 0, 't'},
        {"category", required_argument, 0, 'c'},
        {"single", required_argument, 0, 's'},
        {"output", required_argument, 0, 'o'},
        {"verbose", no_argument, 0, 'v'},
        {"hash-db", required_argument, 0, 1001},
        {"verify-hashes", no_argument, 0, 1002},
        {"update-hashes", no_argument, 0, 1003},
        {"list-tests", no_argument, 0, 1004},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "t:c:s:o:vh", long_options, nullptr)) != -1) {
        switch (c) {
            case 't':
                test_dir = optarg;
                break;
            case 'c':
                category = optarg;
                break;
            case 's':
                single_test = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'v':
                verbose = true;
                break;
            case 1001:
                hash_db_file = optarg;
                break;
            case 1002:
                verify_hashes = true;
                break;
            case 1003:
                update_hashes = true;
                break;
            case 1004:
                list_tests_only = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case '?':
                return 1;
            default:
                break;
        }
    }

    // Check if test directory exists
    if (!std::filesystem::exists(test_dir)) {
        std::cerr << "Error: Test directory does not exist: " << test_dir << std::endl;
        return 1;
    }

    // Collect test files
    std::cout << "Collecting test files from: " << test_dir << std::endl;
    auto test_files = TestCollector::collect_test_files(test_dir);

    if (test_files.empty()) {
        std::cerr << "Error: No .cpp2 test files found in " << test_dir << std::endl;
        return 1;
    }

    std::cout << "Found " << test_files.size() << " test files\n" << std::endl;

    // List tests if requested
    if (list_tests_only) {
        list_tests(test_files);
        return 0;
    }

    // Handle hash database
    if (verify_hashes) {
        std::cout << "Verifying file hashes..." << std::endl;
        if (!SHA256Verifier::verify_all_hashes(test_files)) {
            std::cerr << "Error: Hash verification failed!" << std::endl;
            return 1;
        }
        std::cout << "All hashes verified successfully!" << std::endl;
    }

    if (update_hashes) {
        std::cout << "Updating hash database: " << hash_db_file << std::endl;
        SHA256Verifier::save_hash_database(hash_db_file, test_files);
        std::cout << "Hash database updated!" << std::endl;
    }

    // Create test runner
    TestRunner runner(test_files);

    // Run tests
    std::cout << "Running tests..." << std::endl;

    if (!single_test.empty()) {
        std::cout << "Running single test: " << single_test << std::endl;
        runner.run_single_test(single_test);
    } else if (!category.empty()) {
        std::cout << "Running category: " << category << std::endl;
        runner.run_category(category);
    } else {
        runner.run_all_tests();
    }

    // Get results
    const auto& results = runner.get_results();

    // Print summary
    runner.print_summary();

    // Print detailed results if verbose
    if (verbose) {
        runner.print_detailed_results();
    }

    // Save results to file if requested
    if (!output_file.empty()) {
        std::ofstream out_file(output_file);
        if (out_file.is_open()) {
            out_file << "# Cppfront Regression Test Results\n";
            out_file << "# Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n\n";

            size_t total = results.size();
            size_t passed = std::count_if(results.begin(), results.end(),
                                        [](const TestResult& r) { return r.passed; });

            out_file << "TOTAL," << total << "\n";
            out_file << "PASSED," << passed << "\n";
            out_file << "FAILED," << (total - passed) << "\n\n";

            out_file << "CATEGORY,TEST_NAME,STATUS,DURATION_MS,ERROR_MESSAGE\n";
            for (const auto& result : results) {
                out_file << result.category << ","
                        << result.test_name << ","
                        << (result.passed ? "PASSED" : "FAILED") << ","
                        << result.duration.count() << ","
                        << result.error_message << "\n";
            }

            out_file.close();
            std::cout << "Results saved to: " << output_file << std::endl;
        }
    }

    // Return appropriate exit code
    size_t failed = std::count_if(results.begin(), results.end(),
                                 [](const TestResult& r) { return !r.passed; });

    return failed > 0 ? 1 : 0;
}
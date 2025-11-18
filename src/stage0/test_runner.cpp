// Test Runner Infrastructure - Execute and validate all stage0 tests
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <array>
#include <memory>

struct TestResult {
    std::string test_name;
    bool passed;
    std::string output;
    std::string error_message;
    int exit_code;
};

class TestRunner {
private:
    std::vector<TestResult> results_;
    std::string test_dir_;

    // Execute a command and capture output
    std::pair<std::string, int> execute_command(const std::string& cmd) {
        std::array<char, 128> buffer;
        std::string result;
        int exit_code = 0;

        // Use popen to execute command and capture output
        // Ensure stdin is closed for spawned test processes to avoid blocking
        // and follow the CODING_STANDARDS.md guideline "DO NOT RUN A TEST PROCESS WITHOUT TIMEOUT or '< /dev/null'".
        FILE* pipe = popen((cmd + " < /dev/null 2>&1").c_str(), "r");
        if (!pipe) {
            return {"ERROR: Failed to execute command", -1};
        }

        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }

        exit_code = pclose(pipe);
        return {result, WEXITSTATUS(exit_code)};
    }

    bool file_exists(const std::string& path) {
        std::ifstream f(path);
        return f.good();
    }

public:
    TestRunner(const std::string& test_dir = "./") : test_dir_(test_dir) {}

    // Run a single test executable
    bool run_test(const std::string& test_binary) {
        TestResult result;
        result.test_name = test_binary;
        result.passed = false;
        result.exit_code = -1;

        std::string full_path = test_dir_ + test_binary;

        // Check if test binary exists
        if (!file_exists(full_path)) {
            result.error_message = "Test binary not found: " + full_path;
            results_.push_back(result);
            return false;
        }

        std::cout << "Running: " << test_binary << "... ";
        std::cout.flush();

        // Execute the test
        auto [output, exit_code] = execute_command(full_path);
        result.output = output;
        result.exit_code = exit_code;

        // Test passes if exit code is 0
        if (exit_code == 0) {
            result.passed = true;
            std::cout << "PASS\n";
        } else {
            result.error_message = "Non-zero exit code: " + std::to_string(exit_code);
            std::cout << "FAIL (exit code " << exit_code << ")\n";
        }

        results_.push_back(result);
        return result.passed;
    }

    // Compare actual output with expected output
    bool compare_output(const std::string& expected, const std::string& actual) {
        // Trim whitespace from both strings for comparison
        auto trim = [](const std::string& s) -> std::string {
            auto start = s.find_first_not_of(" \t\r\n");
            auto end = s.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) return "";
            return s.substr(start, end - start + 1);
        };

        std::string exp_trimmed = trim(expected);
        std::string act_trimmed = trim(actual);

        return exp_trimmed == act_trimmed;
    }

    // Load expected output from file
    std::string load_expected_output(const std::string& test_name) {
        std::string expected_file = test_dir_ + "expected/" + test_name + ".expected";
        std::ifstream file(expected_file);
        if (!file.good()) {
            return "";  // No expected output file
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    // Run test with output comparison
    bool run_test_with_validation(const std::string& test_binary) {
        TestResult result;
        result.test_name = test_binary;
        result.passed = false;
        result.exit_code = -1;

        std::string full_path = test_dir_ + test_binary;

        if (!file_exists(full_path)) {
            result.error_message = "Test binary not found: " + full_path;
            results_.push_back(result);
            return false;
        }

        std::cout << "Running: " << test_binary << "... ";
        std::cout.flush();

        auto [output, exit_code] = execute_command(full_path);
        result.output = output;
        result.exit_code = exit_code;

        // Load expected output if available
        std::string expected = load_expected_output(test_binary);

        if (!expected.empty()) {
            // Compare output
            if (compare_output(expected, output)) {
                result.passed = true;
                std::cout << "PASS (output matches expected)\n";
            } else {
                result.error_message = "Output mismatch";
                std::cout << "FAIL (output mismatch)\n";
            }
        } else {
            // No expected output, just check exit code
            if (exit_code == 0) {
                result.passed = true;
                std::cout << "PASS\n";
            } else {
                result.error_message = "Non-zero exit code: " + std::to_string(exit_code);
                std::cout << "FAIL (exit code " << exit_code << ")\n";
            }
        }

        results_.push_back(result);
        return result.passed;
    }

    // Print summary of all test results
    void print_summary() {
        std::cout << "\n=== TEST SUMMARY ===\n";

        int passed = 0;
        int failed = 0;

        for (const auto& result : results_) {
            if (result.passed) {
                passed++;
            } else {
                failed++;
                std::cout << "FAILED: " << result.test_name;
                if (!result.error_message.empty()) {
                    std::cout << " - " << result.error_message;
                }
                std::cout << "\n";
            }
        }

        std::cout << "\nTotal: " << results_.size() << " tests\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";

        if (failed == 0) {
            std::cout << "\n✓ All tests passed!\n";
        } else {
            std::cout << "\n✗ Some tests failed\n";
        }
    }

    // Print detailed results for failed tests
    void print_detailed_failures() {
        bool has_failures = false;

        for (const auto& result : results_) {
            if (!result.passed) {
                if (!has_failures) {
                    std::cout << "\n=== DETAILED FAILURE OUTPUT ===\n";
                    has_failures = true;
                }

                std::cout << "\n--- " << result.test_name << " ---\n";
                std::cout << "Exit code: " << result.exit_code << "\n";
                std::cout << "Error: " << result.error_message << "\n";
                std::cout << "Output:\n" << result.output << "\n";
            }
        }
    }

    // Get total number of passed tests
    int get_pass_count() const {
        int count = 0;
        for (const auto& result : results_) {
            if (result.passed) count++;
        }
        return count;
    }

    // Get total number of tests
    int get_total_count() const {
        return results_.size();
    }

    // Clear all results
    void clear() {
        results_.clear();
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== Stage0 Test Runner ===\n\n";

    // Determine test directory from argv[0] or use current directory
    std::string test_dir = "./";
    if (argc > 1) {
        test_dir = argv[1];
        if (test_dir.back() != '/') {
            test_dir += '/';
        }
    }

    TestRunner runner(test_dir);

    // List of all stage0 tests
    std::vector<std::string> tests = {
        "test_reality_check",
        "test_confix_depth",
        "test_correlation",
        "test_pattern_match",
        "test_tblgen_integration",
        "test_depth_matcher"
    };

    // Run all tests
    for (const auto& test : tests) {
        runner.run_test(test);
    }

    // Print summary
    runner.print_summary();

    // Print detailed failures if requested
    if (argc > 2 && std::string(argv[2]) == "--verbose") {
        runner.print_detailed_failures();
    }

    // Return non-zero if any tests failed
    return (runner.get_pass_count() == runner.get_total_count()) ? 0 : 1;
}

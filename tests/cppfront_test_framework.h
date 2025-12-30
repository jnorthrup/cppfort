#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <map>
#include <functional>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace cppfort::tests {

// Test timeout standard: all tests MUST complete within 15 seconds
inline constexpr std::chrono::milliseconds TEST_TIMEOUT_MS{15000};

struct TestFile {
    std::string path;
    std::string category;
    std::string name;
    std::string content;
    std::string sha256_hash;
    bool should_pass;
    size_t size_bytes;
};

struct TestResult {
    std::string test_name;
    bool passed;
    std::string error_message;
    std::string transpiler_output;
    std::string actual_output;  // stdout from running compiled binary
    std::chrono::milliseconds duration;
    std::string category;
};

class TestCollector {
public:
    static std::vector<TestFile> collect_test_files(const std::string& test_dir);
    static std::string calculate_sha256(const std::string& content);
    static std::string extract_category(const std::filesystem::path& path);
    static std::string extract_name(const std::filesystem::path& path);
    static bool should_pass_by_name(const std::string& name);
};

class TestRunner {
private:
    std::vector<TestFile> test_files_;
    std::map<std::string, std::function<TestResult(const TestFile&)>> category_handlers_;
    std::vector<TestResult> results_;
    std::string transpiler_path_;

public:
    TestRunner(const std::vector<TestFile>& test_files);
    TestRunner(const std::vector<TestFile>& test_files, const std::string& transpiler_path);

    void register_category_handler(const std::string& category,
                                 std::function<TestResult(const TestFile&)> handler);

    void run_all_tests();
    void run_category(const std::string& category);
    void run_single_test(const std::string& test_name);

    const std::vector<TestResult>& get_results() const { return results_; }

    void print_summary() const;
    void print_detailed_results() const;

private:
    TestResult run_transpiler_test(const TestFile& test_file);
    TestResult handle_pure2_test(const TestFile& test_file);
    TestResult handle_mixed_test(const TestFile& test_file);

    // Helper function to execute a command with timeout
    int execute_command(const std::string& command, std::string& output, int timeout_ms);

    void register_default_handlers();

    // Find and validate transpiler binary
    static std::string find_transpiler(const std::string& hint = "");
};

class SHA256Verifier {
public:
    static bool verify_file_hash(const TestFile& test_file);
    static bool verify_all_hashes(const std::vector<TestFile>& test_files);
    static std::map<std::string, std::string> load_hash_database(const std::string& db_path);
    static void save_hash_database(const std::string& db_path,
                                 const std::vector<TestFile>& test_files);
};

} // namespace cppfort::tests
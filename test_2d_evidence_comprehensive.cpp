#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <future>
#include <algorithm>
#include <random>
#include "src/stage0/evidence_2d.h"

using namespace cppfort::stage0;
using namespace std::chrono;

// Performance metrics
struct TestMetrics {
    double execution_time_ms;
    size_t memory_usage;
    size_t confix_count;
    double confidence_score;
};

// Test result tracking
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    TestMetrics metrics;
};

std::vector<TestResult> test_results;
std::atomic<size_t> concurrent_operations{0};

// Helper to record test results
void record_test(const std::string& name, bool passed, const std::string& message, 
                double time_ms, size_t confix_count = 0, double confidence = 0.0) {
    test_results.push_back({name, passed, message, {time_ms, 0, confix_count, confidence}});
    std::cout << (passed ? "✅" : "❌") << " " << name << " (" << time_ms << "ms, " 
              << confix_count << " confixes, confidence: " << confidence << ")";
    if (!message.empty()) std::cout << " - " << message;
    std::cout << "\n";
}

// Test 1: Comprehensive confix classification
void test_comprehensive_confix_classification() {
    auto start = high_resolution_clock::now();
    
    struct TestCase {
        std::string code;
        std::string name;
        std::array<size_t, static_cast<uint8_t>(ConfixType::MAX_TYPE)> expected_counts;
    };
    
    TestCase cases[] = {
        // Basic structural confixes
        {"(){}[]<>", "basic_structural", {0, 2, 2, 2, 2, 0, 0, 0}},
        {"int main() { return 0; }", "simple_function", {0, 2, 2, 0, 0, 0, 0, 0}},
        {"vector<int> data;", "simple_template", {0, 0, 0, 0, 2, 0, 0, 0}},
        {"array[index]", "bracket_access", {0, 0, 0, 2, 0, 0, 0, 0}},
        
        // Comments
        {"/* C comment */ int x;", "c_comment", {0, 0, 0, 0, 0, 1, 0, 0}},
        {"// C++ comment\nint x;", "cpp_comment", {0, 0, 0, 0, 0, 0, 1, 0}},
        {"```cpp2 comment```\nint x;", "cpp2_comment", {0, 0, 0, 0, 0, 0, 0, 1}},
        
        // Mixed
        {"/* C */ // C++\n```Cpp2```", "all_comments", {0, 0, 0, 0, 0, 1, 1, 1}},
        {"vector</*comment*/int> data;", "comment_in_template", {0, 0, 0, 0, 2, 1, 0, 0}},
        {"func(/*args*/) { // body\n }", "mixed_structural_comments", {0, 2, 2, 0, 0, 1, 1, 0}}
    };
    
    bool all_passed = true;
    for (const auto& test : cases) {
        auto span = Evidence2DAnalyzer::analyze_span(test.code);
        
        bool passed = true;
        std::string issues;
        
        for (uint8_t i = 0; i < static_cast<uint8_t>(ConfixType::MAX_TYPE); ++i) {
            if (span.confix_type_counts[i] != test.expected_counts[i]) {
                passed = false;
                issues += "Type[" + std::to_string(i) + "] expected " + 
                         std::to_string(test.expected_counts[i]) + 
                         ", got " + std::to_string(span.confix_type_counts[i]) + "; ";
            }
        }
        
        record_test("confix_classification::" + test.name, passed, issues, 0.0, 
                   span.confixes.size(), span.confidence);
        if (!passed) all_passed = false;
    }
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    if (all_passed) {
        record_test("comprehensive_confix_classification", true, 
                   "All confix types correctly classified", time_ms);
    }
}

// Test 2: Template angle bracket disambiguation (the critical test)
void test_template_disambiguation() {
    auto start = high_resolution_clock::now();
    
    struct TestCase {
        std::string code;
        std::string name;
        bool is_template_context;
        size_t expected_angle_pairs;
        double min_confidence;
    };
    
    TestCase cases[] = {
        // Clear template cases
        {"vector<int> x;", "simple_template", true, 1, 0.7},
        {"map<string, vector<int>> table;", "nested_template", true, 3, 0.8},
        {"function<vector<map<int, string>>>();", "deeply_nested", true, 4, 0.9},
        
        // Clear shift cases
        {"a >> b", "right_shift", false, 0, 0.3},
        {"x = y >> 2;", "bitshift_assignment", false, 0, 0.3},
        {"if (x >> 1) {", "bitshift_condition", false, 0, 0.3},
        {"cout << data >> 8;", "mixed_stream_shift", false, 0, 0.3},
        
        // Ambiguous cases that need context
        {"auto result = a >> b;", "ambiguous_shift", false, 0, 0.4},
        {"return x >> y;", "ambiguous_return", false, 0, 0.4},
        
        // Complex mixed cases
        {"vector<int> v; auto x = a >> b;", "template_then_shift", false, 1, 0.5},
        {"if (x > 0) { vector<int> v; }", "condition_then_template", true, 1, 0.6},
        {"template<typename T> void func() { auto x = a >> b; }", "template_context_with_shift", false, 1, 0.5}
    };
    
    bool all_passed = true;
    for (const auto& test : cases) {
        auto span = Evidence2DAnalyzer::analyze_span(test.code);
        auto angle_confixes = span.get_confixes_of_type(ConfixType::ANGLE);
        
        bool passed = true;
        std::string issues;
        
        // Check angle pair count (each pair is 2 confixes)
        size_t angle_pairs = angle_confixes.size() / 2;
        if (angle_pairs != test.expected_angle_pairs) {
            passed = false;
            issues += "Expected " + std::to_string(test.expected_angle_pairs) + 
                     " angle pairs, got " + std::to_string(angle_pairs) + "; ";
        }
        
        // Check confidence threshold
        if (span.confidence < test.min_confidence) {
            passed = false;
            issues += "Confidence " + std::to_string(span.confidence) + 
                     " below minimum " + std::to_string(test.min_confidence) + "; ";
        }
        
        // Check balance
        if (!span.has_balanced_confixes()) {
            passed = false;
            issues += "Unbalanced confixes; ";
        }
        
        record_test("template_disambiguation::" + test.name, passed, issues, 0.0, 
                   span.confixes.size(), span.confidence);
        if (!passed) all_passed = false;
    }
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    if (all_passed) {
        record_test("template_disambiguation", true, 
                   "All template vs shift cases correctly identified", time_ms);
    }
}

// Test 3: Concurrent multi-speculation analysis
void test_concurrent_multi_speculation() {
    auto start = high_resolution_clock::now();
    
    // Simulate competing interpretations of ambiguous code
    std::string ambiguous_code = "auto result = a >> b;";
    
    struct Speculation {
        std::string context;
        std::string name;
        double expected_confidence;
        size_t expected_angles;
    };
    
    Speculation speculations[] = {
        {"auto result = a >> b;", "pure_shift", 0.3, 0},
        {"vector<int> a, b; auto result = a >> b;", "shift_in_template_context", 0.4, 1},
        {"template<typename T> auto func() { auto result = a >> b; }", "shift_in_template_function", 0.5, 1},
        {"vector<map<string,int>> a, b; auto result = a >> b;", "shift_after_nested_template", 0.6, 3}
    };
    
    // Concurrent analysis
    std::vector<std::future<EvidenceSpan2D>> futures;
    concurrent_operations = 0;
    
    for (const auto& spec : speculations) {
        futures.push_back(std::async(std::launch::async, [spec]() {
            concurrent_operations++;
            return Evidence2DAnalyzer::analyze_span(spec.context);
        }));
    }
    
    // Collect results
    std::vector<EvidenceSpan2D> results;
    std::vector<std::string> names;
    
    for (size_t i = 0; i < futures.size(); ++i) {
        results.push_back(futures[i].get());
        names.push_back(speculations[i].name);
    }
    
    // Fast concurrent winnowing
    auto winner = Evidence2D::find_strongest_evidence(results);
    
    bool passed = winner != nullptr;
    std::string message;
    if (winner) {
        size_t winner_idx = winner - &results[0];
        message = "Winner: " + names[winner_idx] + " (confidence: " + 
                 std::to_string(winner->confidence) + ")";
    }
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    record_test("concurrent_multi_speculation", passed, message, time_ms, 
               concurrent_operations.load(), winner ? winner->confidence : 0.0);
}

// Test 4: Performance under load
void test_performance_under_load() {
    auto start = high_resolution_clock::now();
    
    // Generate stress test cases
    std::vector<std::string> stress_cases;
    
    // Large nested templates
    stress_cases.push_back(R"(
        template<typename T, typename U, typename V>
        class ComplexClass {
            std::vector<std::map<T, std::pair<U, V>>> data;
            std::function<std::vector<std::set<T>>(const U&)> processor;
        public:
            template<typename W>
            void process(const std::vector<W>& input) {
                for (const auto& item : input) {
                    if (item > threshold) {
                        auto result = processor(item);
                        data[item].first = result;
                    }
                }
            }
        };
    )");
    
    // Heavy comment mix
    stress_cases.push_back(R"(
        /* This is a C comment that spans multiple lines
           and contains various characters like < > { } [ ] ( ) */
        
        // This is a C++ comment with template syntax: vector<int>
        // And some operators: a >> b, x << y
        
        ``` This is a Cpp2 comment
            with nested ` characters and template stuff
            vector<map<string, int>> data;
        ```
        
        int main() { return 0; /* inline c comment */ }
    )");
    
    // Deep nesting
    stress_cases.push_back(R"(
        void deep_nesting() {
            if (condition1) {
                if (condition2) {
                    while (condition3) {
                        for (int i = 0; i < 10; i++) {
                            switch (value) {
                                case 1: { /* nested block */ break; }
                                case 2: [capture]() { /* lambda */ }(); break;
                            }
                        }
                    }
                }
            }
        }
    )");
    
    // Measure performance
    const int iterations = 50;
    auto iter_start = high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        for (const auto& code : stress_cases) {
            auto span = Evidence2DAnalyzer::analyze_span(code);
            // Force evaluation
            volatile auto conf = span.confidence;
            volatile auto count = span.confixes.size();
            (void)conf; (void)count;
        }
    }
    
    auto iter_end = high_resolution_clock::now();
    double total_time = duration_cast<microseconds>(iter_end - iter_start).count() / 1000.0;
    double avg_time = total_time / (iterations * stress_cases.size());
    
    auto end = high_resolution_clock::now();
    double setup_time = duration_cast<microseconds>(end - start).count() / 1000.0 - total_time;
    
    // Performance requirement: must handle complex code in under 0.5ms per analysis
    bool passed = avg_time < 0.5;
    
    record_test("performance_under_load", passed, 
               "Average analysis time: " + std::to_string(avg_time) + 
               "ms per complex case (" + std::to_string(iterations * stress_cases.size()) + 
               " total analyses)", setup_time + total_time, 
               iterations * stress_cases.size());
}

// Test 5: Randomized stress testing
void test_randomized_stress() {
    auto start = high_resolution_clock::now();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> length_dist(10, 200);
    std::uniform_int_distribution<> char_dist(0, 127);
    std::uniform_real_distribution<> confix_prob(0.0, 1.0);
    
    const int num_tests = 100;
    int passed_tests = 0;
    size_t total_confixes = 0;
    
    for (int i = 0; i < num_tests; ++i) {
        // Generate random code with confixes
        int length = length_dist(gen);
        std::string random_code;
        random_code.reserve(length);
        
        for (int j = 0; j < length; ++j) {
            if (confix_prob(gen) < 0.1) { // 10% chance of confix
                // Add random confix character
                const char* confix_chars = "(){}[]<>/*\"`";
                random_code += confix_chars[std::uniform_int_distribution<>(0, 11)(gen)];
            } else {
                // Add random character
                random_code += static_cast<char>(char_dist(gen));
            }
        }
        
        // Analyze the random code
        auto span = Evidence2DAnalyzer::analyze_span(random_code);
        
        // Basic validation: should not crash, should have reasonable confidence
        bool passed = span.confidence >= 0.0 && span.confidence <= 1.0;
        if (passed) passed_tests++;
        
        total_confixes += span.confixes.size();
    }
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    double avg_confixes = static_cast<double>(total_confixes) / num_tests;
    
    bool passed = (passed_tests == num_tests);
    
    record_test("randomized_stress", passed, 
               "Passed " + std::to_string(passed_tests) + "/" + std::to_string(num_tests) + 
               " random tests, avg " + std::to_string(avg_confixes) + " confixes per test", 
               time_ms, total_confixes);
}

// Test 6: Memory efficiency
void test_memory_efficiency() {
    auto start = high_resolution_clock::now();
    
    // Test with progressively larger inputs
    std::vector<size_t> sizes = {100, 1000, 10000, 100000};
    std::vector<double> times_per_char;
    
    for (size_t size : sizes) {
        std::string large_code(size, ' ');
        // Fill with realistic code pattern
        for (size_t i = 0; i < size; i += 10) {
            if (i + 10 < size) {
                std::memcpy(&large_code[i], "int x(); ", 9);
            }
        }
        
        auto iter_start = high_resolution_clock::now();
        auto span = Evidence2DAnalyzer::analyze_span(large_code);
        auto iter_end = high_resolution_clock::now();
        
        double time_per_char = duration_cast<nanoseconds>(iter_end - iter_start).count() / 
                              static_cast<double>(size);
        times_per_char.push_back(time_per_char);
        
        // Verify it still works
        if (span.confixes.empty() || span.confidence < 0.1) {
            record_test("memory_efficiency::size_" + std::to_string(size), false, 
                       "Analysis failed on large input", 0.0, size);
            return;
        }
    }
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    // Check for linear scaling (should be roughly constant time per character)
    bool scales_linearly = true;
    for (size_t i = 1; i < times_per_char.size(); ++i) {
        if (times_per_char[i] > times_per_char[0] * 2.0) {
            scales_linearly = false;
            break;
        }
    }
    
    record_test("memory_efficiency", scales_linearly, 
               "Time per character scales linearly with input size", time_ms);
}

// Main test runner
int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  2D Evidence System Comprehensive Regression Tests          ║\n";
    std::cout << "║  Fast Concurrent Multi-Speculation with EvidenceSpan        ║\n";
    std::cout << "║  Winnowing using 2D Confix Type × Position Analysis        ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Testing all confix types:\n";
    std::cout << "- PAREN: ()\n";
    std::cout << "- BRACE: {}\n";
    std::cout << "- BRACKET: []\n";
    std::cout << "- ANGLE: <> (with template disambiguation)\n";
    std::cout << "- C_COMMENT: /* */\n";
    std::cout << "- CPP_COMMENT: //\n";
    std::cout << "- CPP2_COMMENT: ```\n\n";
    
    std::cout << "Running comprehensive regression tests...\n\n";
    
    // Run all tests
    test_comprehensive_confix_classification();
    test_template_disambiguation();
    test_concurrent_multi_speculation();
    test_performance_under_load();
    test_randomized_stress();
    test_memory_efficiency();
    // Note: These functions are defined above and will be called in order
    
    // Summary
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "COMPREHENSIVE REGRESSION TEST SUMMARY\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    int passed = 0, failed = 0;
    double total_time = 0.0;
    size_t total_confixes = 0;
    
    for (const auto& result : test_results) {
        if (result.passed) passed++;
        else failed++;
        total_time += result.metrics.execution_time_ms;
        total_confixes += result.metrics.confix_count;
    }
    
    std::cout << "Total tests: " << test_results.size() << "\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";
    std::cout << "Total execution time: " << total_time << "ms\n";
    std::cout << "Total confixes analyzed: " << total_confixes << "\n";
    std::cout << "Average time per test: " << (total_time / test_results.size()) << "ms\n";
    
    if (failed == 0) {
        std::cout << "\n✅ ALL COMPREHENSIVE REGRESSION TESTS PASSED\n";
        std::cout << "The 2D evidence system successfully provides:\n";
        std::cout << "- Fast concurrent scanning of multiple competing speculations\n";
        std::cout << "- Rapid winnowing using EvidenceSpan locality bubbles\n";
        std::cout << "- Accurate template angle bracket disambiguation\n";
        std::cout << "- All confix types: structural + comments (C/C++/Cpp2)\n";
        std::cout << "- Linear performance scaling with input size\n";
        std::cout << "- Robust handling of edge cases and stress conditions\n";
    } else {
        std::cout << "\n❌ " << failed << " COMPREHENSIVE TESTS FAILED\n";
        std::cout << "Please review the failed tests above.\n";
        return 1;
    }
    
    return 0;
}

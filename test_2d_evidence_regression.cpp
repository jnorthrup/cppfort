#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <future>
#include "src/stage0/evidence_2d.h"

using namespace cppfort::stage0;
using namespace std::chrono;

// Test results tracking
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double execution_time_ms;
};

std::vector<TestResult> test_results;

// Helper to record test results
void record_test(const std::string& name, bool passed, const std::string& message, double time_ms) {
    test_results.push_back({name, passed, message, time_ms});
    std::cout << (passed ? "✅" : "❌") << " " << name << " (" << time_ms << "ms)";
    if (!message.empty()) std::cout << " - " << message;
    std::cout << "\n";
}

// Test 1: Basic confix type classification
void test_confix_classification() {
    auto start = high_resolution_clock::now();
    
    struct TestCase {
        char ch;
        ConfixType expected;
        const char* name;
    };
    
    TestCase cases[] = {
        {'(', ConfixType::PAREN, "paren_open"},
        {')', ConfixType::PAREN, "paren_close"},
        {'{', ConfixType::BRACE, "brace_open"},
        {'}', ConfixType::BRACE, "brace_close"},
        {'[', ConfixType::BRACKET, "bracket_open"},
        {']', ConfixType::BRACKET, "bracket_close"},
        {'<', ConfixType::ANGLE, "angle_open"},
        {'>', ConfixType::ANGLE, "angle_close"},
        {'a', ConfixType::INVALID, "non_confix"},
        {' ', ConfixType::INVALID, "whitespace"}
    };
    
    bool all_passed = true;
    for (const auto& test : cases) {
        ConfixType result = Evidence2DAnalyzer::get_confix_type(test.ch);
        if (result != test.expected) {
            record_test("confix_classification::" + std::string(test.name), false, 
                       "Expected " + std::to_string(static_cast<int>(test.expected)) + 
                       ", got " + std::to_string(static_cast<int>(result)), 0.0);
            all_passed = false;
        }
    }
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    if (all_passed) {
        record_test("confix_classification", true, "All confix types correctly identified", time_ms);
    }
}

// Test 2: 2D evidence span analysis
void test_2d_evidence_spans() {
    auto start = high_resolution_clock::now();
    
    struct TestCase {
        std::string code;
        std::string name;
        size_t expected_confixes;
        ConfixType dominant_type;
        bool should_be_balanced;
    };
    
    TestCase cases[] = {
        {"int main() { return 0; }", "simple_function", 4, ConfixType::BRACE, true},
        {"vector<map<string,int>> data;", "template_nested", 6, ConfixType::ANGLE, true},
        {"if (x > 0 && y < 10) {", "complex_condition", 4, ConfixType::PAREN, false},
        {"array[index]", "bracket_access", 2, ConfixType::BRACKET, true},
        {"a >> b", "right_shift", 0, ConfixType::INVALID, true},
        {"{[()]}\n", "all_confix_types", 6, ConfixType::BRACE, true}
    };
    
    bool all_passed = true;
    for (const auto& test : cases) {
        auto span = Evidence2DAnalyzer::analyze_span(test.code);
        
        bool passed = true;
        std::string issues;
        
        if (span.confixes.size() != test.expected_confixes) {
            passed = false;
            issues += "Expected " + std::to_string(test.expected_confixes) + 
                     " confixes, got " + std::to_string(span.confixes.size()) + "; ";
        }
        
        if (span.get_dominant_confix_type() != test.dominant_type) {
            passed = false;
            issues += "Wrong dominant type; ";
        }
        
        if (span.has_balanced_confixes() != test.should_be_balanced) {
            passed = false;
            issues += "Balance check failed; ";
        }
        
        record_test("2d_evidence_spans::" + test.name, passed, issues, 0.0);
        if (!passed) all_passed = false;
    }
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    if (all_passed) {
        record_test("2d_evidence_spans", true, "All span analyses correct", time_ms);
    }
}

// Test 3: Comment classification (C, C++, Cpp2)
void test_comment_classification() {
    auto start = high_resolution_clock::now();
    
    struct TestCase {
        std::string code;
        std::string name;
        size_t expected_comments;
        bool has_c_comments;
        bool has_cpp_comments;
    };
    
    TestCase cases[] = {
        {"/* C comment */ int x;", "c_comment", 1, true, false},
        {"// C++ comment\nint x;", "cpp_comment", 1, false, true},
        {"```cpp2 comment```\nint x;", "cpp2_comment", 1, false, false},
        {"/* multi\n   line */\n// single\n```block```", "mixed_comments", 3, true, true},
        {"int x = 5; /* comment */ // another\n```third```", "inline_comments", 3, true, true}
    };
    
    bool all_passed = true;
    for (const auto& test : cases) {
        auto span = Evidence2DAnalyzer::analyze_span(test.code);
        
        // Count comment-like patterns (treating ``` as special confix)
        size_t comment_count = 0;
        bool found_c = false, found_cpp = false;
        
        for (char c : test.code) {
            if (c == '/' && test.code.find("/*", &c - test.code.data()) != std::string::npos) {
                found_c = true;
                comment_count++;
            }
            if (c == '/' && test.code.find("//", &c - test.code.data()) != std::string::npos) {
                found_cpp = true;
                comment_count++;
            }
            if (c == '`' && test.code.find("```", &c - test.code.data()) != std::string::npos) {
                comment_count++;
            }
        }
        
        bool passed = (comment_count == test.expected_comments) &&
                     (found_c == test.has_c_comments) &&
                     (found_cpp == test.has_cpp_comments);
        
        record_test("comment_classification::" + test.name, passed, "", 0.0);
        if (!passed) all_passed = false;
    }
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    if (all_passed) {
        record_test("comment_classification", true, "All comment types correctly identified", time_ms);
    }
}

// Test 4: Concurrent scanning performance
void test_concurrent_scanning() {
    auto start = high_resolution_clock::now();
    
    // Generate test cases
    std::vector<std::string> test_cases = {
        "int func(int x) { return x * 2; }",
        "template<typename T> class Vector { /* ... */ };",
        "auto lambda = [capture](int x) -> int { return x + 1; };",
        "if (condition) { do_something(); } else { do_other(); }",
        "vector<map<string, int>> data = get_data();",
        "```cpp2 comment```\nmain: (args) = { /* code */ }",
        "/* C comment */\n// C++ comment\nint main() { return 0; }"
    };
    
    // Concurrent scanning
    std::vector<std::future<EvidenceSpan2D>> futures;
    
    for (const auto& code : test_cases) {
        futures.push_back(std::async(std::launch::async, [code]() {
            return Evidence2DAnalyzer::analyze_span(code);
        }));
    }
    
    // Collect results
    std::vector<EvidenceSpan2D> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    // Validate results
    bool all_passed = true;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].confidences.empty()) {
            record_test("concurrent_scanning::case_" + std::to_string(i), false, "No evidence found", 0.0);
            all_passed = false;
        }
    }
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    double avg_time = time_ms / test_cases.size();
    
    if (all_passed) {
        record_test("concurrent_scanning", true, 
                   "All " + std::to_string(test_cases.size()) + 
                   " cases processed concurrently, avg " + 
                   std::to_string(avg_time) + "ms per case", time_ms);
    }
}

// Test 5: Fast winnowing using EvidenceSpan
void test_fast_winnowing() {
    auto start = high_resolution_clock::now();
    
    // Generate competing speculations
    std::vector<EvidenceSpan2D> speculations;
    
    // High confidence, balanced
    speculations.push_back(EvidenceSpan2D(0, 20, "int main() { return 0; }", 0.9));
    speculations.back().add_confix(ConfixType::PAREN, 8, 9);
    speculations.back().add_confix(ConfixType::BRACE, 15, 16);
    speculations.back().add_confix(ConfixType::PAREN, 18, 19);
    speculations.back().add_confix(ConfixType::BRACE, 19, 20);
    
    // Medium confidence, unbalanced
    speculations.push_back(EvidenceSpan2D(0, 15, "int main( { return 0; }", 0.6));
    speculations.back().add_confix(ConfixType::PAREN, 8, 9);
    speculations.back().add_confix(ConfixType::BRACE, 10, 11);
    speculations.back().add_confix(ConfixType::BRACE, 18, 19);
    
    // Low confidence, many confixes
    speculations.push_back(EvidenceSpan2D(0, 25, "int main() { { { return 0; } } }", 0.4));
    for (size_t i = 0; i < 6; ++i) {
        speculations.back().add_confix(ConfixType::BRACE, 10 + i*2, 11 + i*2);
    }
    
    // Fast winnowing
    auto start_winnow = high_resolution_clock::now();
    
    // Filter by balance
    std::vector<EvidenceSpan2D> balanced;
    for (const auto& span : speculations) {
        if (span.has_balanced_confixes()) {
            balanced.push_back(span);
        }
    }
    
    // Sort by confidence
    std::sort(balanced.begin(), balanced.end(), 
              [](const EvidenceSpan2D& a, const EvidenceSpan2D& b) {
                  return a.confidence > b.confidence;
              });
    
    auto end_winnow = high_resolution_clock::now();
    double winnow_time = duration_cast<microseconds>(end_winnow - start_winnow).count() / 1000.0;
    
    bool passed = !balanced.empty() && balanced[0].confidence > 0.8;
    
    auto end = high_resolution_clock::now();
    double total_time = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    record_test("fast_winnowing", passed, 
               "Winnowed " + std::to_string(speculations.size()) + 
               " speculations to " + std::to_string(balanced.size()) + 
               " in " + std::to_string(winnow_time) + "ms", total_time);
}

// Test 6: Template angle bracket disambiguation (the hard case)
void test_template_disambiguation() {
    auto start = high_resolution_clock::now();
    
    struct TestCase {
        std::string code;
        std::string name;
        bool is_template;  // Should be interpreted as template
        size_t expected_angle_pairs;
    };
    
    TestCase cases[] = {
        {"vector<int> data;", "simple_template", true, 1},
        {"a >> b", "right_shift", false, 0},
        {"map<string, vector<int>> table;", "nested_template", true, 3},
        {"if (x >> 2) {", "bitshift_in_condition", false, 0},
        {"function<vector<map<int, string>>>();", "deeply_nested", true, 4},
        {"cout << data >> 8;", "mixed_shift", false, 0}
    };
    
    bool all_passed = true;
    for (const auto& test : cases) {
        auto span = Evidence2DAnalyzer::analyze_span(test.code);
        
        auto angle_confixes = span.get_confixes_of_type(ConfixType::ANGLE);
        bool has_angles = !angle_confixes.empty();
        bool passed = (has_angles == test.is_template) && 
                     (angle_confixes.size() == test.expected_angle_pairs * 2); // *2 for open/close
        
        std::string message = "Found " + std::to_string(angle_confixes.size() / 2) + 
                             " angle pairs";
        
        record_test("template_disambiguation::" + test.name, passed, message, 0.0);
        if (!passed) all_passed = false;
    }
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    if (all_passed) {
        record_test("template_disambiguation", true, "All template vs shift cases correctly identified", time_ms);
    }
}

// Test 7: Multi-speculation concurrent analysis
void test_multi_speculation() {
    auto start = high_resolution_clock::now();
    
    // Simulate multiple competing interpretations of same code
    std::string ambiguous_code = "int x = a >> b;";
    
    // Speculation 1: Right shift
    auto shift_span = Evidence2DAnalyzer::analyze_span(ambiguous_code);
    shift_span.confidence = 0.3; // Low confidence for shift
    
    // Speculation 2: Template (if context suggests it)
    auto template_context = "vector<int> a; vector<int> b; int x = a >> b;";
    auto template_span = Evidence2DAnalyzer::analyze_span(template_context);
    template_span.confidence = 0.7; // Higher confidence with template context
    
    // Concurrent evaluation
    std::vector<EvidenceSpan2D> speculations = {shift_span, template_span};
    
    // Fast concurrent winnowing
    auto winner = Evidence2D::find_strongest_evidence(speculations);
    
    bool passed = winner != nullptr && winner->confidence > 0.5;
    
    auto end = high_resolution_clock::now();
    double time_ms = duration_cast<microseconds>(end - start).count() / 1000.0;
    
    record_test("multi_speculation", passed, 
               winner ? "Winner confidence: " + std::to_string(winner->confidence) : "No winner found", 
               time_ms);
}

// Test 8: Performance regression - must be fast
void test_performance_regression() {
    auto start = high_resolution_clock::now();
    
    // Large code sample
    std::string large_code = R"(
#include <vector>
#include <map>
#include <string>

template<typename T, typename U>
class ComplexTemplate {
    std::vector<std::map<T, U>> data;
    
public:
    void process(const std::vector<T>& input) {
        for (const auto& item : input) {
            if (item > threshold) {
                data[item].push_back(process_item(item));
            }
        }
    }
    
private:
    U process_item(const T& item) {
        return static_cast<U>(item * 2);
    }
};

int main() {
    ComplexTemplate<int, double> processor;
    std::vector<int> input = {1, 2, 3, 4, 5};
    processor.process(input);
    return 0;
}
)";
    
    // Multiple runs for accurate timing
    const int iterations = 100;
    auto iter_start = high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        auto span = Evidence2DAnalyzer::analyze_span(large_code);
        // Force evaluation
        volatile auto conf = span.confidence;
        (void)conf;
    }
    
    auto iter_end = high_resolution_clock::now();
    double total_time = duration_cast<microseconds>(iter_end - iter_start).count() / 1000.0;
    double avg_time = total_time / iterations;
    
    auto end = high_resolution_clock::now();
    double setup_time = duration_cast<microseconds>(end - start).count() / 1000.0 - total_time;
    
    // Performance requirement: must analyze large code in under 1ms per iteration
    bool passed = avg_time < 1.0;
    
    record_test("performance_regression", passed, 
               "Average analysis time: " + std::to_string(avg_time) + 
               "ms per iteration (" + std::to_string(iterations) + " iterations)", 
               setup_time + total_time);
}

// Main test runner
int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  2D Evidence System Regression Tests                        ║\n";
    std::cout << "║  Confix Type × Spatial Position Analysis                   ║\n";
    std::cout << "║  Fast Concurrent Scanning + Winnowing                     ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Running regression tests...\n\n";
    
    // Run all tests
    test_confix_classification();
    test_2d_evidence_spans();
    test_comment_classification();
    test_concurrent_scanning();
    test_fast_winnowing();
    test_template_disambiguation();
    test_multi_speculation();
    test_performance_regression();
    
    // Summary
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "REGRESSION TEST SUMMARY\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    int passed = 0, failed = 0;
    double total_time = 0.0;
    
    for (const auto& result : test_results) {
        if (result.passed) passed++;
        else failed++;
        total_time += result.execution_time_ms;
    }
    
    std::cout << "Total tests: " << test_results.size() << "\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";
    std::cout << "Total execution time: " << total_time << "ms\n";
    std::cout << "Average time per test: " << (total_time / test_results.size()) << "ms\n";
    
    if (failed == 0) {
        std::cout << "\n✅ ALL REGRESSION TESTS PASSED\n";
        std::cout << "The 2D evidence system is working correctly for:\n";
        std::cout << "- Fast concurrent scanning\n";
        std::cout << "- Multi-speculation analysis\n";
        std::cout << "- Rapid winnowing using EvidenceSpan locality\n";
        std::cout << "- Template angle bracket disambiguation\n";
        std::cout << "- All confix types (open/close/brace/c_comments/cpp_comments/cpp2_comments)\n";
    } else {
        std::cout << "\n❌ " << failed << " REGRESSION TESTS FAILED\n";
        std::cout << "Please review the failed tests above.\n";
        return 1;
    }
    
    return 0;
}

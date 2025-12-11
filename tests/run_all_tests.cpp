#include <iostream>
#include <string>

// Forward declarations for test functions
int test_main();
int cppfront_regression_tests_main();

int main() {
    std::cout << "========================================\n";
    std::cout << "Running All Cpp2 Transpiler Tests\n";
    std::cout << "========================================\n\n";

    int result = 0;

    // Run original test suite
    std::cout << "1. Running Original Test Suite:\n";
    std::cout << "--------------------------------\n";
    result += test_main();
    std::cout << "\n";

    // Run cppfront regression tests
    std::cout << "2. Running Cppfront Regression Tests:\n";
    std::cout << "------------------------------------\n";
    result += cppfront_regression_tests_main();
    std::cout << "\n";

    // Summary
    std::cout << "========================================\n";
    if (result == 0) {
        std::cout << "✅ ALL TESTS PASSED\n";
    } else {
        std::cout << "❌ SOME TESTS FAILED (exit code: " << result << ")\n";
    }
    std::cout << "========================================\n";

    return result;
}
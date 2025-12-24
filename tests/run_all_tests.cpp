#include <iostream>
#include <string>

// Run external test executables so we don't have duplicate symbol conflicts
static int run_and_check(const std::string& cmd) {
    std::cout << "Running: " << cmd << "\n";
    int rc = std::system(cmd.c_str());
    if (rc == -1) return 1;
    return WEXITSTATUS(rc);
}

int main() {
    std::cout << "========================================\n";
    std::cout << "Running All Cpp2 Transpiler Tests\n";
    std::cout << "========================================\n\n";

    int result = 0;

    std::cout << "1. Running Original Test Suite (cpp2_tests):\n";
    std::cout << "-----------------------------------------\n";
    result += run_and_check("./tests/cpp2_tests");
    std::cout << "\n";

    std::cout << "2. Running Cppfront Regression Tests (cppfront_regression_tests):\n";
    std::cout << "-----------------------------------------------------------\n";
    result += run_and_check("./tests/cppfront_regression_tests");
    std::cout << "\n";

    std::cout << "========================================\n";
    if (result == 0) {
        std::cout << "✅ ALL TESTS PASSED\n";
    } else {
        std::cout << "❌ SOME TESTS FAILED (exit code: " << result << ")\n";
    }
    std::cout << "========================================\n";

    return result;
}
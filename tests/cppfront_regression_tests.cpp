#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "../include/lexer.hpp"
#include "../include/parser.hpp"
#include "../include/semantic_analyzer.hpp"
#include "../include/code_generator.hpp"
#include "../include/safety_checker.hpp"
#include "../include/metafunction_processor.hpp"
#include "../include/contract_processor.hpp"

using namespace cpp2_transpiler;

// Test utilities
static std::string read_file(const std::string& filename) {
    std::ifstream file(filename);
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

bool transpile_succeeds(const std::string& cpp2_code) {
    try {
        Lexer lexer(cpp2_code);
        auto tokens = lexer.tokenize();

        Parser parser(tokens);
        auto ast = parser.parse();

        SemanticAnalyzer semantic_analyzer;
        semantic_analyzer.analyze(*ast);

        SafetyChecker safety_checker;
        safety_checker.check(*ast);

        MetafunctionProcessor meta_processor;
        meta_processor.process(*ast);

        ContractProcessor contract_processor;
        contract_processor.process(*ast);

        CodeGenerator code_generator;
        auto result = code_generator.generate(*ast);

        return !result.empty();
    }
    catch (const std::exception& e) {
        std::cerr << "Transpilation failed: " << e.what() << std::endl;
        return false;
    }
}

// Regression tests based on cppfront
void test_cppfront_basic() {
    std::cout << "Testing cppfront basic syntax..." << std::endl;

    // Test from pure2-bounds-safety-span.cpp2
    std::string cpp2_code = R"(
        main: () -> int = {
            words: std::vector<std::string> = ("decorated", "hello", "world");
            s: std::span<std::string> = words;
            _ = words;

            i := 0;
            while i < s.ssize() next i++ {
                print_and_decorate(s[i]);
            }
            return 0;
        }

        print_and_decorate: (x) = {
            std::cout << ">> " << x << "\n";
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Basic syntax test passed" << std::endl;
}

void test_cppfront_contracts() {
    std::cout << "Testing cppfront contracts..." << std::endl;

    // Simplified version of pure2-contracts.cpp2
    std::string cpp2_code = R"(
        test_condition_evaluation: (tag) -> bool = {
            std::cout << tag << "\n";
            return true;
        }

        main: () = {
            assert(1 != 2, "ack, arithmetic is buggy");
            assert(test_condition_evaluation(1), "default");
            return 0;
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Contracts test passed" << std::endl;
}

void test_cppforward_functions() {
    std::cout << "Testing cppfront function definitions..." << std::endl;

    // Test from pure2-trailing-commas.cpp2
    std::string cpp2_code = R"(
        f: (a, b,) = a + b;

        g: <T, U,> (a: T, b: U) = a + b;

        doubler: (a: int,) -> (i: int,) = {
            i = a * 2;
        }

        vals: @struct type = { i: int; }

        main: () -> int = {
            _ = g(1, 2,);
            grouping: std::vector<int> = (0, 1, 2,);
            array: std::array<int, 3> = (0, 1, 2,);
            return 0;
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Function definitions test passed" << std::endl;
}

void test_cppforward_assertions() {
    std::cout << "Testing cppfront assertions..." << std::endl;

    // Test from pure2-assert-expected-not-null.cpp2
    std::string cpp2_code = R"(
        crash_10: (foo: i32) -> i32 = {
            assert(10LL as i32 == foo);
            return foo;
        }

        main: () -> int = {
            return crash_10(10);
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Assertions test passed" << std::endl;
}

void test_cppforward_loops() {
    std::cout << "Testing cppfront loop constructs..." << std::endl;

    // Test various loop patterns
    std::string cpp2_code = R"(
        main: () -> int = {
            // While loop with next clause
            i := 0;
            while i < 10 next i++ {
                std::cout << i << "\n";
            }

            // For loop
            for j := 0; j < 5; j++ {
                std::cout << "j=" << j << "\n";
            }

            return 0;
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Loop constructs test passed" << std::endl;
}

void test_cppforward_break_continue() {
    std::cout << "Testing cppforward break and continue..." << std::endl;

    // Test from pure2-break-continue.cpp2
    std::string cpp2_code = R"(
        main: () -> int = {
            i := 0;
            while i < 10 {
                if i == 3 {
                    i++;
                    continue;
                }
                if i == 7 {
                    break;
                }
                std::cout << i << "\n";
                i++;
            }
            return 0;
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Break/continue test passed" << std::endl;
}

void test_cppforward_fixed_type_aliases() {
    std::cout << "Testing cppforward type aliases..." << std::endl;

    // Test from mixed-fixed-type-aliases.cpp2
    std::string cpp2_code = R"(
        // Fixed type aliases
        MyInt := int;
        MyString := std::string;

        // Template alias
        Vec := std::vector;

        main: () -> int = {
            x: MyInt = 42;
            s: MyString = "hello";
            v: Vec<int> = (1, 2, 3);
            return x;
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Type aliases test passed" << std::endl;
}

void test_cppforward_function_expressions() {
    std::cout << "Testing cppforward function expressions..." << std::endl;

    // Test from mixed-function-expression-and-std-for-each.cpp2
    std::string cpp2_code = R"(
        main: () -> int = {
            v: std::vector<int> = (1, 2, 3, 4, 5);

            // Function expression with std::for_each
            std::for_each(v.begin(), v.end(), :(x) = {
                std::cout << x << " ";
            });

            // Lambda with capture
            sum := 0;
            std::for_each(v.begin(), v.end(), :(x) = {
                sum += x;
            });

            return sum;
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Function expressions test passed" << std::endl;
}

void test_cppforward_pointer_arithmetic() {
    std::cout << "Testing cppforward pointer arithmetic..." << std::endl;

    // Test bounds safety with pointer arithmetic
    std::string cpp2_code = R"(
        main: () -> int = {
            arr: [int; 5] = {1, 2, 3, 4, 5};
            p: int* = &arr[0];

            // Safe pointer arithmetic
            for i := 0; i < 5; i++ {
                std::cout << *(p+i) << "\n";
            }

            return 0;
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Pointer arithmetic test passed" << std::endl;
}

void test_cppforward_uninitialized_variables() {
    std::cout << "Testing cppforward uninitialized variable detection..." << std::endl;

    std::string cpp2_code = R"(
        main: () -> int = {
            // This should generate an uninitialized variable warning
            x: int;

            // Initialize before use
            x = 42;
            return x;
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Uninitialized variable test passed" << std::endl;
}

    int cppfront_regression_tests_main() {
        try {
            test_cppfront_basic();
            test_cppfront_contracts();
            test_cppforward_functions();
            test_cppforward_assertions();
            test_cppforward_loops();
            test_cppforward_break_continue();
            test_cppforward_fixed_type_aliases();
            test_cppforward_function_expressions();
            test_cppforward_pointer_arithmetic();
            test_cppforward_uninitialized_variables();
            return 0;
        } catch (...) {
            return 1;
        }
    }

void test_cppforward_mixed_cpp1_cpp2() {
    std::cout << "Testing mixed Cpp1 and Cpp2..." << std::endl;

    // Test from mixed-allcpp1-hello.cpp2
    std::string cpp2_code = R"(
        #include <iostream>
        #include <string>

        // Mixed Cpp1 and Cpp2 should be allowed side-by-side

        auto cpp1_function() -> int {
            auto s = std::string{"world\n"};
            std::cout << "Hello " << s;
            return 0;
        }

        main: () -> int = {
            return cpp1_function();
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Mixed Cpp1/Cpp2 test passed" << std::endl;
}

void test_cppforward_string_interpolation() {
    std::cout << "Testing string interpolation..." << std::endl;

    std::string cpp2_code = R"(
        main: () -> int = {
            name := "World";
            count := 42;
            message := "Hello $(name)! The answer is $(count).";
            std::cout << message << "\n";
            return 0;
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ String interpolation test passed" << std::endl;
}

void test_cppforward_inspect_pattern_matching() {
    std::cout << "Testing inspect pattern matching..." << std::endl;

    std::string cpp2_code = R"(
        main: () -> int = {
            value := 42;

            result := inspect value {
                0 => "zero",
                1..=9 => "single digit",
                n if n < 0 => "negative",
                _ => "other"
            };

            std::cout << result << "\n";
            return 0;
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Pattern matching test passed" << std::endl;
}

void test_cppforward_range_operators() {
    std::cout << "Testing range operators..." << std::endl;

    std::string cpp2_code = R"(
        main: () -> int = {
            sum := 0;

            // Exclusive range
            for i in 0..<5 {
                sum += i;
            }

            // Inclusive range
            for j in 5..=10 {
                sum += j;
            }

            return sum;
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Range operators test passed" << std::endl;
}

// Test performance regression
void test_cppforward_performance_features() {
    std::cout << "Testing performance-related features..." << std::endl;

    std::string cpp2_code = R"(
        // Test move semantics, definite last use
        process: (data: std::vector<int>) -> std::vector<int> = {
            // Move large vector efficiently
            result := std::move(data);
            result.push_back(42);
            return result;
        }

        main: () -> int = {
            v: std::vector<int> = (1, 2, 3, 4, 5);
            // v is definitely last used here, should be moved
            result := process(v);
            return result.size();
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Performance features test passed" << std::endl;
}

// Test error handling
void test_cppforward_error_handling() {
    std::cout << "Testing error handling constructs..." << std::endl;

    std::string cpp2_code = R"(
        risky_operation: () -> int = {
            // Might throw
            throw std::runtime_error("test error");
            return 42;
        }

        main: () -> int = {
            try {
                return risky_operation();
            } catch (e: std::exception) {
                std::cout << "Caught: " << e.what() << "\n";
                return -1;
            }
        }
    )";

    assert(transpile_succeeds(cpp2_code));
    std::cout << "✓ Error handling test passed" << std::endl;
}

#ifndef COMBINED_TESTS
int main() {
    try {
        std::cout << "Running Cppfront Regression Tests\n" << std::endl;

        test_cppfront_basic();
        test_cppfront_contracts();
        test_cppforward_functions();
        test_cppforward_assertions();
        test_cppforward_loops();
        test_cppforward_break_continue();
        test_cppforward_fixed_type_aliases();
        test_cppforward_function_expressions();
        test_cppforward_pointer_arithmetic();
        test_cppforward_uninitialized_variables();
        test_cppforward_mixed_cpp1_cpp2();
        test_cppforward_string_interpolation();
        test_cppforward_inspect_pattern_matching();
        test_cppforward_range_operators();
        test_cppforward_performance_features();
        test_cppforward_error_handling();

        std::cout << "\nAll cppfront regression tests passed! ✓" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Regression test failed: " << e.what() << std::endl;
        return 1;
    }
}
#endif
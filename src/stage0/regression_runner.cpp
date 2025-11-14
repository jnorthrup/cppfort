#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <memory>

// Direct includes for transpilation
#include "orbit_pipeline.h"
#include "pattern_loader.h"
#include "cpp2_emitter.h"
#include "cpp2_key_resolver.h"
#include "evidence.h"

namespace fs = std::filesystem;

namespace {

std::string current_time_string() {
    const auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    char buffer[128];
    if (std::strftime(buffer, sizeof(buffer), "%c", std::localtime(&tt))) {
        return buffer;
    }
    return "unknown time";
}

// Direct transpilation without system calls
int transpile_direct(const fs::path& input_file,
                     const fs::path& output_file,
                     const fs::path& patterns_file,
                     std::ostream& log) {
    try {
        log << "    Direct transpile: " << input_file << " -> " << output_file << "\n";

        // Load patterns
        cppfort::stage0::PatternLoader pattern_loader;
        if (!pattern_loader.load_patterns(patterns_file.string())) {
            log << "    ERROR: Failed to load patterns from " << patterns_file << "\n";
            return 1;
        }

        // Read input file
        std::ifstream in(input_file, std::ios::binary);
        if (!in) {
            log << "    ERROR: Cannot read input file " << input_file << "\n";
            return 1;
        }
        std::string source((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());

        // Create pipeline and run transpilation
        cppfort::stage0::OrbitPipeline pipeline;
        pipeline.initialize_patterns(pattern_loader);

        // Process through pipeline
        auto result = pipeline.process(source);
        if (!result) {
            log << "    ERROR: Pipeline processing failed\n";
            return 1;
        }

        // Generate output
        cppfort::stage0::Cpp2Emitter emitter;
        std::string output = emitter.emit(*result);

        // Write output file
        std::ofstream out(output_file, std::ios::binary);
        if (!out) {
            log << "    ERROR: Cannot write output file " << output_file << "\n";
            return 1;
        }
        out << output;

        log << "    SUCCESS: Transpilation completed\n";
        return 0;

    } catch (const std::exception& e) {
        log << "    EXCEPTION: " << e.what() << "\n";
        return 1;
    } catch (...) {
        log << "    EXCEPTION: Unknown error\n";
        return 1;
    }
}

// Direct C++ compilation without system calls
int compile_direct(const fs::path& cpp_file,
                   const fs::path& exe_file,
                   std::ostream& log) {
    try {
        log << "    Direct compile: " << cpp_file << " -> " << exe_file << "\n";

        // For now, we'll still use system() for compilation as it's external
        // In a full integration, this would call the compiler API directly
        std::string cmd = "c++ -std=c++20 -I include " + cpp_file.string() +
                         " -o " + exe_file.string() + " 2>&1";

        std::array<char, 128> buffer;
        std::string result;

        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            log << "    ERROR: Failed to execute compiler\n";
            return 1;
        }

        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }

        int exit_code = pclose(pipe);
        if (exit_code != 0) {
            log << "    COMPILE ERROR:\n" << result << "\n";
            return exit_code;
        }

        log << "    SUCCESS: Compilation completed\n";
        return 0;

    } catch (const std::exception& e) {
        log << "    EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}

// Direct execution without system calls
int execute_direct(const fs::path& exe_file, std::ostream& log) {
    try {
        log << "    Direct execute: " << exe_file << "\n";

        // For direct execution, we need to use fork/exec
        // This is platform-specific; for now use popen to capture output
        std::string cmd = exe_file.string() + " 2>&1";

        std::array<char, 128> buffer;
        std::string output;

        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            log << "    ERROR: Failed to execute program\n";
            return 1;
        }

        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            output += buffer.data();
        }

        int exit_code = pclose(pipe);

        if (!output.empty()) {
            log << "    OUTPUT:\n" << output << "\n";
        }

        if (exit_code != 0) {
            log << "    EXIT CODE: " << exit_code << "\n";
        } else {
            log << "    SUCCESS: Execution completed\n";
        }

        return exit_code;

    } catch (const std::exception& e) {
        log << "    EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <test_dir> <patterns> <stage0_cli> [test_name_substring]\n";
        return 1;
    }

    const fs::path test_dir = argv[1];
    const fs::path patterns_path = argv[2];
    const fs::path stage0_cli = argv[3];
    const std::string filter = (argc >= 5) ? argv[4] : "";

    if (!fs::exists(test_dir)) {
        std::cerr << "Error: Test directory does not exist: " << test_dir << "\n";
        return 1;
    }

    std::cout << "Starting regression tests at " << current_time_string() << "\n";
    std::cout << "Test directory: " << test_dir << "\n";
    std::cout << "Patterns: " << patterns_path << "\n";
    std::cout << "Filter: " << (filter.empty() ? "(none)" : filter) << "\n\n";

    // Open log file
    const fs::path log_path = fs::current_path() / "regression_log.txt";
    std::ofstream log(log_path, std::ios::app);
    if (!log) {
        std::cerr << "Error: Cannot open log file: " << log_path << "\n";
        return 1;
    }

    log << "\n================================================================================\n";
    log << "Regression test run at " << current_time_string() << "\n";
    log << "================================================================================\n\n";

    std::set<fs::path> tests;
    for (const auto& entry : fs::recursive_directory_iterator(test_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".cpp2") {
            std::string filename = entry.path().filename().string();
            if (filter.empty() || filename.find(filter) != std::string::npos) {
                tests.insert(entry.path());
            }
        }
    }

    if (tests.empty()) {
        std::cout << "No tests found.\n";
        return 0;
    }

    int num_tests = 0;
    int num_passed = 0;
    int num_failed = 0;

    for (const auto& test : tests) {
        const std::string filename = test.filename().string();
        log << "Testing " << filename << "\n";
        num_tests++;

        // Transpile
        const fs::path output_cpp = test.parent_path() / (test.stem().string() + ".cpp");
        if (transpile_direct(test, output_cpp, patterns_path, log) != 0) {
            log << "  Transpile FAILED\n\n";
            num_failed++;
            continue;
        }

        // Check if we expect compilation to succeed
        bool expect_compile_fail = (filename.find("-error") != std::string::npos ||
                                   filename.find("-fail") != std::string::npos);

        // Compile
        const fs::path output_exe = test.parent_path() / (test.stem().string() + ".exe");
        int compile_rc = compile_direct(output_cpp, output_exe, log);

        if (expect_compile_fail) {
            if (compile_rc != 0) {
                log << "  Compile failed as expected -> PASS\n\n";
                num_passed++;
            } else {
                log << "  Compile succeeded but failure expected -> FAIL\n\n";
                num_failed++;
            }
            // Clean up
            fs::remove(output_cpp);
            fs::remove(output_exe);
            continue;
        } else {
            if (compile_rc != 0) {
                log << "  Compile FAILED\n\n";
                num_failed++;
                fs::remove(output_cpp);
                continue;
            }
        }

        // Execute
        int exec_rc = execute_direct(output_exe, log);
        if (exec_rc == 0) {
            log << "  Execution PASSED\n\n";
            num_passed++;
        } else {
            log << "  Execution FAILED\n\n";
            num_failed++;
        }

        // Clean up
        fs::remove(output_cpp);
        fs::remove(output_exe);
    }

    log << "================================================================================\n";
    log << "Results: " << num_passed << "/" << num_tests << " passed, "
        << num_failed << " failed\n";
    log << "================================================================================\n";

    std::cout << "\n================================================================================\n";
    std::cout << "Results: " << num_passed << "/" << num_tests << " passed, "
              << num_failed << " failed\n";
    std::cout << "Log written to: " << log_path << "\n";
    std::cout << "================================================================================\n";

    return (num_failed > 0) ? 1 : 0;
}
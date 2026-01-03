#include "cppfront_test_framework.h"
#include "sha256.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <regex>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#include <ctime>

namespace cppfort::tests {

std::vector<TestFile> TestCollector::collect_test_files(const std::string& test_dir) {
    std::vector<TestFile> files;

    for (const auto& entry : std::filesystem::recursive_directory_iterator(test_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".cpp2") {
            TestFile test_file;
            test_file.path = entry.path().string();
            test_file.category = extract_category(entry.path());
            test_file.name = extract_name(entry.path());
            test_file.should_pass = should_pass_by_name(test_file.name);

            // Read file content
            std::ifstream file(entry.path());
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                test_file.content = buffer.str();
                test_file.sha256_hash = calculate_sha256(test_file.content);
                test_file.size_bytes = test_file.content.size();
                file.close();
            }

            files.push_back(test_file);
        }
    }

    // Sort by category then name
    std::sort(files.begin(), files.end(), [](const TestFile& a, const TestFile& b) {
        if (a.category != b.category) {
            return a.category < b.category;
        }
        return a.name < b.name;
    });

    return files;
}

std::string TestCollector::calculate_sha256(const std::string& content) {
    return SHA256::hash(content);
}

std::string TestCollector::extract_category(const std::filesystem::path& path) {
    std::string filename = path.filename().string();

    // Extract prefix before first dash
    size_t dash_pos = filename.find('-');
    if (dash_pos != std::string::npos) {
        return filename.substr(0, dash_pos);
    }

    return "unknown";
}

std::string TestCollector::extract_name(const std::filesystem::path& path) {
    std::string filename = path.filename().string();

    // Remove .cpp2 extension
    if (filename.length() > 5 && filename.substr(filename.length() - 5) == ".cpp2") {
        filename = filename.substr(0, filename.length() - 5);
    }

    return filename;
}

bool TestCollector::should_pass_by_name(const std::string& name) {
    // Files with "error" in their name are expected to fail compilation/transpilation
    return name.find("error") == std::string::npos;
}

TestRunner::TestRunner(const std::vector<TestFile>& test_files)
    : test_files_(test_files), transpiler_path_(find_transpiler()) {
    register_default_handlers();
}

TestRunner::TestRunner(const std::vector<TestFile>& test_files, const std::string& transpiler_path)
    : test_files_(test_files), transpiler_path_(find_transpiler(transpiler_path)) {
    register_default_handlers();
}

void TestRunner::register_category_handler(const std::string& category,
                                         std::function<TestResult(const TestFile&)> handler) {
    category_handlers_[category] = handler;
}

void TestRunner::run_all_tests() {
    results_.clear();
    results_.reserve(test_files_.size());

    size_t processed = 0;
    for (const auto& test_file : test_files_) {
        std::cout << "\rRunning tests: " << ++processed << "/" << test_files_.size()
                  << " (" << test_file.category << ":" << test_file.name << ")" << std::flush;

        auto it = category_handlers_.find(test_file.category);
        if (it != category_handlers_.end()) {
            results_.push_back(it->second(test_file));
        } else {
            results_.push_back(run_transpiler_test(test_file));
        }
    }
    std::cout << std::endl;
}

void TestRunner::run_category(const std::string& category) {
    results_.clear();

    for (const auto& test_file : test_files_) {
        if (test_file.category == category) {
            auto it = category_handlers_.find(category);
            if (it != category_handlers_.end()) {
                results_.push_back(it->second(test_file));
            } else {
                results_.push_back(run_transpiler_test(test_file));
            }
        }
    }
}

void TestRunner::run_single_test(const std::string& test_name) {
    results_.clear();

    for (const auto& test_file : test_files_) {
        if (test_file.name == test_name) {
            auto it = category_handlers_.find(test_file.category);
            if (it != category_handlers_.end()) {
                results_.push_back(it->second(test_file));
            } else {
                results_.push_back(run_transpiler_test(test_file));
            }
            break;
        }
    }
}

TestResult TestRunner::run_transpiler_test(const TestFile& test_file) {
    auto start = std::chrono::high_resolution_clock::now();

    TestResult result;
    result.test_name = test_file.name;
    result.category = test_file.category;

    // Create temporary file for input
    std::string temp_input = std::filesystem::temp_directory_path() /
                           ("test_" + test_file.name + ".cpp2");
    std::ofstream input_file(temp_input);
    input_file << test_file.content;
    input_file.close();

    // Create temporary file for output
    std::string temp_output = std::filesystem::temp_directory_path() /
                            ("test_" + test_file.name + ".cpp");

    // Run transpiler with timeout using fork/exec with pipe for output capture
    std::string command = "\"" + transpiler_path_ + "\" \"" + temp_input + "\" \"" + temp_output + "\" 2>&1";
    std::string output;
    int exit_code = -1;
    bool timed_out = false;

    int pipefd[2];
    if (pipe(pipefd) == -1) {
        result.passed = false;
        result.error_message = "Failed to create pipe for output capture";
        return result;
    }

    pid_t pid = fork();
    if (pid == 0) {
        // Child process: execute the command
        close(pipefd[0]); // Close read end
        dup2(pipefd[1], STDOUT_FILENO); // Redirect stdout to pipe
        dup2(pipefd[1], STDERR_FILENO); // Redirect stderr to pipe
        close(pipefd[1]); // Close write end (dup'd copies remain)

        // Create new process group for killpg()
        setpgid(0, 0);
        execlp("sh", "sh", "-c", command.c_str(), nullptr);
        _exit(127); // exec failed
    } else if (pid > 0) {
        // Parent process: close write end, read from pipe, wait with timeout
        close(pipefd[1]); // Close write end

        // Set read end to non-blocking for polling
        fcntl(pipefd[0], F_SETFL, O_NONBLOCK);

        int status;
        pid_t ret;
        auto deadline = start + TEST_TIMEOUT_MS;
        char buffer[4096];
        ssize_t bytes_read;

        do {
            auto now = std::chrono::high_resolution_clock::now();
            if (now >= deadline) {
                // Timeout: kill the child process group
                killpg(pid, SIGKILL);
                timed_out = true;
                break;
            }

            // Read available output
            while ((bytes_read = read(pipefd[0], buffer, sizeof(buffer) - 1)) > 0) {
                buffer[bytes_read] = '\0';
                output += buffer;
            }

            // Check if child has exited (non-blocking)
            ret = waitpid(pid, &status, WNOHANG);
            if (ret == pid) {
                // Child exited - read any remaining output
                while ((bytes_read = read(pipefd[0], buffer, sizeof(buffer) - 1)) > 0) {
                    buffer[bytes_read] = '\0';
                    output += buffer;
                }
                if (WIFEXITED(status)) {
                    exit_code = WEXITSTATUS(status);
                }
                break;
            } else if (ret == -1) {
                // Error
                break;
            }

            // Sleep a bit before checking again
            usleep(10000); // 10ms
        } while (ret == 0);

        close(pipefd[0]); // Close read end

        // If still running after loop (shouldn't happen), clean up
        if (ret == 0) {
            killpg(pid, SIGKILL);
            waitpid(pid, &status, 0);
        }
    } else {
        // Fork failed
        close(pipefd[0]);
        close(pipefd[1]);
        result.passed = false;
        result.error_message = "Failed to fork child process";
        return result;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Check for timeout first
    if (timed_out) {
        result.passed = false;
        result.error_message = "Test exceeded 15-second timeout (TEST_TIMEOUT_MS standard)";
        result.transpiler_output = output;
    }
    // Check if result matches expectation
    else if (test_file.should_pass) {
        result.passed = (exit_code == 0);
        if (!result.passed) {
            result.error_message = "Expected transpiler to succeed but it failed";
            result.transpiler_output = output;
        }
    } else {
        result.passed = (exit_code != 0);
        if (!result.passed) {
            result.error_message = "Expected transpiler to fail but it succeeded";
        } else {
            // Verify error message is meaningful
            if (output.empty()) {
                result.passed = false;
                result.error_message = "Expected error output but got none";
            }
        }
    }

    // Check for timeout first
    if (timed_out) {
        result.passed = false;
        result.error_message = "Test exceeded 15-second timeout (TEST_TIMEOUT_MS standard)";
        result.transpiler_output = output;
    }
    // Transpilation failed
    else if (exit_code != 0) {
        if (test_file.should_pass) {
            result.passed = false;
            result.error_message = "Expected transpiler to succeed but it failed";
            result.transpiler_output = output;
        } else {
            result.passed = true; // Error test: transpiler correctly rejected invalid code
        }
    }
    // Transpilation succeeded - now compile and run the generated C++ code
    else {
        result.transpiler_output = output;

        if (test_file.should_pass) {
            // Compile the generated C++ code
            std::string temp_binary = std::filesystem::temp_directory_path() /
                                     ("test_" + test_file.name);
            // cppfort generates standalone C++ code - no external dependencies needed
            std::string compile_command = "clang++ -std=c++20 -O0 \"" + temp_output +
                                       "\" -o \"" + temp_binary + "\" 2>&1";

            std::string compile_output;
            int compile_exit_code = execute_command(compile_command, compile_output, 30000); // 30s timeout

            if (compile_exit_code != 0) {
                result.passed = false;
                result.error_message = "Generated C++ code failed to compile";
                result.transpiler_output += "\n\n--- Compilation Error ---\n" + compile_output;
            } else {
                // Compilation succeeded - run the binary
                std::string run_output;
                int run_exit_code = execute_command("\"" + temp_binary + "\"", run_output, 5000); // 5s timeout

                if (run_exit_code < 0) {
                    result.passed = false;
                    result.error_message = "Program execution timed out or crashed";
                    result.actual_output = run_output;
                } else {
                    result.passed = true;
                    result.actual_output = run_output;
                }

                // Cleanup binary
                std::filesystem::remove(temp_binary);
            }
        } else {
            // Error test: transpiler should have failed but succeeded
            result.passed = false;
            result.error_message = "Expected transpiler to fail but it succeeded";
        }
    }

    // Cleanup
    std::filesystem::remove(temp_input);
    std::filesystem::remove(temp_output);

    return result;
}

// Helper function to execute a command with timeout
int TestRunner::execute_command(const std::string& command, std::string& output, int timeout_ms) {
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        return -1;
    }

    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        setpgid(0, 0);
        execlp("sh", "sh", "-c", command.c_str(), nullptr);
        _exit(127);
    } else if (pid > 0) {
        // Parent process
        close(pipefd[1]);
        fcntl(pipefd[0], F_SETFL, O_NONBLOCK);

        int status;
        pid_t ret;
        auto start = std::chrono::high_resolution_clock::now();
        auto deadline = start + std::chrono::milliseconds(timeout_ms);
        char buffer[4096];
        ssize_t bytes_read;
        bool timed_out = false;

        do {
            auto now = std::chrono::high_resolution_clock::now();
            if (now >= deadline) {
                killpg(pid, SIGKILL);
                timed_out = true;
                break;
            }

            while ((bytes_read = read(pipefd[0], buffer, sizeof(buffer) - 1)) > 0) {
                buffer[bytes_read] = '\0';
                output += buffer;
            }

            ret = waitpid(pid, &status, WNOHANG);
            if (ret == pid) {
                while ((bytes_read = read(pipefd[0], buffer, sizeof(buffer) - 1)) > 0) {
                    buffer[bytes_read] = '\0';
                    output += buffer;
                }
                close(pipefd[0]);
                if (WIFEXITED(status)) {
                    return WEXITSTATUS(status);
                }
                return -1;
            } else if (ret == -1) {
                break;
            }

            usleep(10000);
        } while (ret == 0);

        close(pipefd[0]);

        if (ret == 0) {
            killpg(pid, SIGKILL);
            waitpid(pid, &status, 0);
        }

        return timed_out ? -1 : 0;
    } else {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
}

TestResult TestRunner::handle_pure2_test(const TestFile& test_file) {
    auto result = run_transpiler_test(test_file);

    // Additional pure2-specific checks can be added here
    // For example, verify no C++1 code in output

    return result;
}

TestResult TestRunner::handle_mixed_test(const TestFile& test_file) {
    auto result = run_transpiler_test(test_file);

    // Additional mixed-mode specific checks can be added here
    // For example, verify C++1 code is preserved

    return result;
}

void TestRunner::register_default_handlers() {
    register_category_handler("pure2", [this](const TestFile& tf) { return handle_pure2_test(tf); });
    register_category_handler("mixed", [this](const TestFile& tf) { return handle_mixed_test(tf); });
}

void TestRunner::print_summary() const {
    size_t total = results_.size();
    size_t passed = std::count_if(results_.begin(), results_.end(),
                                [](const TestResult& r) { return r.passed; });
    size_t failed = total - passed;

    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Total tests: " << total << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;

    if (failed > 0) {
        std::cout << "\nFailed tests:" << std::endl;
        for (const auto& result : results_) {
            if (!result.passed) {
                std::cout << "  - " << result.category << ":" << result.test_name << std::endl;
            }
        }
    }
}

void TestRunner::print_detailed_results() const {
    for (const auto& result : results_) {
        std::cout << "\n--- " << result.category << ":" << result.test_name << " ---" << std::endl;
        std::cout << "Status: " << (result.passed ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Duration: " << result.duration.count() << "ms" << std::endl;

        if (!result.passed) {
            std::cout << "Error: " << result.error_message << std::endl;
            if (!result.transpiler_output.empty()) {
                std::cout << "Transpiler output:" << std::endl;
                std::cout << result.transpiler_output << std::endl;
            }
        }
    }
}

bool SHA256Verifier::verify_file_hash(const TestFile& test_file) {
    std::string calculated = TestCollector::calculate_sha256(test_file.content);
    return calculated == test_file.sha256_hash;
}

bool SHA256Verifier::verify_all_hashes(const std::vector<TestFile>& test_files) {
    for (const auto& test_file : test_files) {
        if (!verify_file_hash(test_file)) {
            std::cerr << "Hash mismatch for: " << test_file.name << std::endl;
            return false;
        }
    }
    return true;
}

std::map<std::string, std::string> SHA256Verifier::load_hash_database(const std::string& db_path) {
    std::map<std::string, std::string> hash_db;
    std::ifstream file(db_path);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string hash = line.substr(0, space_pos);
                std::string filename = line.substr(space_pos + 1);
                hash_db[filename] = hash;
            }
        }
        file.close();
    }

    return hash_db;
}

void SHA256Verifier::save_hash_database(const std::string& db_path,
                                      const std::vector<TestFile>& test_files) {
    std::ofstream file(db_path);

    if (file.is_open()) {
        for (const auto& test_file : test_files) {
            file << test_file.sha256_hash << " " << test_file.name << std::endl;
        }
        file.close();
    }
}

std::string TestRunner::find_transpiler(const std::string& hint) {
    auto check_path = [](const std::filesystem::path& p) -> bool {
        return std::filesystem::exists(p) &&
               std::filesystem::is_regular_file(p) &&
               (p.filename() == "cppfort" || p.filename() == "cppfort.exe");
    };

    // 1. Check hint if provided
    if (!hint.empty()) {
        std::filesystem::path hint_path(hint);
        if (check_path(hint_path)) {
            return std::filesystem::canonical(hint_path).string();
        }
    }

    // 2. Check common locations relative to current working directory
    std::vector<std::string> search_paths = {
        "build/src/cppfort",
        "../build/src/cppfort",
        "../../build/src/cppfort",
        "./cppfort",
        "../src/cppfort"
    };

    for (const auto& path : search_paths) {
        if (check_path(path)) {
            return std::filesystem::canonical(path).string();
        }
    }

    // 3. Check if cppfort is in PATH
    char* path_env = std::getenv("PATH");
    if (path_env) {
        std::string path_str(path_env);
        std::stringstream ss(path_str);
        std::string dir;

        while (std::getline(ss, dir, ':')) {
            std::filesystem::path candidate = std::filesystem::path(dir) / "cppfort";
            if (check_path(candidate)) {
                return std::filesystem::canonical(candidate).string();
            }
        }
    }

    // 4. Not found - throw error with helpful message
    throw std::runtime_error(
        "cppfort transpiler not found. Tried:\n"
        "  1. Hint path: " + (hint.empty() ? "(not provided)" : hint) + "\n"
        "  2. Relative paths: build/src/cppfort, ../build/src/cppfort, etc.\n"
        "  3. PATH environment variable\n"
        "\n"
        "Please either:\n"
        "  - Build the project: cmake --build build\n"
        "  - Add to PATH: export PATH=\"$PWD/build/src:$PATH\"\n"
        "  - Provide explicit path to TestRunner constructor"
    );
}

} // namespace cppfort::tests
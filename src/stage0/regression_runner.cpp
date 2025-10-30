#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

namespace fs = std::filesystem;

namespace {

std::string quote(const fs::path& path) {
    std::string result = path.string();
    if (result.find(' ') != std::string::npos) {
        return "\"" + result + "\"";
    }
    return result;
}

std::string current_time_string() {
    const auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    char buffer[128];
    if (std::strftime(buffer, sizeof(buffer), "%c", std::localtime(&tt))) {
        return buffer;
    }
    return "unknown time";
}

int run_command(const std::string& command, std::ostream& log, int timeout_seconds = 0) {
    log << "    CMD: " << command << "\n";
    
    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        execl("/bin/bash", "bash", "-c", command.c_str(), nullptr);
        _exit(127); // exec failed
    } else if (pid > 0) {
        // Parent process
        int status = 0;
        bool completed = false;

        if (timeout_seconds > 0) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
            while (true) {
                pid_t rc = waitpid(pid, &status, WNOHANG);
                if (rc == pid) {
                    completed = true;
                    break;
                }
                if (rc == -1) {
                    log << "    -> waitpid failed\n";
                    return -1;
                }

                if (std::chrono::steady_clock::now() >= deadline) {
                    kill(pid, SIGKILL);
                    waitpid(pid, &status, 0);
                    log << "    -> TIMEOUT after " << timeout_seconds << " seconds\n";
                    return -1;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } else {
            if (waitpid(pid, &status, 0) == -1) {
                log << "    -> waitpid failed\n";
                return -1;
            }
            completed = true;
        }

        if (!completed) {
            return -1;
        }
        
        int rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
        if (rc != 0) {
            log << "    -> exit code " << rc << "\n";
        }
        return rc;
    } else {
        log << "    -> fork failed\n";
        return -1;
    }
}

std::string read_file(const fs::path& file) {
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <stage0_cli> <tests_dir> <patterns_path> <include_dir> [--verbose] [--capture-traces]\n";
        return 1;
    }

    fs::path stage0_cli = fs::absolute(argv[1]);
    fs::path tests_dir = fs::absolute(argv[2]);
    fs::path patterns_path = fs::absolute(argv[3]);
    fs::path include_dir = fs::absolute(argv[4]);
    const bool verbose = (argc >= 6 && std::string_view(argv[5]) == "--verbose");
    const bool capture_traces = (argc >= 7 && std::string_view(argv[6]) == "--capture-traces");

    if (!fs::exists(stage0_cli)) {
        std::cerr << "Error: stage0_cli not found at " << stage0_cli << "\n";
        return 1;
    }
    if (!fs::exists(tests_dir) || !fs::is_directory(tests_dir)) {
        std::cerr << "Error: tests directory not found at " << tests_dir << "\n";
        return 1;
    }
    if (!fs::exists(patterns_path)) {
        std::cerr << "Error: patterns file not found at " << patterns_path << "\n";
        return 1;
    }

    fs::current_path(tests_dir);

    fs::path log_path = tests_dir / "regression_log.txt";
    std::ofstream log(log_path, std::ios::trunc);
    if (!log) {
        std::cerr << "Error: unable to open log file at " << log_path << "\n";
        return 1;
    }

    log << "Regression test log - " << current_time_string() << "\n";
    log << "Using stage0_cli: " << stage0_cli << " transpile\n";
    log << "Patterns: " << patterns_path << "\n";
    log << "\n";

    std::vector<fs::path> test_files;
    for (const auto& entry : fs::directory_iterator(fs::current_path())) {
        if (entry.is_regular_file() && entry.path().extension() == ".cpp2") {
            test_files.push_back(entry.path().filename());
        }
    }
    std::sort(test_files.begin(), test_files.end());

    if (test_files.empty()) {
        log << "No .cpp2 tests found.\n";
        std::cout << "Regression runner: no tests discovered.\n";
        return 0;
    }

    const std::set<std::string> skip_tests = {
        "pure2-assert-expected-not-null.cpp2",
        "pure2-assert-optional-not-null.cpp2",
        "pure2-assert-shared-ptr-not-null.cpp2",
        "pure2-assert-unique-ptr-not-null.cpp2",
        "pure2-bounds-safety-pointer-arithmetic-error.cpp2",
        "pure2-bounds-safety-span.cpp2"
    };

    int num_tests = 0;
    int num_failures = 0;
    int num_skipped = 0;

    for (const auto& test : test_files) {
        const std::string filename = test.string();
        if (skip_tests.count(filename) != 0) {
            log << "Skipping " << filename << " (requires unimplemented safety features)\n\n";
            num_skipped++;
            continue;
        }

        const fs::path base = test.stem();
        const fs::path output_cpp = base.string() + ".cpp";
        const fs::path binary = base;
        const fs::path output_capture = "output_" + base.string() + ".txt";
        const fs::path expected_output = fs::path("test-results") / (base.string() + ".output");

        log << "Testing " << filename << "\n";
        num_tests++;

        const std::string transpile_cmd = (capture_traces ? "RBCURSIVE_CAPTURE=1 " : "") +
                                          quote(stage0_cli) + " transpile " +
                                          quote(test) + " " + quote(output_cpp) + " " + quote(patterns_path);
        if (run_command(transpile_cmd, log) != 0) {
            log << "  Transpile FAILED\n\n";
            num_failures++;
            fs::remove(output_cpp);
            continue;
        }
        log << "  Transpile OK\n";

        std::string compile_cmd = "g++ -std=c++20 -O0 -g -o " + quote(binary) + " " + quote(output_cpp);
        if (fs::exists(include_dir)) {
            compile_cmd += " -I" + quote(include_dir);
        }
        if (run_command(compile_cmd, log) != 0) {
            log << "  Compile FAILED\n\n";
            num_failures++;
            fs::remove(output_cpp);
            fs::remove(binary);
            continue;
        }
        log << "  Compile OK\n";

        const std::string run_cmd = quote(fs::absolute(binary)) + " > " + quote(fs::absolute(output_capture)) + " 2>&1";
        if (run_command(run_cmd, log) != 0) {
            log << "  Run FAILED\n\n";
            num_failures++;
            fs::remove(output_cpp);
            fs::remove(binary);
            fs::remove(output_capture);
            continue;
        }
        log << "  Run OK\n";

        if (fs::exists(expected_output)) {
            const std::string actual = read_file(output_capture);
            const std::string expected = read_file(expected_output);
            if (actual == expected) {
                log << "  Output matches expected\n";
            } else {
                log << "  Output does not match expected\n";
                if (verbose) {
                    log << "    Expected:\n" << expected << "\n";
                    log << "    Actual:\n" << actual << "\n";
                }
                num_failures++;
            }
        } else {
            log << "  No expected output to compare\n";
        }

        log << "\n";
        fs::remove(output_cpp);
        fs::remove(binary);
        fs::remove(output_capture);
    }

    log << "Total tests: " << num_tests << "\n";
    log << "Failures: " << num_failures << "\n";
    log << "Skipped: " << num_skipped << "\n";

    std::cout << "Regression summary: "
              << num_tests << " run, "
              << num_failures << " failed, "
              << num_skipped << " skipped.\n";
    std::cout << "Log saved to " << log_path << "\n";

    return (num_failures == 0) ? 0 : 1;
}

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
#include <sstream>

#include <array>
#include <cstdio>
#include <cctype>

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

static std::string run_cmd_capture(const std::string& cmd, int& exit_code) {
    std::array<char, 256> buffer;
    std::string result;
    std::string safe_cmd = cmd;
    if (safe_cmd.find("< /dev/null") == std::string::npos) {
        safe_cmd += " < /dev/null";
    }
    FILE* pipe = popen(safe_cmd.c_str(), "r");
    if (!pipe) {
        exit_code = -1;
        return result;
    }
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    exit_code = pclose(pipe);
    return result;
}

// Direct transpilation copied from regression_runner.cpp
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
        cppfort::stage0::CPP2Emitter emitter;
        cppfort::stage0::OrbitIterator iterator = *result;
        std::ostringstream oss;
        emitter.emit(iterator, source, oss, pattern_loader.patterns());
        std::string output = oss.str();

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

// Direct C++ compilation based on compile_direct in regression_runner.cpp
int compile_direct(const fs::path& cpp_file,
                   const fs::path& exe_file,
                   std::ostream& log) {
    try {
        log << "    Direct compile: " << cpp_file << " -> " << exe_file << "\n";

        std::string cmd = "c++ -std=c++20 -I include " + cpp_file.string() +
                 " -o " + exe_file.string() + " < /dev/null 2>&1";

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

// Direct execution
int execute_direct(const fs::path& exe_file, std::ostream& log) {
    try {
        log << "    Direct execute: " << exe_file << "\n";

        std::string cmd = exe_file.string() + " < /dev/null 2>&1";

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

static std::string sanitize_commit_hex(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (isalnum((unsigned char)c)) out.push_back(c);
    }
    return out;
}

// Compute a hash for a string using git hash-object if available, else fallback to std::hash
static std::string compute_content_hash(const std::string& content) {
    int rc;
    // Try using git
    std::string cmd = "git hash-object --stdin";
    std::array<char, 256> buffer;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "w");
    if (pipe) {
        fwrite(content.data(), 1, content.size(), pipe);
        int exit_code = pclose(pipe);
        // git hash-object prints to stdout only when reading from stdin via pipe; using popen with "w" doesn't capture output. Skip this fallback.
    }

    // Fallback: use std::hash
    std::hash<std::string> hasher;
    auto h = hasher(content);
    std::ostringstream ss;
    ss << std::hex << h;
    return ss.str();
}

// Generate reference output for a given input file using the provided ref compiler (cppfront)
static bool generate_reference_output(const fs::path& ref_compiler,
                                      const fs::path& input_file,
                                      const fs::path& out_path,
                                      std::ostream& log) {
    if (!fs::exists(ref_compiler)) {
        log << "    Reference compiler not found: " << ref_compiler << "\n";
        return false;
    }
    std::string cmd = ref_compiler.string() + " " + input_file.string() + " -o " + out_path.string() + " 2>&1";
    int rc;
    std::string out = run_cmd_capture(cmd, rc);
    if (!out.empty()) log << "    ref: " << out << "\n";
    return (rc == 0 && fs::exists(out_path));
}

static bool files_equal(const fs::path& a, const fs::path& b) {
    if (!fs::exists(a) || !fs::exists(b)) return false;
    std::ifstream fa(a, std::ios::binary);
    std::ifstream fb(b, std::ios::binary);
    std::istreambuf_iterator<char> ita(fa), ite;
    std::istreambuf_iterator<char> itb(fb);
    return std::equal(ita, ite, itb);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <repo_root> <patterns> <stage0_cli> [--limit N] [filter]" << "\n";
        return 1;
    }

    fs::path repo_root = argv[1];
    fs::path patterns_path = argv[2];
    fs::path stage0_cli = argv[3];
    std::string filter;
    int limit = -1;
    fs::path ref_compiler_path;
    fs::path ref_cache_dir = fs::current_path() / "reference_cache";
    bool persist_outputs = false;
    bool ref_refresh = false;

    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--limit" && i + 1 < argc) {
            limit = std::stoi(argv[++i]);
        } else if (a == "--verbose") {
            // ignore
        } else if (a == "--reference" && i + 1 < argc) {
            ref_compiler_path = argv[++i];
        } else if (a == "--reference-cache" && i + 1 < argc) {
            ref_cache_dir = argv[++i];
        } else if (a == "--reference-refresh") {
            ref_refresh = true;
        } else if (a == "--persist-outputs") {
            persist_outputs = true;
        } else {
            filter = a;
        }
    }

    if (!fs::exists(repo_root)) {
        std::cerr << "Repo root does not exist: " << repo_root << "\n";
        return 1;
    }

    std::cout << "Scanning git repository: " << repo_root << " at " << current_time_string() << "\n";
    std::cout << "Patterns: " << patterns_path << "\n";
    std::cout << "Filter: " << (filter.empty() ? "(none)" : filter) << "\n";
    std::cout << "Commit limit: " << (limit < 0 ? "(none)" : std::to_string(limit)) << "\n";

    const fs::path log_path = fs::current_path() / "regression_git_log.txt";
    std::ofstream log(log_path, std::ios::app);
    if (!log) {
        std::cerr << "Error: Cannot open log file: " << log_path << "\n";
        return 1;
    }

    log << "\n================================================================================\n";
    log << "Regression (git) test run at " << current_time_string() << "\n";
    log << "Repo: " << repo_root << "\n";
    log << "================================================================================\n\n";

    // Gather commits
    std::string rev_list_cmd = "git -C " + repo_root.string() + " rev-list --all";
    int rc;
    std::string rev_output = run_cmd_capture(rev_list_cmd, rc);
    if (rc != 0) {
        std::cerr << "Failed to list git commits" << std::endl;
        return rc;
    }

    std::istringstream revs(rev_output);
    std::vector<std::string> commits;
    for (std::string line; std::getline(revs, line);) {
        if (!line.empty()) commits.push_back(line);
    }

    if (commits.empty()) {
        std::cout << "No commits found in repo" << std::endl;
        return 0;
    }

    if (limit > 0 && static_cast<int>(commits.size()) > limit) {
        commits.resize(limit);
    }

    // Collect (commit, path) for files under regression-tests with .cpp2
    std::set<std::pair<std::string, std::string>> pairs;

    for (const auto& commit : commits) {
        std::string ls_cmd = "git -C " + repo_root.string() + " ls-tree -r --name-only " + commit;
        std::string list_out = run_cmd_capture(ls_cmd, rc);
        if (rc != 0) continue;
        std::istringstream lss(list_out);
        for (std::string path; std::getline(lss, path);) {
            if (path.find("regression-tests/") == 0 && path.size() > 10 && path.substr(path.size()-5) == ".cpp2") {
                if (!filter.empty()) {
                    std::string filename = fs::path(path).filename().string();
                    if (filename.find(filter) == std::string::npos) continue;
                }
                pairs.emplace(commit, path);
            }
        }
    }

    if (pairs.empty()) {
        std::cout << "No regression tests found in git history" << std::endl;
        return 0;
    }

    int num_tests = 0;
    int num_passed = 0;
    int num_failed = 0;

    fs::path tmp_dir = fs::temp_directory_path() / "cppfort_regression_git";
    fs::create_directories(tmp_dir);

    for (const auto& pr : pairs) {
        const std::string& commit = pr.first;
        const std::string& path = pr.second;
        const std::string filename = fs::path(path).filename().string();
        num_tests++;

        log << "Testing: " << commit << ":" << path << "\n";

        // Extract file contents
        std::string show_cmd = "git -C " + repo_root.string() + " show " + commit + ":" + path + " 2>/dev/null";
        std::string content = run_cmd_capture(show_cmd, rc);
        if (rc != 0 || content.empty()) {
            log << "  WARNING: cannot extract file at commit " << commit << " -> skipping\n\n";
            num_failed++;
            continue;
        }

        // Write to temp file
        std::string commit_hex = sanitize_commit_hex(commit);
        fs::path tmp_cpp2 = tmp_dir / (commit_hex + "-" + filename);
        std::ofstream tmp_out(tmp_cpp2, std::ios::binary);
        if (!tmp_out) {
            log << "  ERROR: cannot write temp file " << tmp_cpp2 << "\n\n";
            num_failed++;
            continue;
        }
        tmp_out << content;
        tmp_out.close();

        // Do transpile + optionally compare with reference + compile + exec
        const fs::path output_cpp = tmp_cpp2.parent_path() / (tmp_cpp2.stem().string() + ".cpp");
        if (transpile_direct(tmp_cpp2, output_cpp, patterns_path, log) != 0) {
            log << "  Transpile FAILED\n\n";
            num_failed++;
            continue;
        }

        // If a reference compiler is provided, get or generate the reference output (memoized)
        if (!ref_compiler_path.empty()) {
            try {
                // Read source to compute a stable key (commit + file path is used) — use commit hash + path to cache
                std::string cache_key = sanitize_commit_hex(commit) + "-" + fs::path(path).filename().string();
                fs::create_directories(ref_cache_dir);
                fs::path cached_ref = ref_cache_dir / (cache_key + ".cpp");
                if (ref_refresh && fs::exists(cached_ref)) {
                    fs::remove(cached_ref);
                }
                if (!fs::exists(cached_ref)) {
                    log << "  Generating reference output via: " << ref_compiler_path << "\n";
                    if (!generate_reference_output(ref_compiler_path, tmp_cpp2, cached_ref, log)) {
                        log << "  WARNING: Could not generate reference output for: " << filename << "\n";
                    }
                } else {
                    log << "  Using cached reference output: " << cached_ref << "\n";
                }

                if (fs::exists(cached_ref)) {
                    if (files_equal(output_cpp, cached_ref)) {
                        log << "  Reference output matches our transpile -> ISOMORPHISM OK\n";
                    } else {
                        log << "  Reference output differs from our output -> ISOMORPHISM MISMATCH\n";
                        // Log the diff using git --no-pager --no-index for readable diff; capture and attach to log
                        int dcrc;
                        std::string diff_cmd = "git --no-pager --no-index diff --color -U3 -- " + cached_ref.string() + " " + output_cpp.string();
                        std::string diff_out = run_cmd_capture(diff_cmd, dcrc);
                        if (!diff_out.empty()) log << diff_out << "\n";
                    }
                }
            } catch (...) {
                log << "  EXCEPTION: Reference generation/compare failed for " << filename << "\n";
            }
        }

        bool expect_compile_fail = (filename.find("-error") != std::string::npos || filename.find("-fail") != std::string::npos);

        const fs::path output_exe = output_cpp.parent_path() / (output_cpp.stem().string() + ".exe");
        int compile_rc = compile_direct(output_cpp, output_exe, log);
        if (expect_compile_fail) {
            if (compile_rc != 0) {
                log << "  Compile failed as expected -> PASS\n\n";
                num_passed++;
            } else {
                log << "  Compile succeeded but failure expected -> FAIL\n\n";
                num_failed++;
            }
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

        int exec_rc = execute_direct(output_exe, log);
        if (exec_rc == 0) {
            log << "  Execution PASSED\n\n";
            num_passed++;
        } else {
            log << "  Execution FAILED\n\n";
            num_failed++;
        }

        if (!persist_outputs) {
            fs::remove(tmp_cpp2);
            fs::remove(output_cpp);
            fs::remove(output_exe);
        }
    }

    log << "================================================================================\n";
    log << "Results: " << num_passed << "/" << num_tests << " passed, " << num_failed << " failed\n";
    log << "================================================================================\n";

    std::cout << "\n================================================================================\n";
    std::cout << "Results: " << num_passed << "/" << num_tests << " passed, " << num_failed << " failed\n";
    std::cout << "Log written to: " << log_path << "\n";
    std::cout << "================================================================================\n";

    return (num_failed > 0) ? 1 : 0;
}

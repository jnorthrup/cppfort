// Test to enforce a few coding and policy requirements from CODING_STANDARDS.md and ARCHITECTURE.md
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <regex>
#include <cassert>

int main() {
    namespace fs = std::filesystem;
#ifdef REPO_ROOT
    fs::path repo_root = fs::path(REPO_ROOT);
#else
    fs::path repo_root = fs::current_path();
#endif

    std::vector<std::string> prohibited_patterns = {"popen(", "system(\"", "fork("};
    std::vector<std::string> whitelist_dirs = {"build", "_deps", "third_party", "third-party", ".git", ".github", "build_clean", "build_ninja", "build_ninja_test", "build_sanitizer", "build_test_check", "build_makes_panic"};

    bool found_violation = false;
    for (auto& p : fs::recursive_directory_iterator(repo_root / "src")) {
        if (!p.is_regular_file()) continue;
        std::string s = p.path().string();
        // Skip known vendor/build directories
        bool skip = false;
        for (auto& w : whitelist_dirs) {
            if (s.find(std::string("/") + w + std::string("/")) != std::string::npos) { skip = true; break; }
        }
        if (skip) continue;

        std::ifstream file(p.path());
        if (!file.good()) continue;
        // Skip self and other known non-sensical code locations that contain policy snippets
        if (p.path().filename() == "test_code_compliance.cpp") continue;
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        // Search for popen/system/fork invocations without redirecting stdin to /dev/null
        std::smatch m;
        for (auto& pat : prohibited_patterns) {
            size_t pos = content.find(pat);
            if (pos != std::string::npos) {
                // Special handling: system(...) should include `< /dev/null` to avoid blocking.
                if (pat == "system(\"") {
                    // Find the start of the system( invocation and the next closing )
                    size_t start = content.find("system(", pos);
                    size_t end = content.find(")", start);
                    if (end == std::string::npos) end = start + 64;
                    std::string call = content.substr(start, end - start + 1);
                    if (call.find("< /dev/null") == std::string::npos) {
                        std::cerr << "Policy violation: found 'system(' in " << s << " without '< /dev/null' redirect.\n";
                        found_violation = true;
                    }
                } else if (content.find("< /dev/null") == std::string::npos) {
                    std::cerr << "Policy violation: found '" << pat << "' in " << s << " without '< /dev/null' redirect.\n";
                    found_violation = true;
                }
            }
        }
    }

    if (found_violation) {
        assert(!found_violation); // Fail the test
    }

    // Check for source files outside `src/`, `include/`, `tests/` or allowed paths
    // 'scripts' is intentionally not allowed per CODING_STANDARDS.md; all scripts should be removed
    std::vector<std::string> allowed_top_dirs = {"src", "include", "tests", "docs", ".github", "build", "third_party", "third-party", "patterns"};
    std::vector<std::string> source_ext = {".cpp", ".c", ".cc", ".h", ".hpp"};
    std::vector<fs::path> bad_sources;
    for (auto& p : fs::recursive_directory_iterator(repo_root)) {
        if (!p.is_regular_file()) continue;
        std::string path = p.path().string();
        // skip whitelist dirs
        bool skip = false;
        for (auto& w : whitelist_dirs) {
            if (path.find(std::string("/") + w + std::string("/")) != std::string::npos) { skip = true; break; }
        }
        if (skip) continue;
        // If file ends with one of source extensions and is not under allowed top dirs -> fail
        for (auto& ext : source_ext) {
            if (path.size() >= ext.size() && path.substr(path.size() - ext.size()) == ext) {
                bool allowed = false;
                for (auto& d : allowed_top_dirs) {
                    if (path.find(std::string("/") + d + std::string("/")) != std::string::npos) { allowed = true; break; }
                }
                if (!allowed) bad_sources.push_back(p.path());
            }
        }
    }
    if (!bad_sources.empty()) {
        std::cerr << "Found source files outside allowed locations:\n";
        for (auto& b : bad_sources) std::cerr << " - " << b << "\n";
        assert(bad_sources.empty());
    }

    std::cout << "test_code_compliance: OK\n";
    return 0;
}

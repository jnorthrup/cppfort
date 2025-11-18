#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <unordered_set>

int main() {
    namespace fs = std::filesystem;
    std::vector<std::string> script_exts = {".sh", ".py", ".bat"};
    std::vector<fs::path> found;
        #ifdef REPO_ROOT
        fs::path repo_root = fs::path(REPO_ROOT);
        #else
        fs::path repo_root = fs::current_path();
        #endif
    // Only scan code/repo areas where scripts are considered a violation.
    std::vector<fs::path> search_paths = {repo_root / "src", repo_root / "include", repo_root / "tests", repo_root / "scripts", repo_root};
    std::unordered_set<std::string> skip_dirs = {"build", "_deps", "third_party", "third-party", ".git", ".github",
                                                 "tests/regression-tests", "tests/passthrough-tests", "tests/regression-tests/test-results"};

    for (const auto& base : search_paths) {
        if (!fs::exists(base)) continue;
        for (auto& p : fs::recursive_directory_iterator(base)) {
            if (!p.is_regular_file()) continue;
            // Skip files under known vendor or build directories
            bool skip = false;
            for (const auto& sd : skip_dirs) {
                if (p.path().string().find(std::string("/") + sd + std::string("/")) != std::string::npos) {
                    skip = true;
                    break;
                }
            }
            if (skip) continue;
            auto ext = p.path().extension().string();
            if (std::find(script_exts.begin(), script_exts.end(), ext) != script_exts.end()) {
                // File considered a script
                found.push_back(p.path());
            }
        }
    }

    // If any script-like files exist, we fail the test to signal coding standard violation
    if (!found.empty()) {
        std::cerr << "Found script-like files that violate CODING_STANDARDS.md:\n";
        for (auto& p : found) {
            std::cerr << " - " << p << "\n";
        }
        assert(found.empty()); // Force test failure
    }
    std::cout << "test_no_repo_scripts: OK\n";
    return 0;
}

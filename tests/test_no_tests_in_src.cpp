// Ensure no ad-hoc test files exist under `src/` except explicitly whitelisted items
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <cassert>

int main() {
    namespace fs = std::filesystem;
#ifdef REPO_ROOT
    fs::path repo_root = fs::path(REPO_ROOT);
#else
    fs::path repo_root = fs::current_path();
#endif

    // Allowlist: intentionally placed/required tests under src/
    std::vector<std::string> allow = {"src/stage0/test_timeout_verification.cpp"};

    std::vector<fs::path> violations;
    for (auto& p : fs::recursive_directory_iterator(repo_root / "src")) {
        if (!p.is_regular_file()) continue;
        std::string fn = p.path().filename().string();
        if (fn.rfind("test_", 0) == 0 && p.path().extension() == ".cpp") {
            std::string rel = fs::relative(p.path(), repo_root).generic_string();
            bool ok = false;
            for (auto& a : allow) if (a == rel) { ok = true; break; }
            // Allow any tests under `src/stage0/` which are part of the stage0 library
            if (!ok) {
                if (rel.rfind("src/stage0/", 0) == 0) ok = true;
            }
            if (!ok) violations.push_back(p.path());
        }
    }
    if (!violations.empty()) {
        std::cerr << "Found test files under src/ which are not allowed:\n";
        for (auto& v : violations) std::cerr << " - " << v << "\n";
        assert(violations.empty());
    }
    std::cout << "test_no_tests_in_src: OK\n";
    return 0;
}

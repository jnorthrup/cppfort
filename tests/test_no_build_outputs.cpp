// Fail if any committed build artifacts (CMakeCache, build.ninja, object files) exist in repo
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

    std::vector<std::string> build_indicators = {"CMakeCache.txt", "build.ninja", "CMakeFiles", ".o", ".obj", ".a", ".so", ".dll"};
    std::vector<fs::path> violations;
    std::vector<std::string> allowed_dirs = {"_deps", "third_party", "third-party", ".git", ".github", "build", "build_ninja", "build_clean", "build_test_check", "build_makes_panic", "build_ninja_test", "build_sanitizer"};

    for (auto& p : fs::recursive_directory_iterator(repo_root)) {
        if (!p.is_regular_file()) continue;
        std::string sp = p.path().string();
        // skip allowlist directories
        bool skip = false;
        for (auto& a : allowed_dirs) if (sp.find(std::string("/") + a + std::string("/")) != std::string::npos) { skip = true; break; }
        if (skip) continue;

        std::string fn = p.path().filename().string();
        for (auto& b : build_indicators) {
            if (fn == b) { violations.push_back(p.path()); break; }
            // object files or static libs
            if (p.path().extension() == b) { violations.push_back(p.path()); break; }
        }
    }
    if (!violations.empty()) {
        std::cerr << "Found committed build artifacts (please do not check these in):\n";
        for (auto& v : violations) std::cerr << " - " << v << "\n";
        assert(violations.empty());
    }
    std::cout << "test_no_build_outputs: OK\n";
    return 0;
}

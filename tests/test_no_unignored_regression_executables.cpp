// Fails if there are extensionless executable files in tests/regression-tests that are not covered by .gitignore
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <cassert>

int main() {
    namespace fs = std::filesystem;
#ifdef REPO_ROOT
    fs::path repo_root = fs::path(REPO_ROOT);
#else
    fs::path repo_root = fs::current_path();
#endif

    fs::path gitignore = repo_root / ".gitignore";
    if (!fs::exists(gitignore)) {
        std::cerr << ".gitignore not found at " << gitignore << "\n";
        assert(false);
    }

    std::ifstream in(gitignore);
    std::string gi((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    std::vector<std::string> allowed_exts = {".cpp2", ".cpp", ".h", ".md", ".txt", ".bat", ".sh"};
    std::set<std::string> violations;

    fs::path tests_root = repo_root / "tests" / "regression-tests";
    if (!fs::exists(tests_root)) {
        std::cout << "No regression tests directory present; skipping test." << std::endl;
        return 0;
    }

    for (auto &p : fs::recursive_directory_iterator(tests_root)) {
        if (!p.is_regular_file()) continue;
        auto ext = p.path().extension().string();
        if (!ext.empty()) continue; // ignore files with extensions
        // Check executable bit
        auto perms = fs::status(p).permissions();
        bool exec = (static_cast<unsigned>(perms) & static_cast<unsigned>(fs::perms::owner_exec)) ||
                    (static_cast<unsigned>(perms) & static_cast<unsigned>(fs::perms::group_exec)) ||
                    (static_cast<unsigned>(perms) & static_cast<unsigned>(fs::perms::others_exec));
        if (!exec) continue;

        // Check if .gitignore has an entry containing the filename or deeper path
        std::string rel = p.path().lexically_relative(repo_root).string();
        std::string base = p.path().filename().string();
        if (gi.find(rel) == std::string::npos && gi.find(base) == std::string::npos) {
            violations.insert(rel);
        }
    }

    if (!violations.empty()) {
        std::cerr << "Found extensionless executables in tests/regression-tests not covered by .gitignore:\n";
        for (auto &v : violations) std::cerr << " - " << v << "\n";
        assert(violations.empty());
    }

    std::cout << "test_no_unignored_regression_executables: OK\n";
    return 0;
}


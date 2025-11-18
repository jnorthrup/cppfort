// Ensures .gitignore contains patterns to prevent committing test-generated binaries and artifacts
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
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
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    std::vector<std::string> required = {
        "tests/regression-tests/**",
        "!/tests/regression-tests/**/*.cpp2",
        "!/tests/regression-tests/**/*.cpp",
        "!/tests/regression-tests/**/*.h",
        "!/tests/regression-tests/**/*.md",
        "!/tests/regression-tests/**/*.txt",
           "tests/regression-tests/**/*_out",
           "tests/**/test-results/",
           "tests/**/test-results/*",
           "tests/**/.dSYM/",
           "tests/**/simple_test*",
           "tests/**/run-tests.sh",
           "tests/**/rm-empty-files.*",
           "tests/**/regression_log.txt",
           "tests/**/regression_git_log.txt",
        "build*/",
    };

    std::vector<std::string> missing;
    for (auto &p : required) {
        if (content.find(p) == std::string::npos) missing.push_back(p);
    }

    if (!missing.empty()) {
        std::cerr << "Missing required .gitignore patterns:\n";
        for (auto &m : missing) std::cerr << " - " << m << "\n";
        assert(missing.empty());
    }

    std::cout << "test_gitignore_completeness: OK\n";
    return 0;
}


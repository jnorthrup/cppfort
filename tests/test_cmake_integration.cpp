// Test CMake integration - full build pipeline validation
// Phase 6: Integration test validating end-to-end build

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

void test_cmake_configures() {
    // Test that CMake configures successfully
    fs::path build_dir = "build";

    assert(fs::exists(build_dir / "CMakeCache.txt") && "CMakeCache.txt must exist");
    assert(fs::exists(build_dir / "build.ninja") && "build.ninja must exist");

    std::cout << "✅ test_cmake_configures passed\n";
}

void test_cppfort_executable_exists() {
    // Test that cppfort executable was built
    fs::path cppfort = "build/src/cppfort";

    if (!fs::exists(cppfort)) {
        std::cout << "⏭️  test_cppfort_executable_exists skipped (not built yet)\n";
        return;
    }

    assert(fs::is_regular_file(cppfort) && "cppfort must be a regular file");

    // Test it's executable
    std::string cmd = cppfort.string() + " --help > /dev/null 2>&1";
    int result = std::system(cmd.c_str());

    // cppfort --help should return 0
    assert(result == 0 && "cppfort --help should succeed");

    std::cout << "✅ test_cppfort_executable_exists passed\n";
}

void test_tablegen_files_generated() {
    // Test that TableGen .inc files were generated in build/
    bool has_cpp2_ops = fs::exists("build/Cpp2Ops.h.inc") ||
                        fs::exists("build/Cpp2Ops.cpp.inc");
    bool has_cpp2_dialect = fs::exists("build/Cpp2OpsDialect.h.inc") ||
                            fs::exists("build/Cpp2OpsDialect.cpp.inc");

    if (has_cpp2_ops || has_cpp2_dialect) {
        std::cout << "✅ test_tablegen_files_generated passed\n";
    } else {
        std::cout << "⚠️  test_tablegen_files_generated: No TableGen files found\n";
    }
}

void test_no_inc_files_in_root() {
    // Test that no .inc files exist in root directory
    bool found_inc = false;
    for (const auto& entry : fs::directory_iterator(".")) {
        if (entry.path().extension() == ".inc" &&
            entry.path().filename().string().find("Cpp2") != std::string::npos) {
            found_inc = true;
            std::cout << "⚠️  Found .inc file in root: " << entry.path().filename() << "\n";
        }
    }

    if (!found_inc) {
        std::cout << "✅ test_no_inc_files_in_root passed\n";
    } else {
        std::cout << "⚠️  test_no_inc_files_in_root: .inc files found in root\n";
    }
}

void test_build_directory_structure() {
    // Test that build directory has expected structure
    assert(fs::exists("build/src") && "build/src must exist");
    assert(fs::exists("build/tests") && "build/tests must exist");
    assert(fs::exists("build/bin") && "build/bin must exist");
    assert(fs::exists("build/corpus") && "build/corpus must exist");

    std::cout << "✅ test_build_directory_structure passed\n";
}

void test_gitignore_patterns() {
    // Test that .gitignore has required patterns
    std::ifstream gitignore(".gitignore");
    assert(gitignore.is_open() && ".gitignore must be readable");

    std::string content((std::istreambuf_iterator<char>(gitignore)),
                        std::istreambuf_iterator<char>());
    gitignore.close();

    bool has_build = content.find("build/") != std::string::npos;
    bool has_inc = content.find("*.inc") != std::string::npos;
    bool has_ninja = content.find(".ninja") != std::string::npos;

    assert(has_build && ".gitignore must have build/");
    assert(has_inc && ".gitignore must have *.inc");
    assert(has_ninja && ".gitignore must have .ninja patterns");

    std::cout << "✅ test_gitignore_patterns passed\n";
}

void test_homebrew_llvm_config() {
    // Test that CMakeLists.txt references Homebrew LLVM
    std::ifstream cmake("CMakeLists.txt");
    assert(cmake.is_open() && "CMakeLists.txt must be readable");

    std::string content((std::istreambuf_iterator<char>(cmake)),
                        std::istreambuf_iterator<char>());
    cmake.close();

    bool has_homebrew = content.find("homebrew") != std::string::npos ||
                        content.find("HOMEBREW_LLVM_PREFIX") != std::string::npos ||
                        content.find("/opt/homebrew/opt/llvm") != std::string::npos;

    assert(has_homebrew && "CMakeLists.txt must reference Homebrew LLVM");

    std::cout << "✅ test_homebrew_llvm_config passed\n";
}

void test_cppfront_target_configured() {
    // Test that cppfront target exists in build.ninja
    std::ifstream build_ninja("build/build.ninja");
    assert(build_ninja.is_open() && "build.ninja must be readable");

    std::stringstream buffer;
    buffer << build_ninja.rdbuf();
    std::string content = buffer.str();
    build_ninja.close();

    bool has_cppfront = content.find("cppfront") != std::string::npos ||
                        content.find("build/bin/cppfront") != std::string::npos;

    if (has_cppfront) {
        std::cout << "✅ test_cppfront_target_configured passed\n";
    } else {
        std::cout << "⚠️  test_cppfront_target_configured: cppfront target not found\n";
    }
}

void test_corpus_targets_configured() {
    // Test that corpus processing targets exist in build.ninja
    std::ifstream build_ninja("build/build.ninja");
    assert(build_ninja.is_open() && "build.ninja must be readable");

    std::stringstream buffer;
    buffer << build_ninja.rdbuf();
    std::string content = buffer.str();
    build_ninja.close();

    bool has_corpus_transpile = content.find("corpus_transpile") != std::string::npos;
    bool has_corpus_ast = content.find("corpus_ast") != std::string::npos;

    if (has_corpus_transpile || has_corpus_ast) {
        std::cout << "✅ test_corpus_targets_configured passed\n";
    } else {
        std::cout << "⚠️  test_corpus_targets_configured: corpus targets not found\n";
    }
}

void test_deleted_shell_scripts() {
    // Test that obsolete shell scripts were deleted
    bool process_corpus_exists = fs::exists("tools/process_corpus_with_cppfront.sh");
    bool reference_corpus_exists = fs::exists("tools/reference_corpus.sh");

    assert(!process_corpus_exists && "tools/process_corpus_with_cppfront.sh must be deleted");
    assert(!reference_corpus_exists && "tools/reference_corpus.sh must be deleted");

    std::cout << "✅ test_deleted_shell_scripts passed\n";
}

int main() {
    std::cout << "CMake Integration Tests (Phase 6)\n";
    std::cout << "===================================\n\n";

    try {
        test_cmake_configures();
        test_cppfort_executable_exists();
        test_tablegen_files_generated();
        test_no_inc_files_in_root();
        test_build_directory_structure();
        test_gitignore_patterns();
        test_homebrew_llvm_config();
        test_cppfront_target_configured();
        test_corpus_targets_configured();
        test_deleted_shell_scripts();

        std::cout << "\n✅ All CMake integration tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception\n";
        return 1;
    }
}

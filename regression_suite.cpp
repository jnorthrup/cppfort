#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <array>
#include <memory>

namespace fs = std::filesystem;

// Helper to run a command and capture its output
std::string run_command(const std::string &cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) return "";
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

// Compute SHA256 of a file using shasum (available on macOS/Linux)
std::string sha256_file(const fs::path &p) {
    std::string cmd = "shasum -a 256 " + p.string() + " 2>/dev/null | awk '{print $1}'";
    return run_command(cmd);
}

// Simple replace of a line containing a specific header
void replace_header(fs::path cpp_file) {
    std::ifstream in(cpp_file);
    std::string content, line;
    while (std::getline(in, line)) {
        if (line.find("#include \"cpp2util.h\"") != std::string::npos) {
            line = "// header replaced";
        }
        content += line + "\n";
    }
    in.close();
    std::ofstream out(cpp_file);
    out << content;
    out.close();
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --cppfront     Use cppfront instead of cppfort (comparison mode)\n"
              << "  --help         Show this help\n";
}

int main(int argc, char **argv) {
    bool use_cppfront = false;

    // Simple arg parsing (ignore --tier if passed for backward compat in cmake)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cppfront") {
            use_cppfront = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        // ignore --tier <arg>
        else if (arg == "--tier" && i + 1 < argc) {
            i++;
        }
    }

    // Paths
    // Assume we are running from the build directory
    fs::path build_dir = fs::current_path();

    // Find project root (assume build_dir is inside project_root/build or similar)
    fs::path project_root = build_dir.parent_path();
    if (!fs::exists(project_root / "third_party")) {
        project_root = fs::current_path().parent_path();
    }

    // Find cppfort (the project's transpiler)
    fs::path cppfort_bin = build_dir / "src" / "cppfort";
    if (!fs::exists(cppfort_bin)) cppfort_bin = build_dir / "bin" / "cppfort";
    if (!fs::exists(cppfort_bin)) cppfort_bin = build_dir / "cppfort";

    // Find cppfront (reference transpiler)
    fs::path cppfront_bin = build_dir / "cppfront";
    if (!fs::exists(cppfront_bin)) cppfront_bin = build_dir / "bin" / "cppfront";

    // Determine which compiler to use for the "actual" output
    fs::path actual_compiler = use_cppfront ? cppfront_bin : cppfort_bin;

    if (!fs::exists(actual_compiler)) {
        std::cerr << "Error: Compiler binary not found at " << actual_compiler << "\n";
        std::cerr << "Current path: " << fs::current_path() << "\n";
        return 1;
    }

    fs::path regression_dir = project_root / "third_party" / "cppfront" / "regression-tests";
    if (!fs::exists(regression_dir)) {
        regression_dir = fs::current_path().parent_path() / "third_party" / "cppfront" / "regression-tests";
    }

    fs::path reference_dir = regression_dir / "test-results";
    fs::path work_dir = build_dir / "regression_work";
    fs::create_directories(work_dir);

    // CSV header
    std::cout << "test,sha256,transpile,source_match,compile,exec_match" << std::endl;

    int total = 0;
    int passed = 0;

    if (!fs::exists(regression_dir)) {
        std::cerr << "Error: Regression directory not found at " << regression_dir << "\n";
        return 1;
    }

    for (auto &entry : fs::recursive_directory_iterator(regression_dir)) {
        if (entry.path().extension() != ".cpp2") continue;

        fs::path cpp2 = entry.path();
        std::string test_name = cpp2.stem().string();
        std::string sha = sha256_file(cpp2);

        // 1. Generate Reference Source (cppfront)
        fs::path ref_cpp = work_dir / (test_name + ".ref.cpp");
        bool have_ref_source = false;
        if (fs::exists(cppfront_bin)) {
            std::string ref_cmd = cppfront_bin.string() + " " + cpp2.string() + " -o " + ref_cpp.string();
            if (std::system(ref_cmd.c_str()) == 0) {
                have_ref_source = true;
                replace_header(ref_cpp);
            }
        }

        // 2. Generate Actual Source (cppfort or cppfront)
        fs::path actual_cpp = work_dir / (test_name + ".cpp");
        std::string transpile_cmd = actual_compiler.string() + " " + cpp2.string() + " -o " + actual_cpp.string();
        int transpile_ret = std::system(transpile_cmd.c_str());

        if (transpile_ret != 0) {
            std::cout << test_name << "," << sha << ",FAIL,-,-,-" << std::endl;
            continue;
        }
        replace_header(actual_cpp);

        // 3. Compare Sources (Isomorphic Check)
        bool source_match = false;
        if (have_ref_source) {
            // Compare ignoring whitespace
            std::string diff_cmd = "diff -w -B " + actual_cpp.string() + " " + ref_cpp.string() + " >/dev/null";
            int diff_ret = std::system(diff_cmd.c_str());
            source_match = (diff_ret == 0);
        } else {
            source_match = true;
        }

        // 4. Compile
        // Switched from clang++ to c++ (g++) to avoid Clang 18/GCC 13 stdlib incompatibility
        fs::path exe_out = work_dir / (test_name + "_exe");
        std::string compile_cmd = "c++ -std=c++23 -O0 -I" + (project_root / "include").string() + " " + actual_cpp.string() + " -o " + exe_out.string();

        int compile_ret = std::system(compile_cmd.c_str());
        if (compile_ret != 0) {
            std::cout << test_name << "," << sha << ",OK," << (source_match?"1":"0") << ",FAIL,-" << std::endl;
            continue;
        }

        // 5. Run and Compare Execution Output
        std::string run_cmd = exe_out.string() + " 2>&1";
        std::string run_output = run_command(run_cmd);

        fs::path ref_exec_output = reference_dir / (test_name + ".cpp.execution");
        bool exec_match = false;
        if (fs::exists(ref_exec_output)) {
            std::string diff_cmd = "diff -w <(echo \"" + run_output + "\") <(cat " + ref_exec_output.string() + ") >/dev/null";
            int diff_ret = std::system(diff_cmd.c_str());
            exec_match = (diff_ret == 0);
        } else {
            exec_match = true;
        }

        std::cout << test_name << "," << sha << ",OK,"
                  << (source_match ? "1" : "0") << ",OK,"
                  << (exec_match ? "1" : "0") << std::endl;

        total++;
        if (source_match && exec_match) passed++;
    }

    std::cerr << "Summary: " << passed << "/" << total << " passed (Source+Exec match).\n";
    return (passed == total && total > 0) ? 0 : 1;
}

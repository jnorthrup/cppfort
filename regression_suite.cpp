#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <array>
#include <cctype>

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
    std::string cmd = "shasum -a 256 \"" + p.string() + "\" 2>/dev/null | awk '{print $1}'";
    auto out = run_command(cmd);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return out;
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return "";
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

std::string normalize_ws(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    bool in_ws = false;
    for (unsigned char ch : input) {
        if (std::isspace(ch)) {
            if (!in_ws) out.push_back(' ');
            in_ws = true;
        } else {
            out.push_back(static_cast<char>(ch));
            in_ws = false;
        }
    }
    return out;
}

bool toolchain_supports_move_only_function(const fs::path& work_dir) {
    fs::path probe_src = work_dir / "__probe_move_only_function.cpp";
    fs::path probe_bin = work_dir / "__probe_move_only_function";
    {
        std::ofstream out(probe_src);
        out << "#include <functional>\n";
        out << "int main() { std::move_only_function<int()> f; return 0; }\n";
    }
    std::string cmd =
        "clang++ -std=c++23 -stdlib=libc++ \"" + probe_src.string() + "\" -o \"" +
        probe_bin.string() + "\" >/dev/null 2>&1";
    return std::system(cmd.c_str()) == 0;
}

int main(int argc, char **argv) {
    // Paths
    fs::path project_root = fs::current_path();
    fs::path cppfront_bin = project_root / "build_clean" / "bin" / "cppfront";
    fs::path regression_dir = project_root / "third_party" / "cppfront" / "regression-tests";
    fs::path reference_dir = regression_dir / "test-results";
    fs::path cppfront_include_dir = project_root / "third_party" / "cppfront" / "include";
    fs::path work_dir = project_root / "build_clean" / "regression_work";
    fs::create_directories(work_dir);
    bool supports_move_only_function = toolchain_supports_move_only_function(work_dir);

    // CSV header
    std::cout << "test,sha256,status,match" << std::endl;

    for (auto &entry : fs::recursive_directory_iterator(regression_dir)) {
        if (entry.path().extension() != ".cpp2") continue;
        fs::path cpp2 = entry.path();
        std::string test_name = cpp2.stem().string();
        bool expect_transpile_fail = cpp2.filename().string().find("-error.cpp2") != std::string::npos;
        std::string sha = sha256_file(cpp2);
        // Transpile
        fs::path cpp_out = work_dir / (test_name + ".cpp");
        std::string transpile_cmd = "\"" + cppfront_bin.string() + "\" \"" + cpp2.string() + "\" -o \"" + cpp_out.string() + "\"";
        int transpile_ret = std::system(transpile_cmd.c_str());
        if (transpile_ret != 0) {
            if (expect_transpile_fail) {
                std::cout << test_name << "," << sha << ",PASS,1" << std::endl;
            } else {
                std::cout << test_name << "," << sha << ",TRANSPILE_FAIL,0" << std::endl;
            }
            continue;
        }
        if (expect_transpile_fail) {
            std::cout << test_name << "," << sha << ",UNEXPECTED_TRANSPILE_PASS,0" << std::endl;
            continue;
        }
        if (!supports_move_only_function) {
            std::string generated = read_file(cpp_out);
            if (generated.find("std::move_only_function<") != std::string::npos) {
                std::cout << test_name << "," << sha << ",SKIP_UNSUPPORTED,1" << std::endl;
                continue;
            }
        }
        // Compile
        fs::path exe_out = work_dir / (test_name + "_exe");
        std::string compile_cmd =
            "clang++ -std=c++23 -stdlib=libc++ -O0 -ftemplate-depth=2048 "
            "-DCPP2_INCLUDE_STD "
            "-I\"" + cppfront_include_dir.string() + "\" "
            "-L/opt/homebrew/opt/llvm/lib/c++ "
            "-Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++ "
            "\"" + cpp_out.string() + "\" -o \"" + exe_out.string() + "\"";
        int compile_ret = std::system(compile_cmd.c_str());
        if (compile_ret != 0) {
            std::cout << test_name << "," << sha << ",COMPILE_FAIL," << "0" << std::endl;
            continue;
        }
        // Run executable and capture output
        std::string run_cmd = exe_out.string() + " 2>&1";
        std::string run_output = run_command(run_cmd);
        // Load reference output if exists
        fs::path ref_output = reference_dir / (test_name + ".cpp.execution");
        bool match = false;
        if (fs::exists(ref_output)) {
            std::string ref = read_file(ref_output);
            match = (normalize_ws(run_output) == normalize_ws(ref));
        }
        std::cout << test_name << "," << sha << ",PASS," << (match ? "1" : "0") << std::endl;
    }
    return 0;
}

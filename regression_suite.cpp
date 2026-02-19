#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <iomanip>

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
            // replace with an empty line (or appropriate alternative)
            line = "// header replaced";
        }
        content += line + "\n";
    }
    in.close();
    std::ofstream out(cpp_file);
    out << content;
    out.close();
}

int main(int argc, char **argv) {
    // Paths
    fs::path project_root = fs::current_path();
    fs::path cppfront_bin = project_root / "build_clean" / "bin" / "cppfront";
    fs::path regression_dir = project_root / "third_party" / "cppfront" / "regression-tests";
    fs::path reference_dir = regression_dir / "test-results";
    fs::path work_dir = project_root / "build_clean" / "regression_work";
    fs::create_directories(work_dir);

    // CSV header
    std::cout << "test,sha256,status,match" << std::endl;

    for (auto &entry : fs::recursive_directory_iterator(regression_dir)) {
        if (entry.path().extension() != ".cpp2") continue;
        fs::path cpp2 = entry.path();
        std::string test_name = cpp2.stem().string();
        std::string sha = sha256_file(cpp2);
        // Transpile
        fs::path cpp_out = work_dir / (test_name + ".cpp");
        std::string transpile_cmd = cppfront_bin.string() + " " + cpp2.string() + " -o " + cpp_out.string();
        int transpile_ret = std::system(transpile_cmd.c_str());
        if (transpile_ret != 0) {
            std::cout << test_name << "," << sha << ",TRANSPILE_FAIL," << "0" << std::endl;
            continue;
        }
        // Header replacement
        replace_header(cpp_out);
        // Compile
        fs::path exe_out = work_dir / (test_name + "_exe");
        std::string compile_cmd = "clang++ -std=c++23 -stdlib=libc++ -O0 -ftemplate-depth=2048 " + cpp_out.string() + " -o " + exe_out.string();
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
            std::string ref = run_command("cat " + ref_output.string());
            // Compare ignoring whitespace (diff -w style)
            std::string diff_cmd = "diff -w <(echo \"" + run_output + "\") <(cat " + ref_output.string() + ")";
            // Use system diff and capture exit status
            int diff_ret = std::system(diff_cmd.c_str());
            match = (diff_ret == 0);
        }
        std::cout << test_name << "," << sha << ",PASS," << (match ? "1" : "0") << std::endl;
    }
    return 0;
}

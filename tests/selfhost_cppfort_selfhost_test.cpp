#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <memory>
#include <stdexcept>

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

int main() {
    // Step 1: Run cppfort on the source file to generate C++ code
    std::string cppfort_path = "../build/src/selfhost/cppfort";
    std::string input_file = "../src/selfhost/cppfort.cpp2";
    std::string temp_cpp_file = "/tmp/cppfort_generated.cpp";

    std::string command = cppfort_path + " " + input_file + " > " + temp_cpp_file;
    int result = std::system(command.c_str());
    if (result != 0) {
        std::cerr << "Failed to run cppfort on " << input_file << std::endl;
        return 1;
    }

    // Step 2: Compile the generated C++ file
    std::string compile_command = "c++ -std=c++20 -x c++ " + temp_cpp_file + " -o /tmp/cppfort_generated";
    result = std::system(compile_command.c_str());
    if (result != 0) {
        std::cerr << "Failed to compile generated C++ file" << std::endl;
        return 1;
    }

    // Step 3: Run the generated executable and capture output
    std::string run_command = "/tmp/cppfort_generated 2>&1";
    std::string output = exec(run_command.c_str());

    // Extract the number of tags parsed from the output
    // We expect a line like: "Parsed X tags"
    std::istringstream iss(output);
    std::string line;
    int tags_parsed = -1;
    while (std::getline(iss, line)) {
        if (line.find("Parsed ") == 0 && line.find(" tags") != std::string::npos) {
            std::istringstream line_stream(line);
            std::string prefix;
            line_stream >> prefix >> tags_parsed; // prefix is "Parsed", tags_parsed is the number
            break;
        }
    }

    if (tags_parsed <= 1) {
        std::cerr << "Self-hosting test failed: only parsed " << tags_parsed << " tags. Expected more than 1." << std::endl;
        std::cerr << "Output was:\n" << output << std::endl;
        return 1;
    }

    std::cout << "Self-hosting test passed: parsed " << tags_parsed << " tags." << std::endl;
    return 0;
}
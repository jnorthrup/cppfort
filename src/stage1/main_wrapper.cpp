// Wrapper to provide proper argc/argv to cpp2 transpiler
#include <cstdlib>
#include <string>

// Forward declare the cpp2 main function
int cpp2_main();

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.cpp2> <output.cpp>\n";
        return 1;
    }

    // Set environment variables for cpp2 code to read
    setenv("CPP2_ARGC", std::to_string(argc).c_str(), 1);
    setenv("CPP2_ARG1", argv[1], 1);
    setenv("CPP2_ARG2", argv[2], 1);

    // Call the cpp2 main
    return cpp2_main();
}
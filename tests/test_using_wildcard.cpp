#include <cstdlib>

int main() {
    // Run the transpiler on the targeted corpus input and return non-zero on failure
    int rc = system("${CMAKE_BINARY_DIR}/../build/src/cppfort corpus/inputs/mixed-bugfix-for-literal-as-nttp.cpp2 /tmp/out_test_using.cpp");
    // system() returns exit status << 8 on Unix, but for simplicity interpret non-zero as failure
    return (rc != 0) ? 1 : 0;
}

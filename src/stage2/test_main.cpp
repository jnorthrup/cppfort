#include <iostream>
#include "anticheat.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: anticheat <binary_path>\n";
        return 1;
    }
    const std::string path = argv[1];
    const std::string hash = cppfort::stage2::attest_binary(path);
    std::cout << "Attestation SHA‑256: " << hash << std::endl;
    return 0;
}
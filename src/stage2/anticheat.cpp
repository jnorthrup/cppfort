#include "anticheat.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <array>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstring>
#include <string.h>
#include <CommonCrypto/CommonDigest.h>

namespace cppfort::stage2 {

std::string attest_binary(const std::string& binary_path) {
    // Run objdump to obtain disassembly.
    std::string command = "objdump -d " + binary_path + " 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to run objdump on " << binary_path << std::endl;
        return {};
    }

    std::ostringstream disassembly;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        disassembly << buffer;
    }
    pclose(pipe);

    const std::string& disasm_str = disassembly.str();

    // Compute SHA‑256 hash of the disassembly using CommonCrypto.
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(reinterpret_cast<const unsigned char*>(disasm_str.data()),
              static_cast<CC_LONG>(disasm_str.size()),
              hash);

    // Convert hash to hex string.
    std::ostringstream hex;
    hex << std::hex << std::setfill('0');
    for (unsigned char byte : hash) {
        hex << std::setw(2) << static_cast<int>(byte);
    }
    return hex.str();
}

} // namespace cppfort::stage2
#pragma once

#include <string>

namespace cppfort::stage2 {

/// Compute a SHA‑256 attestation of the disassembly of a compiled binary.
/// @param binary_path Path to the compiled executable.
/// @return Hexadecimal SHA‑256 digest of the `objdump -d` output.
std::string attest_binary(const std::string& binary_path);

} // namespace cppfort::stage2
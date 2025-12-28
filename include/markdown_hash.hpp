#pragma once

#include <string>
#include <string_view>

namespace cpp2_transpiler {

// Compute SHA256 hash of markdown content using trim-and-concatenate algorithm
// Algorithm:
//   1. Split content into lines
//   2. Trim whitespace from each line (leading and trailing)
//   3. Concatenate all lines with single '\n' between each
//   4. Compute SHA256 hash
//   5. Return as 64-character lowercase hexadecimal string
std::string compute_markdown_hash(std::string_view content);

} // namespace cpp2_transpiler
